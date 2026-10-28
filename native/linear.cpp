// Portable tensor storage with FBGEMM runtime packing. No quantized ATen tensors.
// FBGEMM follows the installed libtorch_cpu ABI; rebuild after changing PyTorch.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <fbgemm/Fbgemm.h>
#include <fbgemm/QuantUtils.h>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace uocr {

struct PackedLinear : torch::CustomClassHolder {
  std::unique_ptr<fbgemm::PackBMatrix<int8_t>> packed;
  at::Tensor scales;
  std::optional<at::Tensor> bias;
  std::vector<int32_t> offsets, zeros;
  int64_t n, k;

  PackedLinear(at::Tensor weight, at::Tensor scale,
               std::optional<at::Tensor> bias_value) {
    TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "FBGEMM requires a supported CPU");
    TORCH_CHECK(weight.device().is_cpu() && weight.scalar_type() == at::kChar &&
                weight.dim() == 2, "weight must be a CPU int8 matrix");
    n = weight.size(0);
    k = weight.size(1);
    TORCH_CHECK(n > 0 && k > 0 && n <= INT32_MAX && k <= INT32_MAX / (255 * 128),
                "weight dimensions exceed supported INT32 accumulation bounds");
    TORCH_CHECK(scale.device().is_cpu() && scale.scalar_type() == at::kFloat &&
                scale.dim() == 1 && scale.numel() == n,
                "scales must be a CPU FP32 vector with one value per output");
    TORCH_CHECK(at::isfinite(scale).all().item<bool>() && scale.gt(0).all().item<bool>(),
                "scales must be finite and positive");
    if (bias_value.has_value()) {
      const auto& b = *bias_value;
      TORCH_CHECK(b.device().is_cpu() && b.scalar_type() == at::kFloat &&
                  b.dim() == 1 && b.numel() == n,
                  "bias must be a CPU FP32 vector with one value per output");
      bias = b.contiguous();
    }
    scales = scale.contiguous();
    auto contiguous = weight.contiguous();
    const auto* data = contiguous.const_data_ptr<int8_t>();
    offsets.resize(n);
    zeros.resize(n, 0);
    for (int64_t row = 0; row < n; ++row) {
      int32_t sum = 0;
      for (int64_t col = 0; col < k; ++col) sum += data[row * k + col];
      offsets[row] = sum;
    }
    packed = std::make_unique<fbgemm::PackBMatrix<int8_t>>(
        fbgemm::matrix_op_t::Transpose, k, n, data, k);
  }

  at::Tensor apply(const at::Tensor& input) {
    TORCH_CHECK(input.device().is_cpu() && input.scalar_type() == at::kFloat,
                "input must be a CPU FP32 tensor");
    TORCH_CHECK(input.dim() >= 1 && input.size(-1) == k,
                "input last dimension must match weight input features");
    const int64_t m = input.numel() / k;
    TORCH_CHECK(m <= INT32_MAX, "input row count exceeds INT32 range");
    auto shape = input.sizes().vec();
    shape.back() = n;
    auto output = at::empty(shape, input.options());
    if (m == 0) return output;
    const auto x = input.contiguous();
    const float* data = x.const_data_ptr<float>();
    float low = std::numeric_limits<float>::quiet_NaN();
    float high = low;
    fbgemm::FindMinMax(data, &low, &high, x.numel());
    TORCH_CHECK(std::isfinite(low) && std::isfinite(high),
                "input quantization range must be finite");
    // Dynamic Linear's reduce_range=True uses one activation range per tensor.
    const auto q = fbgemm::ChooseQuantizationParams(low, high, 0, 127);
    auto scratch = at::empty(shape, input.options().dtype(at::kInt));
    const int tasks = at::get_num_threads();
    at::parallel_for(0, tasks, 1, [&](int64_t begin, int64_t end) {
      fbgemm::PackAWithQuantRowOffset<uint8_t> a(
          fbgemm::matrix_op_t::NoTranspose, m, k, data, k, nullptr,
          q.scale, q.zero_point);
      fbgemm::DoNothing<float, float> end_op;
      fbgemm::ReQuantizeForFloat<false, fbgemm::QuantizationGranularity::OUT_CHANNEL> post(
          end_op, q.scale, scales.const_data_ptr<float>(), q.zero_point,
          zeros.data(), a.getRowOffsetBuffer(), offsets.data(),
          bias ? bias->const_data_ptr<float>() : nullptr, n);
      for (int64_t task = begin; task < end; ++task) {
        fbgemm::fbgemmPacked(a, *packed, output.mutable_data_ptr<float>(),
                             scratch.mutable_data_ptr<int32_t>(), n, post,
                             task, tasks);
      }
    });
    return output;
  }
};

at::Tensor linear(const at::Tensor& x, const c10::intrusive_ptr<PackedLinear>& packed) {
  return packed->apply(x);
}

using PackedList = c10::List<c10::intrusive_ptr<PackedLinear>>;

struct PackedExperts : torch::CustomClassHolder {
  PackedList gates, ups, downs;

  PackedExperts(PackedList gate_values, PackedList up_values, PackedList down_values)
      : gates(std::move(gate_values)), ups(std::move(up_values)), downs(std::move(down_values)) {
    TORCH_CHECK(gates.size() > 0 && gates.size() == ups.size() && gates.size() == downs.size(),
                "gate/up/down expert lists must have matching nonzero lengths");
  }
};

at::Tensor moe(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& weights,
               const c10::intrusive_ptr<PackedExperts>& experts) {
  const auto& gates = experts->gates;
  const auto& ups = experts->ups;
  const auto& downs = experts->downs;
  TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kFloat &&
              x.dim() == 2 && x.size(0) == 1,
              "MoE requires a single CPU FP32 token [1, hidden]");
  TORCH_CHECK(ids.device().is_cpu() && ids.scalar_type() == at::kLong &&
              ids.dim() == 2 && ids.size(0) == 1 && ids.size(1) > 0,
              "expert ids must be CPU int64 [1, selected_experts]");
  TORCH_CHECK(weights.device().is_cpu() && weights.sizes() == ids.sizes() &&
              (weights.scalar_type() == at::kFloat || weights.scalar_type() == at::kDouble),
              "expert weights must be CPU FP32/FP64 with the ids shape");
  auto selected = ids.contiguous();
  const auto* index = selected.const_data_ptr<int64_t>();
  for (int64_t slot = 0; slot < ids.size(1); ++slot) {
    const int64_t expert = index[slot];
    TORCH_CHECK(expert >= 0 && expert < static_cast<int64_t>(gates.size()),
                "expert index outside packed expert lists");
    auto gate = gates.get(expert);
    auto up = ups.get(expert);
    auto down = downs.get(expert);
    TORCH_CHECK(gate->k == x.size(1) && up->k == x.size(1) && gate->n == up->n &&
                down->k == gate->n && down->n == x.size(1),
                "expert gate/up/down dimensions do not match hidden/intermediate widths");
  }
  std::vector<at::Tensor> outputs;
  outputs.reserve(ids.size(1));
  for (int64_t slot = 0; slot < ids.size(1); ++slot) {
    const int64_t expert = index[slot];
    auto gate = gates.get(expert)->apply(x);
    auto up = ups.get(expert)->apply(x);
    auto hidden = at::silu(gate).mul(up);
    outputs.push_back(downs.get(expert)->apply(hidden));
  }
  // Keep the original concatenation, cast, multiplication and reduction order.
  // In particular, routing weights can request an FP64 multiply and sum.
  auto values = at::cat(outputs, 0).view({1, ids.size(1), -1}).to(weights.scalar_type());
  return values.mul_(weights.unsqueeze(-1)).sum({1}).to(x.scalar_type());
}

} // namespace uocr

TORCH_LIBRARY(unlimited_ocr, m) {
  m.class_<uocr::PackedLinear>("PackedLinear")
      .def(torch::init<at::Tensor, at::Tensor, std::optional<at::Tensor>>());
  m.class_<uocr::PackedExperts>("PackedExperts")
      .def(torch::init<uocr::PackedList, uocr::PackedList, uocr::PackedList>());
  m.def("linear(Tensor input, __torch__.torch.classes.unlimited_ocr.PackedLinear packed) -> Tensor");
  m.def("moe(Tensor x, Tensor ids, Tensor weights, "
        "__torch__.torch.classes.unlimited_ocr.PackedExperts experts) -> Tensor");
}

TORCH_LIBRARY_IMPL(unlimited_ocr, CPU, m) {
  m.impl("linear", uocr::linear);
  m.impl("moe", uocr::moe);
}
