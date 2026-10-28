# 硬件、量化与加速选择

面向已跑通推理、需要调整性能或排查回退的使用者。返回 [README](../README.md)。
本页给出选择方法，不提供绑定某台机器的性能承诺。

## 检查硬件和实际分派

```sh
lscpu
lscpu -e=CPU,CORE,SOCKET,NODE
free -h

UOCR_CPU_ISA=auto .venv/bin/python - <<'PY'
import json
from unlimited_ocr.cpu_isa import isa_report
print(json.dumps(isa_report(), indent=2))
PY
```

AVX2 路径要求 `avx2` 和 `fma`；AVX-512 还要求 `avx512f/bw/dq/vl`。
`avx512_vnni` 和 `avx512_bf16` 分别表示相应的附加能力。
程序以所有逻辑 CPU 的共同能力为基准，不能仅凭处理器商品名判断。

`selected` 是原生内核 ISA；`fbgemm` 是矩阵内核配置；有可用探针时，`fbgemm_observed`
报告 FBGEMM 实际分派。它们不代表每个算子都使用同一种 ISA。

## 选择策略

| `UOCR_CPU_ISA` | 行为 |
|---|---|
| `auto` | 默认自动检测；具备 AVX512-VNNI 时使用 VNNI-YMM 策略 |
| `avx2` | 请求 AVX2 路径 |
| `avx512` | 请求基础 AVX-512，不启用 VNNI/BF16 策略 |
| `vnni256` / `vnni512` | 分别请求 VNNI-YMM / VNNI-ZMM 矩阵内核 |
| `generic` | 禁用本项目原生 SIMD 和 FBGEMM INT8，使用 PyTorch 通用计算 |

在新进程启动前设置，例如：

```sh
UOCR_CPU_ISA=avx2 .venv/bin/python -m unlimited_ocr.cli samples/smoke.png
```

请求硬件不支持的 ISA 会警告并降级；通用回退仍要求安装的 PyTorch 支持目标 CPU/OS，
不是指令集模拟器。不要在已经初始化 Torch 的进程或 notebook 中改变策略；应重启进程。
底层 ATen、oneDNN、FBGEMM 环境变量由策略统一设置，不要再单独强制覆盖。

`auto` 是默认策略，不保证任何 CPU 上都最快。需要比较时，在不同新进程里串行测速，
保持相同输入、token 预算和线程数，方法见 [开发指南](development.md#测速)。

## 加载时量化

默认 `load_model()` 自动完成量化，不需要校准数据或单独执行转换命令：

1. 在 meta device 创建模型，逐层读取 BF16 checkpoint，避免整模型 FP32 副本。
2. 当前线性层临时转 FP32，按输出通道做对称 INT8 权重量化，零点为 0；
   `scale = max(max(abs(weight_row)), 1e-8) / 127`。
3. FBGEMM 为目标 ISA 预打包权重；激活动态量化，输出仍为 FP32。

路由与 indexer 保留 FP32，Embedding 用 BF16 存储、FP32 输出；卷积和归一化等
其他模块不套用该线性层 INT8 转换。它不是整模型所有计算都变成 INT8。

原始权重不被修改。使用原始目录时，新进程会重新量化和打包；使用
[离线 INT8 目录](offline-model.md)时，跳过量化，只为当前 CPU 重新打包。
导出保存可移植的量化张量而非 CPU 特定的内存布局；它仍是本项目格式，不是通用 HF 权重。
进程内仍应复用模型实例，避免反复初始化和打包。

无 AVX2/FMA 或显式选择 `generic` 时，自动改用 BF16 权重存储、逐算子转 FP32 计算，
不再使用 FBGEMM INT8。这可能更慢且内存占用不同，注意查看 `weight_format`。

离线模型没有原始 BF16 权重；其 `generic` 回退保留 INT8 存储，每次线性计算时解量化为
FP32，不调用 FBGEMM。该路径不能恢复量化丢失的精度，也不等同于原始 BF16 参考基线。

## 线程与实验选项

默认视觉 / prefill / decode 线程为 6 / 3 / 3，可分别调整：

```sh
.venv/bin/python -m unlimited_ocr.cli samples/smoke.png --vision-threads 4 --threads 4 --decode-threads 2
```

线程数受物理核心、内存带宽、缓存和同步成本影响，更多线程或更宽 SIMD 不必然更快。
上述命令仅演示参数，不是推荐的硬件通用最优值。

- `--vision-bf16`：实验性原生 BF16 SAM，能力不足时回退；不默认开启。
- `--optimize-moe`：实验性 gate/up 融合，替代默认脚本化专家路径；不保证整页收益。
- `--vision-chunk`：视觉 query 分块大小，默认 256，影响内存/速度折中。

精度和性能基线选项的区别见 [开发指南](development.md#测速)。
