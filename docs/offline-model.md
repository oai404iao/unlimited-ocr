# 离线量化与部署

面向需要重复启动、分发模型或减少部署加载工作的使用者。先完成 [安装](setup.md)，
以下命令在仓库根目录运行。返回 [README](../README.md)。

## 1. 导出一次

在具有 AVX2/FMA、可运行当前 INT8 路径的机器上执行：

```sh
.venv/bin/python -m tools.export_quantized \
  --source models/unlimited-ocr-2 \
  --output models/unlimited-ocr-2-int8 \
  --threads 3
```

导出使用与原始加载相同的量化算法，不重新训练或改变量化精度。输出包含量化权重、
其余未量化参数、模型配置与 tokenizer；不会修改源目录或复制原始大权重分片。
导出阶段需要同时容纳模型及序列化权重，内存峰值可能高于单纯加载；应预留充足内存和磁盘。

**目标目录必须不存在**，不覆盖已有模型。完成标记 `quantized.json` 最后写入；
中断的目录不能作为已完成模型部署。重试时使用新的输出目录。

```text
models/unlimited-ocr-2-int8/
├── quantized.json       格式及 Torch 版本信息、完成标记
├── quantized.pt         量化 state_dict，含原始量化参数和非量化权重
├── config.json
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

示例使用的 `models/` 目录和 `.pt` 文件被 Git 忽略；其他输出位置应放在仓库外或显式忽略
整个目录。不要把导出文件、计时结果或部署机器信息提交到 Git。

## 2. 复制整个目录并加载

部署端需要相同的项目代码、依赖和模型目录。无需原始 BF16 权重或网络连接：

```sh
.venv/bin/python -m tools.build_native
.venv/bin/python -m unlimited_ocr.cli samples/smoke.png \
  --model-dir models/unlimited-ocr-2-int8 \
  --max-new-tokens 192 --output results/offline.json
```

Python 调用同样支持：

```python
from examples.inference import load_ocr, recognize

model, tokenizer = load_ocr(model_dir="models/unlimited-ocr-2-int8")
result = recognize(model, tokenizer, ["samples/smoke.png"], max_new_tokens=192)
print(result["text"])
```

离线加载使用内存映射读取权重，跳过 BF16→FP32 转换、量化尺度计算和 INT8 转换，
但**仍需初始化模型并为目标 CPU 重新打包**。导出文件不保存绑定某个 ISA 的打包内存，
因此无需为 AVX2、VNNI-YMM、VNNI-ZMM 分别导出。CPU 策略仍由 `UOCR_CPU_ISA` 决定。

## 限制与安全

- 格式使用 PyTorch 量化 `state_dict`，不是 pickled 模型对象，也不是通用 Hugging Face
  `save_pretrained` 模型。加载明确使用 `weights_only=True`；仍只接收可信来源的文件。
- 加载要求导出端与部署端的 Torch 基础版本一致，忽略 `+cpu` 等构建后缀；更换版本应重新
  导出并验证，不提供跨版本迁移。原生二进制始终在部署端重新构建。
- `--fp32`、`--bf16-reference`、`--vision-bf16` 需要原始权重，不能搭配离线 INT8 目录。
  对 INT8 权重解量化并不能恢复原始精度。
- `generic` 或缺少 AVX2 时，使用 INT8 存储、逐算子解量化为 FP32 的安全回退；不调用
  FBGEMM。它的算术和性能不同于动态 INT8 路径，输出需要独立验收。
- 使用内存映射时，不要覆盖或修改正在加载/使用的文件；更新部署应使用新的模型目录。

## 验证加载收益

启动收益不是每 token 的推理加速。在相同 ISA、线程和输入下，用新的进程串行比较：

```sh
.venv/bin/python -m benchmarks.benchmark_model samples/smoke.png \
  --model-dir models/unlimited-ocr-2 --repeats 3 --output results/original.json
.venv/bin/python -m benchmarks.benchmark_model samples/smoke.png \
  --model-dir models/unlimited-ocr-2-int8 --repeats 3 --output results/offline.json
```

比较 `load_seconds`，同时核对输出 token、文本、页数和结束原因。该字段还包含优化安装、
tokenizer 和输入准备；`--repeats` 重复的是生成，不是加载。评估启动稳定性应重复启动新进程，
区分操作系统文件缓存与模型内存复用，并记录峰值内存。结果只保存在本地。
