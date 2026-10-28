# 安装与模型数据

面向首次运行项目的使用者。返回 [README](../README.md)；安装完成后阅读 [推理用法](usage.md)。

## 1. 准备环境

- Linux x86-64；安装的 PyTorch 必须支持目标 CPU。无需 GPU 或 CUDA。
- Python 3.12，由 `uv` 管理虚拟环境。
- 支持 C++20 的 `g++`，用于构建可选原生内核及 ISA 探针。
- 建议预留至少 20 GiB 磁盘空间；两片原始权重约 6.69 GB，下载缓存和环境额外占用空间。
- 内存需求取决于页面数量、切片、输出长度和权重格式。建议 32 GiB RAM；
  不能只按 INT8 权重大小估算总内存，FP32 基线和长文档需要更多内存。

例如 Debian/Ubuntu 可安装 `git g++`，Arch Linux 可安装 `git gcc`。
`uv` 的安装方法见 [官方说明](https://docs.astral.sh/uv/getting-started/installation/)。
获取代码后进入仓库根目录，后续命令均在该目录执行：

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt
```

先从 CPU wheel 索引安装 Torch / TorchVision，再安装其余依赖。不要直接升级到未经测试的
Torch 版本：量化接口、TorchScript 和 FBGEMM 探针都依赖其具体实现。

## 2. 下载并校验模型

```sh
.venv/bin/python -m tools.fetch_weights
(cd models/unlimited-ocr-2 && sha256sum --check ../../upstream/weights.sha256)
```

下载脚本固定模型 revision，仅下载 JSON 和 safetensors 数据到
`models/unlimited-ocr-2/`，不会替换 `upstream/` 中的 Python 实现。目录应包含：

```text
models/unlimited-ocr-2/
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── config.json
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

这些是可恢复的下载数据，不进入 Git，不需要 Git LFS。中断后重新执行下载命令。
如果 Xet 下载路径有问题，可在新进程中尝试：

```sh
HF_HUB_DISABLE_XET=1 .venv/bin/python -m tools.fetch_weights
```

不要将当前实现与其他 revision 或其他模型的权重、配置、索引混用。

## 3. 构建并运行

```sh
.venv/bin/python -m tools.build_native
.venv/bin/python -m unlimited_ocr.cli samples/smoke.png --max-new-tokens 192
```

构建结果放在 `unlimited_ocr/` 内：`native.so`、`native_attention.so`，以及可选诊断探针
`isa_probe.so`。换机器或升级 PyTorch 后应重新构建，不复制虚拟环境和二进制。

不要给整个共享库添加 `-march=native`、`-mavx2` 或 `-mavx512*`；构建脚本仅在特定
helper 内启用 SIMD，以免破坏不支持这些指令的 CPU 的回退路径。没有可用原生库时，
归一化和注意力可退回 PyTorch，但完整原生内核测试需要先构建。

运行失败时，先确认工作目录、数据是否完整，以及依赖版本；选择具体加速路径的方法见
[硬件与量化](acceleration.md)。

需要在多次启动或多台机器上复用量化结果时，继续阅读 [离线量化与部署](offline-model.md)。
接收完整离线模型目录的部署端无需再下载原始 BF16 权重，但仍需本项目代码、依赖及本机原生构建。
