# Unlimited-OCR-2 · CPU 推理

在 Linux x86-64 CPU 上解析单页或多页文档图片，保留 Unlimited-OCR-2 的页感知稀疏注意力。
支持加载时 INT8 量化、AVX2 / AVX-512 / VNNI 自动选择，以及能力不足时的 PyTorch 回退。

**使用范围**：单进程串行推理，batch=1；输入为图片，输出为带页面、区块和坐标的结构化文本。
不提供 PDF 转换、HTTP 服务或训练接口。历史 KV 随上下文增长，不是常量内存的无限上下文实现。

## 快速开始 · 使用者

需要 Linux x86-64、`uv` 和支持 C++20 的 `g++`；以下命令均在仓库根目录运行。
权重约 6.69 GB，另需环境、缓存和推理内存。准备条件与下载校验见 [安装指南](docs/setup.md)。

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m tools.fetch_weights
.venv/bin/python -m tools.build_native
.venv/bin/python -m unlimited_ocr.cli samples/smoke.png --max-new-tokens 192
```

默认自动选择 CPU 路径并在加载时量化；输出写入 `results/ocr.json`。原始权重不会被修改。
重复部署可先导出离线 INT8 模型，后续加载跳过量化。查看下一步：

- [推理用法](docs/usage.md)：单页、多页、输出解释、Python 集成。
- [离线量化与部署](docs/offline-model.md)：导出一次，复用量化文件并减少加载工作。
- [硬件与量化](docs/acceleration.md)：检查 ISA、选择加速、线程设置和回退边界。

## 修改项目 · 开发者

- [开发指南](docs/development.md)：目录边界、构建、测试、测速和提交要求。
- [实现说明](docs/architecture.md)：模块职责、优化机制与必须保留的语义。
- [Python 示例](examples/inference.py)：模型只加载一次，串行处理多份文档。

## 模型来源与授权

模型来自 [baidu/Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2)，固定 revision
`04647c7a6098d8d58915f003536a741c82eafb40`。上游代码与数据的来源及授权说明见
[upstream/ORIGIN.md](upstream/ORIGIN.md)。本地新增代码尚未单独指定整体开源许可证。
