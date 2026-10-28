# 上游来源与本地修改

- 模型：[baidu/Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2)
- 固定 revision：`04647c7a6098d8d58915f003536a741c82eafb40`
- 本目录的模型实现来自该 revision；原模型卡保留在 `README.md`，图示等资源可在线查看。
- `python -m tools.fetch_weights` 仅下载模型数据（JSON / safetensors），不会替换本目录 Python 实现。
- 配置、索引、权重及 tokenizer 数据下载到 `models/unlimited-ocr-2/`，不进入 Git；
  权重校验值见 `weights.sha256`。

本地修改集中在 `modeling_unlimitedocr.py`：去除图像注入时的 CUDA 硬编码，使用
CPU FP32 图像输入，接入复用 KV cache，避免读取量化 `lm_head.weight` 属性，以及
使用单调计时。其余 CPU 优化主要由 `unlimited_ocr/cpu_*.py` 安装到模型实例或模块上。
缓存接入使用 `unlimited_ocr` 包导入。
空的 `__init__.py` 用于本地包导入。

上游模型卡标注 MIT；`_sparse_base_impl.py` 同时保留 DeepSeek-AI / HuggingFace 的
Apache-2.0 版权头，全文见 `LICENSE-APACHE-2.0.txt`。应保留各文件原有授权与版权声明，
不要把模型卡的单一标签视作所有组成代码的统一授权声明。
本说明不为仓库本地新增代码另行授予开源许可证。
