# 推理用法

面向运行模型或集成 Python 调用的使用者。先完成 [安装](setup.md)，所有命令在仓库根目录运行。
返回 [README](../README.md)。

## 命令行

```sh
# 仓库内置合成样例
.venv/bin/python -m unlimited_ocr.cli samples/smoke.png --max-new-tokens 192

# 私人图片放在 Git 忽略的 inputs/ 中
.venv/bin/python -m unlimited_ocr.cli inputs/page.png --max-new-tokens 2048 --output results/page.json

# 同一文档的多个页面，按顺序给出；token 预算针对整份输出
.venv/bin/python -m unlimited_ocr.cli inputs/page-1.png inputs/page-2.png --max-new-tokens 4096 --output results/document.json

# 使用已导出的离线模型；导出步骤见“离线量化与部署”
.venv/bin/python -m unlimited_ocr.cli inputs/page.png --model-dir models/unlimited-ocr-2-int8

# 完整参数说明
.venv/bin/python -m unlimited_ocr.cli --help
```

默认每页使用 1024×1024 global view。`--tiles` 开启上游动态切片，可能提高小字的可读性，
但增加计算和内存开销。多个图片路径表示一份多页文档，**不是并行 batch**。

输出 JSON 中常用字段：

| 字段 | 含义 |
|---|---|
| `text` | 含页面、区块、类型与坐标的模型文本，不是纯逐行文本 |
| `finish_reason` | `stop` 为 EOS 结束；`length` 为达到 token 上限，内容可能不完整 |
| `token_ids` | 生成的 token |
| `time` / `prefill_time` | 生成总时间 / 包含视觉编码、语言 prefill 和首 token 的时间 |
| `load_seconds` | 加载、量化、优化安装和输入准备的时间，与生成耗时分开 |
| `isa` / `weight_format` | 实际策略与权重格式，不能只根据请求参数判断是否启用了加速 |
| `attention_backend` / `norm_backend` | 实际注意力 / 归一化实现 |

`--model-dir` 同时指定权重、配置和 tokenizer 所在目录，支持原始 BF16 数据与本项目
[导出的离线 INT8 目录](offline-model.md)，不需要另外指定 tokenizer。

`--profile` 添加阶段计时，但有额外开销。合成样例用于检查运行链路；真实文档的文字、
布局、表格和坐标准确性仍需自行验收。数值路径变化可能改变框坐标。

## Python：加载一次，串行复用

可运行示例：

```sh
.venv/bin/python -m examples.inference samples/smoke.png --max-new-tokens 192
```

在仓库根目录运行自己的脚本：

```python
import os
os.environ["UOCR_CPU_ISA"] = "auto"  # 在导入 torch 或项目模块之前设置

from examples.inference import load_ocr, recognize

model, tokenizer = load_ocr(threads=3, vision_threads=6, decode_threads=3)

first = recognize(model, tokenizer, ["samples/smoke.png"], max_new_tokens=192)
print(first["text"])

second = recognize(model, tokenizer,
                   ["samples/smoke.png", "samples/smoke.png"], max_new_tokens=512)
print(second["finish_reason"], second["text"])
```

每次调用为新文档创建 KV cache；打包权重在模型实例内复用。`load_ocr()` 和线程初始化
每进程调用一次，不要每页重新加载模型。阶段线程和部分模型补丁是进程全局状态，
不支持在同一进程中并发调用不同请求或其他模型。

使用离线模型时，将初始化改为
`load_ocr(model_dir="models/unlimited-ocr-2-int8")`；示例命令行同样支持 `--model-dir`。

示例固定使用当前模型版本的内部生成接口，不是通用 Hugging Face 模型 API；需要完整
运行参数时使用命令行入口。有关量化、线程和 ISA 设置，参见 [硬件与量化](acceleration.md)。
