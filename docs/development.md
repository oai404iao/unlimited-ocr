# 开发指南

面向修改、验证或评估实现的开发者。运行模型请先看 [安装指南](setup.md) 和
[推理用法](usage.md)；机制与约束见 [实现说明](architecture.md)。返回 [README](../README.md)。

## 目录与版本控制边界

```text
unlimited_ocr/  CPU 运行代码和 CLI
native/         C++ 内核及 ISA 探针源码
upstream/       固定版本的第三方模型源码、授权与权重校验清单
tools/          下载、构建和合成样例生成工具
examples/       最小 Python 集成示例
benchmarks/     可复现的测速与性能探索代码
tests/          小规模、无需完整权重的单元测试
samples/        合成测试输入及预期文本，不是推理输出
docs/           按使用者/开发者任务拆分的说明
```

模型数据放 `models/`，私人输入放 `inputs/`，运行结果放 `results/`，私人记录放 `local/`。
这些目录、虚拟环境、缓存、二进制和凭据均被 Git 忽略。
**只提交通用代码、测试夹具和方法文档，不提交机器/局域网部署信息、运行结果或个人调优报告。**
不要用 `git add -f` 绕过这些边界。合成样例及人工真值是固定测试夹具，应保留版本管理。

本项目从源码根目录通过 `python -m ...` 运行；未引入安装包、发布流程或旧路径兼容入口。
修改时同时更新 Python 导入、模块命令和文档，不保留根目录脚本的转发副本。

## 构建与测试

按照 [安装指南](setup.md) 准备依赖后：

```sh
.venv/bin/python -m tools.build_native
.venv/bin/python -m pytest -q
.venv/bin/python -m unlimited_ocr.cli --help
.venv/bin/python -m examples.inference --help
```

单元测试使用小规模张量和临时权重，不需要完整模型下载。AVX-512 执行测试需要相应硬件，
缺少能力时应跳过，不能据此宣称已验证该 ISA 的执行。原生内核采用目标属性隔离 ISA，
不要全局加宽编译指令集。依赖升级后重建，并检查实际分派和回退。

下载数据后执行完整 smoke test，检查文本、页数和 EOS，而不只是进程退出状态：

```sh
.venv/bin/python -m examples.inference samples/smoke.png --max-new-tokens 192 --output results/smoke.json
```

`samples/smoke.txt` 是预期文字。框坐标可能因数值路径改变；合成样例不代表真实文档的
CER、表格、布局或长文档质量。修改数值计算时需要针对目标输入补充验收。

通常无需重新生成样例；如需调整夹具，显式提供可显示中文的字体：

```sh
.venv/bin/python -m tools.make_sample --font /path/to/CJK-font.ttc
```

## 测速

性能测量在同一台机器上串行执行，不与其他测试或推理同时运行。
改变 ISA 时使用新进程；比较时保持输入、切片、线程和 token 预算一致。

```sh
# 一次热身，随后报告测量中位数；加载时间单独记录
UOCR_CPU_ISA=auto .venv/bin/python -m benchmarks.benchmark_model samples/smoke.png --repeats 3 --output results/auto.json
UOCR_CPU_ISA=avx2 .venv/bin/python -m benchmarks.benchmark_model samples/smoke.png --repeats 3 --output results/avx2.json

# 定位阶段耗时；profile 有额外开销，不直接当作最终速度
.venv/bin/python -m unlimited_ocr.cli samples/smoke.png --profile --output results/profile.json

# 小规模内核/后端比较，不等于整页加速
.venv/bin/python -m benchmarks.benchmark --threads 3 --output results/kernels.json
.venv/bin/python -m benchmarks.benchmark_attention
.venv/bin/python -m benchmarks.benchmark_backends
```

`benchmarks.explore_cpu` 提供更广的优化组合实验，可用 `--help` 查看；不作为默认运行入口。
结果保存在本地 `results/`，不提交 Git。评估结论必须区分启动、视觉/prefill、decode、
峰值内存、输出长度与数值质量；不要把微基准收益或跨机器差异当作同机整页收益。

CLI 对照选项不是同一种基线：

| 参数 | 对照含义 |
|---|---|
| `--baseline` | gather decode、chunk64、统一线程，仍可使用量化权重 |
| `--dense-attention` | 恢复上游稠密注意力，仍使用相同量化权重 |
| `--bf16-reference` | 未量化 BF16 存储，每次线性计算前转 FP32，包含转换开销 |
| `--fp32` | 未量化 FP32 权重，需充足内存；不是低内存默认路径 |

## 提交改动

从 `main` 创建任务分支，明确暂存文件，运行与改动相关的检查，再 squash 合并回 `main`。
使用 Conventional Commits；本机绝对路径、SSH 别名、地址和性能产物不应进入提交。
改变模型版本时，应同步核对上游源码、数据索引、校验清单、许可证与数值测试。
