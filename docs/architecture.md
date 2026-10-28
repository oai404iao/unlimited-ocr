# 实现说明

面向修改运行时与内核的开发者。验证方法见 [开发指南](development.md)，策略配置见
[硬件与量化](acceleration.md)。返回 [README](../README.md)。

## 请求路径

```text
固定 revision 的数据 → 流式加载/量化 → 安装 CPU 优化
图片与提示 → 视觉编码 → 页感知稀疏 prefill → 逐 token decode → 结构化文本
```

第三方模型实现位于 `upstream/`，数据由工具下载到 `models/unlimited-ocr-2/`。
`unlimited_ocr.cli` 组合运行时、预处理和生成；`examples.inference` 展示同一模型多次串行调用。
不在此基础上额外抽象通用模型框架。

## 模块职责

下列 Python 模块均位于 `unlimited_ocr/`：

| 模块 | 职责 |
|---|---|
| `cpu_isa` | 在 Torch 初始化前检测共同 CPU 能力并设置后端策略 |
| `cpu_runtime` | meta 初始化、逐层读取 checkpoint、量化预打包、参考权重格式 |
| `cpu_attention` / `cpu_native_attention` | gathered 稀疏 prefill 和直接读取 KV 的原生 decode |
| `cpu_cache` | 请求级 KV 缓冲区，按几何容量扩展并原地追加 |
| `cpu_router` | 请求级已完成块摘要与增量路由统计 |
| `cpu_norm` | RMSNorm 原生实现和 PyTorch 回退 |
| `cpu_vision` / `cpu_sam_layout` | SAM 注意力分块与连续 BHWC 布局 |
| `cpu_moe_script` | 单 token 选中专家的脚本化执行，复用原 packed weights |
| `cpu_threads` | 按视觉、prefill、decode 阶段调整进程全局线程数 |
| `cpu_profile` | 阶段归因；专家计算仍计入 MoE exclusive 时间，并非纯调度开销 |
| `cpu_moe` / `cpu_bf16_vision` | 非默认的融合 MoE / BF16 SAM 实验 |

通用矩阵运算优先复用 FBGEMM 和 PyTorch SDPA。`native/` 只实现专用 RMSNorm、
稀疏 decode 和诊断探针；共享库使用基线安全编译，ISA 特化函数通过运行时检测调用。

## 必须保留的语义

- 保留上游 entmax 选择、SR-SWA/DASH、page-2D RoPE、远程块偏置和局部窗口语义。
  不得用普通稠密注意力或固定窗口注意力冒充等价实现。
- 稀疏路径只读取选中远程块和局部 token；原生 decode 直接读取 KV，避免 gather 副本。
- SAM query 分块仍让每个 query 关注所有 key，并保留相对位置偏置，不缩小感受野。
- KV 与路由摘要属于当前请求，新文档必须隔离状态；权重打包对象可在请求间复用。
- 脚本化 MoE 复用原 packed weights，不复制或重新量化；多 token prefill 保留原专家分组。
- FP32 路由与权重量化是不同边界，不能只为吞吐量改变选择逻辑。

## 明确限制

历史 KV 不淘汰，路由存储和评分随上下文增长；缓冲区复用不等于常量内存。
量化和浮点归约顺序变化可能影响坐标或文本，需要数值与真实输出共同验证。
安装补丁及阶段线程具有进程级影响，当前只支持串行、单请求、batch=1 推理，
不支持训练、beam search 或同进程并发服务。
