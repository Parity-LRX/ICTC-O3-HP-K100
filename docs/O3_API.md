# ICTC O3 算子 API

`ictc_o3/` 与 `mace_ictc/` 位于仓库同一层。前者提供独立算子，后者提供
MACE 兼容模型、训练器和转换接口。两个包随 `ictc-o3-hp-k100` 一起安装，
共用现有 HP 内核。使用独立算子无需创建 MACE 模型。

```python
from ictc_o3 import (
    EdgeTensorProduct, GraphMessage, PackedGraphMessage,
    direction_harmonics, create_symmetric_contraction,
    configure_symmetric_contraction, channel_linear, normalized_silu,
)
```

以上符号也可从 `ictc_o3.operators` 导入。旧路径 `mace_ictc.hp` 和
`mace_ictc.hp.operators` 导出同一组类与函数，保留现有调用和模块序列化兼容性。
仅导入 `ictc_o3` 不加载 MACE 模型、不初始化 GPU、不编译扩展。

## API 一览

| API | 调用 / 工厂 | 输入与输出 |
| --- | --- | --- |
| `EdgeTensorProduct` | `.create(channels=..., input_lmax=..., edge_lmax=..., output_lmax=..., layout="ir_mul", device="cuda")`；也可包装已有固定系数 TP | `op(features, harmonics, gates)` → `[E, op.output_width]` |
| `GraphMessage` | `(tp, input_lmax=..., edge_lmax=..., layout="packed", reduction="auto", edge_chunk_size=65536)` | `op(node_blocks, harmonic_blocks, gates, src, dst, mask=None)` → 节点路径块字典 |
| `PackedGraphMessage` | `.create(channels=..., input_lmax=..., edge_lmax=..., output_lmax=..., reduction="atomic", device="cuda")` | `op(features, harmonics, gates, src, dst, mask=None)` → `[N, op.output_width]` |
| `direction_harmonics` | `(unit_directions, lmax, policy="auto")` | `[E,3]` → 各阶 `[E,2l+1]` 的列表 |
| `create_symmetric_contraction` | `(*, num_elements=..., channels=..., lmax=..., target_lmax=..., correlation=..., training=...)` | 返回已配置的可训练 SC 模块 |
| `configure_symmetric_contraction` | `(sc, policy="auto", training=...)` | 原地选择已有 SC 的执行方案，返回选择结果 |
| `channel_linear` | `(features, weight, bias=None)` | `[*,Cin,m]` 与 `[Cout,Cin]` → `[*,Cout,m]` |
| `normalized_silu` | `(features, scale=1.0)` | 标量 / invariant 输入 → `SiLU(features) * scale` |

## 独立边张量积

```python
import torch
from ictc_o3 import EdgeTensorProduct, direction_harmonics

channels, lmax, edges = 32, 2, 257
tp = EdgeTensorProduct.create(
    channels=channels, input_lmax=lmax, edge_lmax=lmax,
    output_lmax=lmax, layout="ir_mul", device="cuda",
)
blocks = {
    l: torch.randn(edges, channels, 2*l+1, device="cuda",
                   dtype=torch.float32, requires_grad=True)
    for l in range(lmax+1)
}
vectors = torch.randn(edges, 3, device="cuda", dtype=torch.float32,
                      requires_grad=True)
harmonics = torch.cat(direction_harmonics(
    torch.nn.functional.normalize(vectors, dim=-1), lmax, policy="generated",
), dim=-1)
gates = torch.randn(edges, tp.gate_width, device="cuda",
                    dtype=torch.float32, requires_grad=True)
output = tp(tp.pack_input(blocks), harmonics, gates)
output_blocks = tp.unpack_output(output)
output.square().mean().backward()
```

`channels`、输入/边/输出 `lmax` 由调用方指定。`features` 的宽度为
`C*(input_lmax+1)**2`，`harmonics` 的宽度为 `(edge_lmax+1)**2`。
gates 支持 `[E, tp.gate_width]` 或 `[E, len(tp.paths), C]`，顺序以 `tp.paths`
为准。输出保留每条耦合路径，后续可用 `channel_linear` 混合路径通道。
`ir_mul` 在各 degree/path 内采用 `[magnetic,channel]`；`mul_ir` 采用
`[channel,magnetic]`。打包只改变存储布局，使用原生 ICTC 角基。

## 节点图消息

```python
from ictc_o3 import PackedGraphMessage

message = PackedGraphMessage.create(
    channels=32, input_lmax=2, edge_lmax=2, output_lmax=2,
    reduction="csr", edge_chunk_size=65536, device="cuda",
)
# features: [N, 32*9]，ir_mul；harmonics: [E,9]
# gates: [E, message.gate_width]；src/dst: [E] int64 CUDA
node_paths = message(features, harmonics, gates, src, dst)
```

src/dst 使用当前节点索引，可以重复或无序。调用方提供邻居表和实时径向 gates；
算子执行 gather、TP、gate 和 scatter reduction。边 TP 的 `pack_input` 也适用于
同规格的节点块。字典接口用 `GraphMessage(tp, input_lmax=..., edge_lmax=...)`。

## 覆盖与梯度

- 独立边 TP / 图消息使用 eager CUDA/HIP FP32；任意正整数通道数，degree 为非负整数。
  CSR 图前向的已验证覆盖为 L≤3；其他 degree 可使用已有 atomic 路径。
- 当前独立 TP 工厂的路径满足三角条件及 `l_in + l_edge + l_out` 为偶数，
  对应自然宇称 `p=(-1)^l`。它没有暴露任意 `(l, parity)` 输入的通用 O(3) TP。
- TP 基础系数固定，外部 gates 与输入可训练；包装已有 TP 会保留其系数，
  不会自动冻结可训练参数。新工厂使用原生 ICTC 归一化。
- 一阶和混合二阶导数使用标准 `torch.autograd.grad` / `loss.backward()`。
  调用 `autograd.grad(..., create_graph=True)` 可继续反传几何或力损失。
- SC 支持 native full-CG、correlation 1/2/3；提供 `node_type_idx` 使用高性能分解。
  方向谐波输入为单位方向；通道线性只允许标量角块添加 bias；SiLU 用于标量或 invariant。
- GPU HP 使用严格 FP32，禁用 TF32。首次规划和 JIT 属于冷启动开销。

完整参数、SC 输入约定、内存布局、高阶覆盖和导出路径见 [完整 API 指南](HP_API.md)。
整网构建和能量/力 API 从 `mace_ictc.hp` 导入，训练器从 `mace_ictc.training` 导入。
