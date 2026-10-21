# ICTC-O3-HP-K100 API

`ictc_o3` 提供独立算子 API，`mace_ictc.hp` 提供整网 API；二者是并列的
顶层 Python 包，随 `ictc-o3-hp-k100` 一起安装。算子速查见 [O3 API](O3_API.md)。
旧的 `mace_ictc.hp` 算子导入路径继续兼容；
`models/` 里的 `_source`、`_plan`、原始 CUDA launch 等内部函数无需由用户调用。
性能对照、测试与验证资料由维护者本地保存，不包含在交付源码和安装包中。

## API 一览

| API | 用途 | 输入 / 输出要点 |
|---|---|---|
| `EdgeTensorProduct.create(...)` | 新建独立边 TP | 任意正整数 C、CUDA FP32、非负 L、自然宇称、单位固定基础系数 |
| `EdgeTensorProduct(tp, ...)` | 包装已有 ICTC TP | 保留已有固定系数、路径和外部门控约定 |
| `GraphMessage(tp, ...)` | 真实图 gather/TP/gate/scatter | 接收节点/边 degree 字典、门控和重复 src/dst，返回节点路径块 |
| `PackedGraphMessage(tp, ...)` / `.create(...)` | 原生 packed 图 TP | 与 `EdgeTensorProduct(ir_mul)` 同布局；直接接收/返回二维缓冲及实时 src/dst |
| `direction_harmonics(n, L, policy=...)` | 笛卡尔方向谐波 | 单位方向 `[E,3]` -> 各阶 `[E,2l+1]` 的列表 |
| `configure_symmetric_contraction(sc, ...)` | 启用已有 SC 的高性能分解 | 原地选择执行路径；forward 继续使用原模块接口 |
| `create_symmetric_contraction(...)` | 新建高性能 SC | 可选 C/L/cor，native full-CG bridge-U，返回原类模块 |
| `channel_linear(x, W, bias=None)` | 共享通道 GEMM | `[*,Cin,m]`、`[Cout,Cin]` -> `[*,Cout,m]` |
| `normalized_silu(x, scale=1.0)` | 融合标量激活与导数 | 计算 `SiLU(x)*scale`，不自行选择模型归一化系数 |
| `create_model(...)` | 创建整网演示/新训练模型 | 可选 channels/lmax/edge_lmax/interaction 层数/cor，返回普通 ICTC 模型 |
| `check_model(model, ...)` | 只读检查整网覆盖 | 返回 `supported` 和 `reasons`，不编译/修改模型 |
| `configure_model(model, ...)` | 选择整网 eager 执行方案 | 返回选择结果、原因、算子计数和布局 |
| `energy_and_forces(model, ...)` | 已建图的完整能量/力计算 | 返回 `energy[B]`、`forces[N,3]`、`atomic_energy[N]` |
| `prepare_for_export(model, ...)` | 准备整网 FX/AOTI 导出 | 注册/准备可被导出降级的独立 Triton 算子；不直接生成文件 |

导入 API 不会触发 CUDA 编译。首次实际调用可能生成常量计划并 JIT 编译；冷启动
耗时不能计作热吞吐。生成源码与编译缓存都要指定到对应远端任务目录。
GPU HP 使用严格 FP32，没有 TF32/TF32x3 执行选项。配置整网、独立 TP/图消息、
通道 GEMM 和 SC 配置会将 `torch.set_float32_matmul_precision("highest")`、
matmul/cuDNN 的 `allow_tf32=False` 固定到进程中，不自动恢复此前设置。
这不会转换张量 dtype 或修改参数 trainability。自定义脚本、导出和其他库的调用
也应保持这三个设置；已编译的旧二进制需重新导出才能更新编译时算术选择。

## 1. 独立边 TP

新建自己的原生 ICTC 算子：

```python
import torch
from ictc_o3 import EdgeTensorProduct, direction_harmonics

op = EdgeTensorProduct.create(
    channels=96, input_lmax=3, edge_lmax=3, output_lmax=3,
    layout="ir_mul", device="cuda",
)
E = 257
C = op.tp.channels
blocks = {
    l: torch.randn(E, C, 2*l+1, device="cuda", requires_grad=True)
    for l in range(4)
}
direction = torch.nn.functional.normalize(torch.randn(E, 3, device="cuda"), dim=-1)
harmonics = torch.cat(direction_harmonics(direction, 3, policy="generated"), dim=-1)
gates = torch.randn(E, op.gate_width, device="cuda", requires_grad=True)

features = op.pack_input(blocks)
output = op(features, harmonics, gates)
output_blocks = op.unpack_output(output)
loss = output.square().mean()
loss.backward()
```

所有浮点输入显式使用 FP32；如果进程默认 dtype 被设为 float64，请在创建输入时
加 `dtype=torch.float32`。上面的门控是示例；真实调用应使用径向网络的实时输出。

稳定的形状约定：

- `features`：`[E, C*(input_lmax+1)^2]`，degree 按 0..L 排列。
- `harmonics`：`[E, (edge_lmax+1)^2]`，笛卡尔 degree 块按 0..L 排列。
- `gates`：`[E, P*C]` 或 `[E,P,C]`；**路径顺序以 `op.paths` 为准**。
- `output`：`[E, op.output_width]`；先按输出 degree，再按原 TP 的路径槽排列。
- `unpack_output` 返回 `{l: [E, path_count_l*C, 2*l+1]}`，保留路径 multiplicity。

`layout="ir_mul"` 在每个 degree/path 内排列 `[magnetic,channel]`；输出具体是
逐路径 `[E,path_count_l,2l+1,C]` 再展平。`layout="mul_ir"` 对应逐路径
`[E,path_count_l,C,2l+1]`；必要的输入/输出转换计入调用。
`pack_input`/`unpack_output` 仅转换内存布局，不旋转角基。

独立 TP 可显式选择旧生成式执行路线：

```python
op = EdgeTensorProduct.create(
    channels=64, input_lmax=3, edge_lmax=3,
    layout="mul_ir", execution="legacy",
)
```

`execution="legacy"` 使用原联合伴随调度，在 TP 的 autograd 边界内重算
`mul_ir` 输入和输出梯度的布局转换；`ir_mul` 继续直接写出原生缓冲。
它使用当前库的笛卡尔系数生成器及二阶导数实现，保留相同的输入、输出、路径和
固定系数约定。默认 `execution="direct"` 保持按梯度类型专化的执行路线。
已有 TP 包装构造也接受此参数。该选项仅控制独立 TP；整网使用 `configure_model`。

新工厂选用原生 ICTC 系数归一化，不会自动按 e3nn/cuEq 做路径尺度转换。
从已有模型抽取算子，应包装原 TP：

```python
op = EdgeTensorProduct(
    model.interactions[1].tp,
    input_lmax=model.lmax,
    edge_lmax=model.ictd_fix_edge_lmax,
    layout="ir_mul",
)
```

请按该层实际输入阶数填写 `input_lmax`；首层可能仅有 scalar 输入，不能一概填
整网 L。已有 TP 基础系数必须事先由模型架构设为固定；API 会拒绝仍可训练的系数，
不会替用户冻结它们。非单位固定系数按原值参与计算；门控仍是可训练的实时输入。
此 wrapper 用于独立调用，不是原模型 TP 的同签名替代物；自身保存状态有 `tp.*`
前缀。整网优化请使用 `configure_model`，它不替换原模块或改变 checkpoint keys。

独立边算子没有节点聚合，使用直接写出。普通一阶、混合二阶和既有高阶 autograd
路径保留；不支持直接把该 CUDA JIT wrapper 放进 make_fx、torch.compile、
functorch 或 autocast。整网 AOTI 使用第 6 节的 prepared 方案。

VJP 使用标准 PyTorch API，无需调用内部 CUDA adjoint：

```python
vectors = torch.randn(E, 3, device="cuda", dtype=torch.float32, requires_grad=True)
Y = torch.cat(direction_harmonics(torch.nn.functional.normalize(vectors, dim=-1), 3), -1)
y = op(features, Y, gates)
dx, dr, dg = torch.autograd.grad(
    y, (features, vectors, gates), grad_outputs=torch.randn_like(y), create_graph=True,
)
```

`create_graph=True` 保留继续对力/几何梯度求参数梯度所需的图。此例保留从方向谐波
到原始 edge vectors 的链式导数；实际位置/晶胞链路由用户图或整网 helper 提供。

## 2. 图消息聚合

需要真实图聚合时调用 `GraphMessage`，而不是把独立边 TP 当作节点消息：

```python
from ictc_o3 import GraphMessage

message = GraphMessage(
    tp, input_lmax=3, edge_lmax=3,
    layout="packed", edge_chunk_size=65536, reduction="auto",
)
# node_blocks[l]: [N,C,2l+1], C=tp.channels
# edge_blocks[l]: [E,1,2l+1]
# radial_gates: [E,tp.num_paths*C], live radial MLP output
# src/dst: [E] int64 CUDA, valid indices into the same N-node buffer
node_paths = message(node_blocks, edge_blocks, radial_gates, src, dst, mask=None)
```

source/destination 可以重复，真实图 reduction 仍累加到正确节点。路径块保留，后续
等变线性层可以按原模型混合路径。此 API 不计算径向 MLP，不构建邻居表。
可选 `mask` 为固定 FP32 `[E]` 或 `[E,1]`；可训练 mask 应使用原参考实现。
该 wrapper 的布局/分块选项属于自身，不改写共享 TP 的运行选择或参数。

`reduction="auto"` 在 C>=96、所有 degree<=3、E>=4096、平均每节点至少 16 条
输入边时启用 CUDA CSR 前向；其他图使用原 atomic 前向。
`reduction="atomic"` 固定原调度，`reduction="csr"` 强制 L<=3 的 CSR 前向。
CSR 在 GPU 上对实时 topology 建计划，保留原边数组顺序、不复制整块 gate；
每个 warp 在寄存器中累加邻居后写回节点。普通一阶与混合二阶导数继续使用
已有解析 adjoint。FP32 归约顺序会改变，结果不承诺逐位一致。

`EdgeTensorProduct` 的 `ir_mul/mul_ir` 和图/整网的 `packed/fused_pack` 是不同层次的
选项，不能互换。`fused_pack` 将图 shader 边界的 degree copy 合并，仍保留真实
gather/scatter、初始化和图 reduction。默认保持 `packed`。

## 3. 其他高性能算子

```python
from ictc_o3 import (
    direction_harmonics, create_symmetric_contraction, configure_symmetric_contraction,
    channel_linear, normalized_silu,
)

Y = direction_harmonics(unit_directions, 3, policy="generated")

sc = create_symmetric_contraction(num_elements=3, channels=32, lmax=2,
                                  target_lmax=2, correlation=2)
# Or keep an existing trained SC: sc = model.products[layer_index]
info = configure_symmetric_contraction(sc, policy="generated")
# sc: ICTDBridgeUSymmetricContractionSO3, CUDA FP32/native full CG
# x: [N, C*(L+1)^2], ICTC mul_ir degree blocks
# node_attrs: [N,num_elements] fixed one-hot, species_idx: [N] int64
value = sc(x, node_attrs, node_type_idx=species_idx)

mapped = channel_linear(block, W)  # block [N,Cin,2l+1], W [Cout,Cin]
activated_scalar = normalized_silu(scalar_features, scale=model_activation_scale)
```

SC 的原权重、parameter identity 与 checkpoint keys 保留。提供离散
`node_type_idx` 才启用 factorization；只给 `node_attrs` 时按原模块约定执行，
不会以 argmax 替换可微 species 输入。原保存 U 的重构检查仍在内核计划中。
SC 保留已验证的内部 e3nn basis bridge，不宣称重新生成纯笛卡尔高阶 U。

通道 GEMM 支持普通 autograd，包括矩阵梯度；bias 仅允许 scalar (`m=1`) 块。
SiLU 只可直接用于标量或模型原本允许的 invariant 输入；逐分量作用于非标量
角块一般破坏等变性。`scale` 是固定数值，API 不猜测模型的归一化常数。
方向谐波和 SC 的 `policy="auto"` 保留已建立的参考 fallback；显式 `generated`
要求覆盖，`reference` 选参考方案。GEMM/SiLU 的可微 CPU fallback 仍可用。

## 4. 整网建模与求力

新模型工厂只构建已初始化的网络，不加载已训练势，也不会自动选 HP plan：

```python
from mace_ictc.hp import create_model, check_model, configure_model, energy_and_forces

model = create_model(
    channels=96, lmax=2, edge_lmax=2, num_interactions=4, correlation=2,
    atomic_numbers=(1,6,8), cutoff=5.0,
    avg_num_neighbors=50.0, device="cuda", dtype=torch.float32,
)
print(check_model(model, training=True))
```

工厂参数可选，默认仍是 C64/L3/两层/cor3。新 TP 基础系数按架构固定，径向、
线性、SC 参数可训练；不会修改已有 checkpoint 的参数可训练性。
`channels` 和 `num_interactions` 为任意正整数，`lmax` 为非负整数，
`edge_lmax` 默认为 `lmax` 且不得小于它。快核按实际 C/路径/角分量数生成；
非 32 倍数的尾部通道有真实 load/store 掩码。
`correlation` 支持 2、3，兼容 1。
native full-CG/baseline bridge-U 保留，`use_reduced_cg=True` 不会启用 cuEq 基生成，
而是明确报错。它保留调用前的进程默认 dtype。
为动态 AOTI，工厂以 FX 构建 e3nn 子模块，并恢复调用前的 e3nn 优化默认值。
interaction 深度和角阶数没有固定为两层/L<=3。结构支持仍受原生模型基构造和
设备资源约束；尚未验证的高阶/宽网络组合不应被当作已验收性能配置。

求力推理：

```python
model.eval().requires_grad_(False)  # Explicit inference choice, not training.
info = configure_model(model, training=False, policy="generated", layout="packed")
with torch.no_grad():
    result = energy_and_forces(model, pos, Z, batch, src, dst, shifts, cell)
energy = result["energy"]          # [B]
forces = result["forces"]          # [N,3]
```

此处 `Z`/`batch`/`src`/`dst` 都为同设备 int64；`pos`/`shifts`/`cell` 为 FP32。
`shifts[E,3]` 是晶格坐标平移，`cell[B,3,3]` 是每个 graph 的晶胞。
几何和邻居表须已准备；helper 不添加 fitted E0，不重建图，不进行 MPI/MD。
能量就是原模型 forward 的原子输出之和；如果模型 core 只输出 interaction energy，
此 API 也只返回 interaction energy。绝对能量继续使用原 checkpoint/E0 wrapper，
或者按相同 fitted E0 表补齐；不能在训练时把两种能量标签混用。

helper 在 `no_grad` 内仍打开求力所需 autograd，推理结果会 detach。
`torch.inference_mode()` 不适合坐标求导，API 会明确拒绝。可传
`owned_mask[N]` 固定所有权掩码，仅对 owned 原子能量求导，返回所有 local/ghost
坐标的力；实际反向 ghost 通信仍由 LAMMPS 接口负责。

已有 checkpoint：

```python
from mace_ictc.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF

potential = LAMMPS_MLIAP_MFF.from_checkpoint(
    "trained_model.pth", element_types=["H","C","O"], device="cuda",
    avg_num_neighbors=training_average_neighbors,
    operator_plan="generated", generated_layout="packed",
)
model = potential.wrapper.model
```

species、训练平均邻居数、cutoff、dtype 与 E0 应来自真实 checkpoint/训练记录。
新 API 不把 FP64 checkpoint 强制转 FP32，也不把可训练 TP 系数偷偷冻结。

## 5. 训练

在原模型已保持其正确可训练 flags 的前提下配置：

```python
configure_model(model, training=True, policy="generated", layout="packed")
model.train()
parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(parameters, lr=1e-3)

optimizer.zero_grad(set_to_none=True)
out = energy_and_forces(model, pos, Z, batch, src, dst, shifts, cell, training=True)
loss = (out["energy"] - target_energy).square().mean() \
     + (out["forces"] - target_forces).square().mean()
loss.backward(inputs=parameters)
optimizer.step()
```

`training=True` 保留求力图和混合二阶导数；不会自动调用 `model.train()`，也不会
改变参数 trainability。从推理模型恢复训练时，应按 checkpoint 保存的原 flags
恢复，避免 `model.requires_grad_(True)` 把固定基础系数也重新设为可训练。
若确实需要 loss 对 leaf positions 的梯度，调用前令 `pos.requires_grad_(True)`
并按自己的 targets 请求梯度；普通参数训练使用上面的 parameter-only backward。

K100 (`gfx926`) 的 FP32 CG 电路使用原生 HIP 处理特征梯度、实时 SC 权重梯度和
混合二阶梯度。species bank 直接按节点种类索引，权重梯度按 256 个节点分块归约。
固定 TP 系数的生成式消息二阶反向直接累加到球谐输入，也支持普通
`loss.backward()` 请求的坐标梯度。这些路径沿用原有节点／通道和边／通道布局，
无需 HCU Triton 编译对应的训练伴随内核。三阶及更高阶求导仍使用可微表达式。
本次范围为 eager FP32、固定 TP 系数、实时可训练门控和 SC 权重；
`training_gate_schedule="materialized"` 使用上述完整二阶路径。

`ForceTrainer` 和 CLI 同样暴露布局选项：

```python
trainer = ForceTrainer(
    model, train_loader, config=training_config, device="cuda", dtype=torch.float32,
    operator_plan="generated", generated_layout="fused_pack",
    backward_targets="parameters",  # other established trainer arguments unchanged
)
```

从头训练的命令示例（使用已预处理数据；数据处理也在对应远端任务中运行）：

```bash
mff-train --data-dir /task/data --dtype float32 \
  --channels 64 --lmax 3 --max-ell 3 --correlation 3 --num-interaction 3 \
  --route baseline --product-backend ictd-bridge-u --angular-basis ictd \
  --freeze-conv-tp-weight --operator-plan generated --generated-layout packed \
  --backward-targets parameters
```

`--freeze-conv-tp-weight` 是新模型架构的显式选择；不能作为任意已有可训练 TP
checkpoint 的无影响优化。`--operator-plan auto` 默认不会修改 dtype 或冻结权重。
显式 make_fx 训练 capture 在 `auto` 下使用参考方案，不能把它与 generated eager
训练速度叠加宣称为同一个模式。

## 6. AOTI / LAMMPS

Python 准备入口：

```python
from mace_ictc.hp import prepare_for_export

model.eval().requires_grad_(False)
info = prepare_for_export(model, policy="generated", layout="fused_pack")
# Continue with the existing force make_fx -> torch.export -> AOTInductor pipeline.
# Optional preparation-only control: retain canonical U on its original device.
# info = prepare_for_export(model, policy="generated", reference_buffer_policy="device")
```

`layout=None` 保留已显式选择的布局，新模型默认为 packed。
`dynamic=True` 是默认值。已有 TorchScript e3nn 子模块会在 symbolic trace 中把 N
固定为示例大小，API 会提前报出模块名；需在模型重建/加载之前设置
`e3nn.set_optimization_defaults(jit_script_fx=False)`。仅明确准备固定 N 包时传
`dynamic=False`。正式导出 CLI 已在模型加载前处理这一设置。
函数只准备模型；正式包、验证与 LAMMPS metadata 用现有导出 CLI：

```bash
mff-export-aoti --checkpoint trained_model.pth --elements H,C,O \
  --device cuda --operator-plan generated --generated-layout packed \
  --avg-num-neighbors TRAINING_VALUE --embed-e0 --out /task/model.pt2
```

CLI 使用 checkpoint 的权重、实际结构和精度。`--generated-layout fused_pack`
将融合 copy 编入新包；已有 `.pt2` 不会因为 Python 配置改变而更新。
导出保留 owned-energy 接口、动态 N/E 和消息传递深度；通过 CLI 生成的 `.meta`
也保存 requested layout。`pair_style mff/torch` / `mff/torch/kk` 的完整构建与
调用见 [接口文档](../lammps_user_mfftorch/README.md)。
高 L 推理准备使用有界导出实现：L4/5、cor3 的 SC 使用共享二次前缀、稀疏最终
耦合及解析一阶伴随；cor2 的非 32 倍数通道、非标量目标也使用稀疏导出路径。
高 L TP 按输入 degree 和最多两条路径分区，C>=32 再按 32 通道分块，
保留真实图聚合和几何导数。
`reference_buffer_policy="auto"` 默认在合格单设备 cor3 的分解完成后将原始 U 留在
CPU；`"device"` 只改变准备阶段原缓冲驻留，不改变导出算子。CLI 对应选项为
`--reference-buffer-policy auto|device`。准备不在每步传输这些常量。
稠密 U 仍以原 checkpoint keys 注册；只有求力图实际引用的分解表、投影和冻结权重
进入导出包。准备后修改参数或 U 必须重新准备/导出，不能继续使用旧冻结银行。
不能把单边直接写出的 CUDA wrapper 直接塞入该导出图，图消息需要真实 reduction。

## 7. 支持范围、默认与回退

整网自动覆盖：CUDA FP32、内部 FP32、正整数 channels、非负 hidden/edge L
（edge L>=hidden L）、正整数 interaction 层数、笛卡尔 ICTC、
cor=2/3（兼容 1）、native full-CG/baseline bridge-U、无 interaction attention、无 cuEq 混合。
推理要求参数已冻结；训练要求架构的 TP 基础系数固定。径向/线性/SC 权重保持原样。
API 的覆盖与正确性门槛不保证每个图/层数/配置都更快。

### 统一策略入口

用户通过一个入口启用整组已验证的 HP 策略，无需逐层设置内部开关：

```python
# Existing CUDA FP32 model; preserve its parameters and checkpoint keys.
info = configure_model(model, training=True, policy="generated")
print(info["selected"], info["counts"])
```

`wide_channel_policy` 和 `high_l_policy` 默认均为 `auto`。`generated` 要求覆盖，
条件不满足时明确报错；`auto` 允许返回参考方案和原因。两者都不自动冻结权重。
推理时显式冻结参数并传 `training=False`，沿用第 4 节的完整求力调用。

| 模块 | 统一策略的适用规则 |
|---|---|
| TP、消息与径向网络 | 启用生成消息、实时门控与径向重算，保留真实 gather/scatter 和力导数 |
| SC | cor2 使用电路；L4/5 训练及 L5 尾部通道推理按 degree 拆分；cor3 在 L4/5 使用有界稀疏耦合 |
| 通道与元素矩阵 | 推理使用共享通道 GEMM；C>=96、hidden/edge L<=3、元素数<=4 时启用共享元素 GEMM |
| 节点聚合 | C>=96、hidden/edge L<=3 且 E>=4096、平均入度>=16 时选择 CSR 前向 |
| 高阶 TP | L4/5、C>=64 时流式几何计算；按训推阶段和设备选择通道分块，详见下文 |
| 精度与参数 | 始终严格 FP32、TF32 关闭；SC、径向和线性权重实时参与计算与求导 |

这些条件属于同一自动策略集。不同 SC 电路和聚合内核是互斥调度；
统一入口会选择适用路径。启用一组策略不能保证每个架构都同时降低延迟和显存。

对 C>=96 且 hidden/edge L<=3 的整网默认启用两项优化：
元素数<=4 时，按元素共享 GEMM，避免 `[N,C,C]` 权重复制及其梯度 scatter；
大且较密的图使用上文的 CSR 前向。元素银行、径向、通道与 SC 权重仍可训练。
较低通道仍保留原宽通道调度，interaction 层数和 cor=2/3 的选择仍独立。
`wide_channel_policy="off"` 关闭这两项宽通道优化，保留其他 HP 算子：

```python
configure_model(model, training=True, policy="generated", wide_channel_policy="auto")
# Restore the previous HP schedule for a controlled comparison.
configure_model(model, training=True, policy="generated", wide_channel_policy="off")
```

显式 `species_linear_schedule="grouped"` 可对整数物种索引的元素条件线性层
（包括 `self_connection`）按实际物种分组，每个非空组调用一次 GEMM，避免
`[N,C_out,C_in]` 临时权重。该算法不限定 C、L、物种数或 interaction 数；
按实际张量形状运行，参数梯度和高阶导数保留普通 PyTorch autograd。
每次调用重新建立节点分组，同一调用中的各角动量块复用分组；不缓存会随
ghost/owner 变化的成员关系。显存开销随节点特征大小增长。

```python
configure_model(model, policy="generated", species_linear_schedule="grouped")
# Controlled baseline using per-node weights.
configure_model(model, policy="generated", species_linear_schedule="gather")
```

该选项只支持 eager：分组计数需少量 GPU→CPU 元数据同步，不支持 capture；
小体系未必提速。`auto` 保留上表原有默认策略，`reference` 恢复参考路径。
它与对称收缩的 `species_schedule` 独立；软物种混合的 `forward(node_attrs)`
仍保留原公式。整网 HP 的覆盖范围仍由 `check_model` 判定。
大体系 MD 还须协调 PyTorch 缓存池与 LAMMPS/RCCL 原生分配器；
本次复现脚本在 Torch 池超过显存容量比例时，求力结束后返还闲置缓存，
开销包含在 MD 测速中。该策略位于接口层，不以模型超参数或固定 N 分派。

高阶整网默认由 `high_l_policy="auto"` 选择 L4/5 调度：cor3 的 SC 使用共享
二次 CG 前缀和稀疏最终耦合，避免生成过大的整段 cubic 电路；cor2 保留
原电路，训练及 L5 非 32 倍数通道推理的非标量输出按 degree 拆分，限制编译复杂度。
C>=64 的高阶 TP 使用逐输入/输出 magnetic 分量生成的几何计算。L4 训练用
32 通道分块、推理用原分块；L5 cor2 训练用原分块、推理用 32 分块；
L5 cor3 训练使用 32 分块；推理在 SM 8.9 GPU 使用原分块，其余保持 32 分块。
这个硬件选择已在 4090D 和 A100 40GB 上验证，不保证同架构的所有卡/图都有收益。
cor3 稀疏计算按每行系数数目控制计算块，避免大 K
导致每线程中间数组过大。所有原始 SC 学习权重经实时投影参与求值和求导，
没有缓存训练权重、截断系数或改变模型参数化。TP 和对外特征保持笛卡尔基；
SC 继续沿用既有 bridge-U 与独立 CG 前缀的实现。

```python
model = create_model(channels=128, lmax=5, correlation=3, num_interactions=3)
info = configure_model(model, training=True, policy="generated", high_l_policy="auto")
print(info["high_l_active"], info["counts"])
# Restore the preceding high-L schedules without replacing weights/checkpoints.
configure_model(model, training=True, policy="generated", high_l_policy="off")
```

独立 SC 使用相同策略；力损失混合反向可传
`configure_symmetric_contraction(sc, training=True)` 选择有界训练电路。
可传 `configure_symmetric_contraction(sc,
high_l_policy="off")` 恢复原整段电路。L<=3 的默认执行方案保持原样。
高阶调度的覆盖不等于所有 channels、图大小和 GPU 都有性能收益；较小图的
启动开销、首次常量规划和 JIT 编译仍需单独考虑。

默认将单设备 L4/5、cor3 的原始稠密 U 和其衍生转置缓冲在分解计划
验收后移到 CPU。GPU 热运行使用已有分解系数，特征、门控、全部学习权重和
反向计算继续保持 GPU FP32。此处只移动固定常量，不逐步搬运激活或训练权重。
原缓冲仍以相同 checkpoint keys、形状、dtype 和数值注册，`named_buffers()`
因此可能同时包含 CPU 和 CUDA 张量。参考调用和设备驻留选择会还原
原始设备缓冲；可微 `node_attrs` 的参考路径仍保留。高 L generated 导出准备
沿用 CPU 驻留；checkpoint/回退导出准备仍恢复设备缓冲。

```python
# Same default one-call HP strategy, including qualified high-L memory savings.
configure_model(model, training=True, policy="generated")
# Retain conventional device buffers for external tooling or a control run.
configure_model(model, training=True, policy="generated", reference_buffer_policy="device")
# Standalone SC has the same escape.
configure_symmetric_contraction(sc, training=True, reference_buffer_policy="device")
```

已初始化的分布式进程组及 DDP/DataParallel 封装自动保留设备缓冲。请在封装后
配置 HP；若要在已有 CPU 常量驻留的模型上新建并行封装，先选择 `device`。
频繁交替离散 species 的 HP 调用与可微属性参考调用会产生常量传输；这种用法
可固定 `device`。高 L generated 导出默认保留 CPU U，也可通过
`prepare_for_export(..., reference_buffer_policy="device")` 选择设备驻留。
prepared Python、独立 `.pt2` 运行器和实际 LAMMPS 进程的峰值分别测量；
不能直接把 eager 减少的常量字节数当作另外两种进程的峰值差值。

新的 CUDA CSR 调度用于 eager 训推。AOTI 继续使用 prepared Triton 消息内核；
高阶 prepared SC/TP 按第 6 节选择独立的有界实现；AOTI/LAMMPS 的速度仍以
该导出包的完整求力实测为准，不能直接套用 eager 的加速比。
显式 `generated` 会保留真实 U 张量的重构精度检查；不满足预算时直接报错，
不会把原始 SC 回退伪装成高性能支持。首次规划/JIT 成本单独于热吞吐。

```python
# Keep the established default; auto reports unsupported-model reasons.
info = configure_model(model, training=True, policy="auto", layout="packed")
# Opt in to fused copies on a qualified model.
info = configure_model(model, training=True, policy="generated", layout="fused_pack")
# Revert only the HP layout.
configure_model(model, training=True, policy="generated", layout="packed")
# Revert the eager operator recipe.
configure_model(model, training=True, policy="reference")
```

训练 CLI 回退 `--operator-plan reference`；checkpoint/export 保留原路径使用
`operator_plan="checkpoint"` / `--operator-plan checkpoint`。
`prepare_for_export(..., policy="checkpoint")` 保留当前 checkpoint 路径，不负责
撤销已经准备的 deployment callable；需要干净原路径时先重新加载原 checkpoint。

`fused_pack` 保持可选，速度与显存效果随图形、模型和执行模式变化。默认 `packed`。

## 8. 已单独资格验证的高级组合

cor=2/reduced-CG 不在当前 `auto` 覆盖内；保留原来的公开低层 API
`mace_ictc.models.operator_plans.set_mace_backbone_plan` 用于显式组合。对已经由
调用者单独验证的 FP32/C64 模型，可以显式选择：

```python
from mace_ictc.models.operator_plans import set_mace_backbone_plan

set_mace_backbone_plan(
    model, bond_first=True, angular_backend="triton_full", path_assembly="concat",
    edge_fusion="triton_remat", activation_fusion=True, compact_contraction=True,
    factorized_circuit=True, streamed_gates=True, generated_message=True,
    generated_second=True, fused_harmonics=True, fold_frozen_path_weights=True,
    shared_channel_gemm=False, edge_chunk_size=65536, gate_cache_bytes=0,
    generated_layout="packed",
)
```

低层 selector 返回注册模块计数，不保证任意架构/输入都命中快核。必须保留对应
数值资格验证，不能用它绕过高层 `generated` 的覆盖/可训练性检查来宣称自动支持。
验证应覆盖 FP32 E/F、原子 reference energy、所有 force-loss 参数梯度和实际
保存 U 的 span 重构，不用模块计数代替正确性验证。

## 低阶求导与图消息调度

独立 `EdgeTensorProduct` 在 L<=3 时默认按实际请求的一阶/二阶梯度类型选择
内核。逐边二阶路径保留原生 packed 布局并直接写回，不再绕行图聚合的布局
转换和原子加法。几何二阶直接归约到球谐输入，更高阶导数仍有原生表达式回退。已有可训练 TP
基础系数依然不会被 API 自动冻结；学习的径向门控保持实时计算和求导。

整网新增显式 `message_schedule`，默认为 `"native"`，保留既有高性能方案：

```python
# 已有模型和 checkpoint 可直接配置，权重/模型架构均不改变。
configure_model(model, training=True, policy="generated", layout="fused_pack",
                message_schedule="roles")
configure_model(model, training=False, policy="generated", layout="fused_pack",
                message_schedule="roles")

# 回到原调度，其他 SC/高阶/宽通道策略仍由既有参数控制。
configure_model(model, training=True, policy="generated", message_schedule="native")
```

`"roles"` 使用原笛卡尔坐标和按梯度类型编译的 CUDA 伴随。
`"compact"` 坐标切换调度不受支持。调用方应使用 `"roles"` 或 `"native"`，
保持当前原生笛卡尔基；此限制不涉及独立 TP 的 `execution="legacy"` 布局兼容路线。

按梯度类型选择的显式消息方案只覆盖 hidden/edge L<=3；高阶仍可用既有 `native`
方案、`high_l_policy` 和 AOTI。特别是 `canonical` 访存和实验性的多 warp/分组
方案不属于默认推荐配置。channels、interaction 层数和 cor 的结构参数
仍由用户选择；具体设备和形状的性能验收范围应另行确认。

原生 packed 图接口供已经持有二维缓冲的调用方使用：

```python
from ictc_o3 import PackedGraphMessage

op = PackedGraphMessage.create(channels=64, input_lmax=3, edge_lmax=3,
                               cartesian_schedule="native", reduction="csr")
# x: [N, C*(L_in+1)**2]，每阶 magnetic 在前、channel 在后
# sh: [E, (L_edge+1)**2]；g: [E, op.gate_width]
# src/dst: [E]，int64 CUDA，有效节点索引，可重复或未排序
# 输出: [N, op.output_width]，路径不归约，布局同 EdgeTensorProduct.ir_mul
out = op(x, sh, g, src, dst)
grad_x, grad_g = torch.autograd.grad(out, (x, g), output_cotangent)
```

包装已有固定系数 TP 时使用 `PackedGraphMessage(tp, input_lmax=..., edge_lmax=...)`；
`create` 构造新的单位固定基础系数 TP。`cartesian_schedule="native"` 使用原坐标，
也是当前唯一支持的坐标方案。接口支持普通及混合 autograd；此接口为 eager
CUDA FP32，不直接用于 AOTI。现有字典接口 `GraphMessage` 保持可用，便于兼容
使用 `[N,C,2l+1]` 特征块的调用方。不要将纯边吞吐当作完整能量/力吞吐。

## 大图的设备端 CSR

整网可以显式选择 CSR 前向和设备端分组；一阶/混合反向保持既有方案。
本轮 C64/L3、50 邻居的图消息同卡复测有收益，显存峰值基本相同。
整网推理的单独进程优势未在同卡配对中稳定复现，因此这些开关用于显式
调优，不替换默认方案；结论不推广到所有 channels、L 或 GPU。

```python
configure_model(model, training=False, policy="generated", layout="fused_pack",
                message_schedule="roles", graph_reduction="csr",
                topology_backend="device", csr_accumulator_limit=24)
configure_model(model, training=True, policy="generated", layout="fused_pack",
                message_schedule="roles", graph_reduction="csr",
                topology_backend="device", csr_accumulator_limit=24)

op = PackedGraphMessage.create(channels=64, input_lmax=3, edge_lmax=3,
     cartesian_schedule="native", reduction="csr", topology_backend="device",
     csr_accumulator_limit=24)
out = op(x, sh, g, src, dst)
```

`graph_reduction="auto"` 默认沿用已有宽通道/高 L 策略；`"atomic"`/`"csr"`
显式选择前向归约。`topology_backend="native"` 默认用稳定排序分组；
`"device"` 每次调用在 GPU 上重新计数、构造指针并分组，支持未排序和重复
节点索引，不缓存图、特征或学习权重。已排序输入由 GPU 检测并保留边顺序；
未排序输入的节点内部边顺序不保证稳定，浮点
归约顺序可改变，验收采用数值容差。`csr_accumulator_limit=None` 保留原策略
的分块设置；正整数控制 CSR 内核的累加器预算。参考模式不使用设备端快路径。

这些开关不改变模型结构、dtype 或参数的可训练性。已验证的配置仍为 eager
CUDA FP32；新 CSR 调度没有增加 AOTI 性能验收范围。源节点融合反向、跨阶
融合和持久化 warp 调度的实验未表现出稳定收益，因此没有作为默认方案。

## 原生基的实现边界

前向、反向和训练状态始终保持已有 ICTC 角向基与 CG/SC 约定，不在 TP 内
切换到仅选择z指标数<=1的 STF 分量。保留逐边二阶直接写回、按梯度类型
编译、原生 packed 缓冲、分块径向门控及既有高L/宽通道方案。
`compact_contraction` 指既有 SC 实现的压缩策略，不是已退役的坐标切换。

加载带旧实验标志的模型后调用 `configure_model(...,message_schedule="roles")`
会清除该标志和临时坐标缓存，保留权重、架构、dtype和参数可训练性。
明确请求旧 `compact` 的调用会报错，不会静默替换并沿用旧性能结论。

原生基的 CSR 分块可显式启用 `csr_slice_search=True`。稳定分组中的原始边ID
按节点单调排列时，用上下界搜索缩小当前chunk的邻居遍历范围；节点没有本块
边时直接结束计算。device分组只有GPU确认输入有序才使用该搜索，否则沿用
完整扫描，避免把未排序的原始边ID误当有序。整个过程不改变CG、坐标、路径
或求导公式；默认关闭，需对使用形状验收性能。

## 可选按节点索引读取 SC 元素权重

`configure_model(..., species_schedule="indexed")` 和
`configure_symmetric_contraction(..., species_schedule="indexed")` 为 L<=3 的
原生 full-CG circuit 选择按节点读取元素系数的路径。它沿用已有 CG 多项式与
实时权重银行，省去按元素 `nonzero`、节点分组和整块回填；不缓存输入特征、
元素分组或学习参数。首阶、参数梯度和混合二阶仍可计算。

默认 `species_schedule="grouped"` 保留原执行路径。高阶与已有分区/稀疏 SC
继续用原计划；返回信息中的 `indexed_species_circuits` 计数（独立 SC 为
`indexed_species_active`）说明实际覆盖。`policy="reference"` 清除索引执行选择。
离散 `node_type_idx` 必须为同设备 int64 `[N]`，且每个值均在元素银行范围内。
只有 `node_attrs` 的调用继续用原约定，不将可微元素输入替换成离散索引。

这项选择不改变径向分页或门控缓存预算。调用方可分别选择容量与吞吐配置，
并对实际网络检查能量、力、参数梯度、热吞吐和峰值显存。固定形状 CUDA Graph
仍是调用方单独准备、验证的执行协议；启用本选项本身不会捕获或缓存整网调用。

## 原生几何调度与边块大小

生成 CG 表现在沿用原生稀疏 TP 的 `1e-12` 数值零规则。L3 表中删去的
构造残差约为 `1e-16`；不生成新的基，也不改变参考投影矩阵或学习权重。

`geometry_schedule="auto"` 保留已有计划；`"streamed"` 在一阶伴随中
按输入分量计算原生几何项，降低同时存活的临时量；`"native"` 显式恢复
此前的几何计算顺序。不同 channels/L/设备的速度仍须按实际配置测量。

已经持有全部 gates 的独立图算子可以使用较大的边块：

```python
op = PackedGraphMessage.create(
    channels=64, input_lmax=3, edge_lmax=3,
    reduction="csr", topology_backend="device", csr_accumulator_limit=24,
    csr_slice_search=True, geometry_schedule="streamed",
    edge_chunk_size=1_048_576,
)
y = op(x, sh, gates, src, dst)
```

整网训练可显式选择相同的几何调度和块大小：

```python
configure_model(
    model, training=True, policy="generated", layout="packed",
    message_schedule="roles", graph_reduction="csr", topology_backend="device",
    csr_accumulator_limit=24, csr_slice_search=True,
    geometry_schedule="streamed", message_chunk_edges=1_048_576,
)
```

容量求力推理保留小块径向门控：

```python
model.eval().requires_grad_(False)  # 由调用方明确选择冻结推理参数
configure_model(
    model, training=False, policy="generated", layout="packed",
    message_schedule="roles", graph_reduction="csr", topology_backend="device",
    csr_accumulator_limit=24, csr_slice_search=True,
    geometry_schedule="streamed", message_chunk_edges=65_536,
    gate_cache_bytes=0,
)
```

`message_chunk_edges=None` 保留已有块大小；正整数显式覆盖每一层。训练或
独立图算子已有完整 gates 时，大块减少调用开销；流式径向推理的大块会扩大
临时 gate 缓冲，须同时看峰值显存。默认不自动扩大容量推理的边块。
本轮未把这些 eager 调度的速度结论扩展到 AOTI 或已有部署包。

### K100 上 C64/L2 大体系的前向与反向分块

对已冻结参数的两层 C64、hidden/edge L2、full-CG correlation3 模型，
K100/DTK2604 的大体系求力可显式扩大前向块，同时固定较小的反向块：

```python
model.eval().requires_grad_(False)
configure_model(
    model, training=False, policy="generated", layout="fused_pack",
    message_schedule="roles", species_schedule="indexed",
    graph_reduction="csr", topology_backend="native",
    csr_accumulator_limit=24, csr_slice_search=True,
    geometry_schedule="streamed", message_chunk_edges=1_048_576,
    gate_cache_bytes=0, inference_message_layout="degree",
    inference_gate_adjoint_rows=16_384,
    inference_message_adjoint_edges=32_768,
    inference_gemm_policy="dtk2604", inference_basis_fusion=True,
)
```

CSR 每个前向块都会为全部目标节点启动任务，再排除块外节点。较大的
前向块减少重复查找和启动次数；独立的反向块限制求力临时量。该配方
保持 FP32、关闭 TF32，不改变权重或 CG/SC 公式，不启用 gate 缓存。

实际部署应按模型、图规模和设备预算选择前向及反向块大小，并单独验证显存。

## 可选训练 gate 显存调度与混合反向分块

`configure_model(..., training=True, training_gate_schedule=...)` 可在原来的
Cartesian/full-CG 消息公式上改变 gate 中间量的保存方式：

| 值 | 保存和重算行为 |
|:---|:---|
| `"materialized"` | 默认，保留既有训练路径。 |
| `"cached"` | 保留本次 forward 的 gates；第一反向只保留窄径向特征梯度，混合二阶反向按块重算 gate 梯度。 |
| `"recompute"` | 在上述基础上，反向也按块重算 forward gates。 |

这两个可选方案以重算和额外矩阵乘法换取较低峰值显存，不保证提速。
`message_chunk_edges` 控制其边块大小。`gate_cache_bytes` 仍只控制冻结参数
的求力推理缓存，训练时必须为零；`cached` 训练方案保存本次 tape 的全部
forward gates，不受该推理缓存预算控制，也不跨调用缓存活跃特征或权重。

```python
info = configure_model(
    model, training=True, policy="generated", layout="fused_pack",
    message_schedule="roles", species_schedule="indexed",
    graph_reduction="csr", topology_backend="device",
    csr_accumulator_limit=24, csr_slice_search=True,
    geometry_schedule="streamed", message_chunk_edges=65_536,
    training_gate_schedule="cached",  # 更低显存，同时增加训练耗时
)
```

适用范围为 eager CUDA FP32、hidden/edge L<=3、原来的固定 TP 系数，以及
活跃径向线性层参数。参数对象、可训练性、checkpoint keys 和原有 CG/SC
多项式保持不变。快速混合反向针对 `autograd.grad(loss, trainable_parameters)`
这样的参数梯度请求；包含坐标 Hessian 的反向和更高阶导数保留可微参考表达式，
不承诺同样的显存收益。此方案未用于 AOTI 或推理加速。

再次调用 `configure_model` 且不指定 `training_gate_schedule`，或切换到
`policy="reference"`，会恢复 `"materialized"`。接口返回所选 schedule 和
`counts["paged_training_gates"]`，不会自动启用这两个增加重算的方案。

`mixed_channel_tile=64` 或 `32` 只调整生成式混合二阶 TP 内核的通道分块，
不改变第一反向或 forward 的分块，也不增加 gate 重算。默认 `None` 保留
既有调度；`0` 明确使用完整通道的单个分块。该选项同样只覆盖训练的 L<=3，
并可与 `training_gate_schedule` 独立组合。宽通道可能受益，C64 不保证收益。
不要把单个形状的速度推广到全部通道数。切换配置时省略该参数会清除旧值。

### K100 含力训练的算子布局

`training_adjoint_channel_tile=32` 或 `64` 单独调整第一 TP 反向的通道布局，
同时覆盖 `create_graph=True` 求力和 loss backward 中的一阶 VJP。
前向和混合二阶保持各自的布局选择。默认 `None` 保留既有调度，`0` 使用
完整通道展开。该选项要求 materialized、eager、hidden/edge L<=3。
K100 C128/256、L3 实测采用第一反向 64 通道、混合二阶 32 通道更快。

`mixed_geometry_schedule="split"` 将 TP 的球谐二阶伴随放到独立的
edge/channel 内核，沿用 32 lane 通道归约，直接写回球谐梯度。
该布局降低特征、门控和输出伴随内核的寄存器压力。默认 `"fused"`
保留联合内核；C128/256、L3 周期图实测适合 `"split"`，C64/L2 适合
`mixed_channel_tile=32` 配合 `"fused"`。应按实际 channel/lmax 和图大小选择。

`training_gate_gemm_policy="k100_tiled"` 对已验证的径向末层投影使用
16384 行的特征和权重梯度 GEMM，保留原 forward 线性映射。首阶和混合二阶
梯度均覆盖实时权重和输入，包含标准 `loss.backward()` 的坐标梯度。
适用条件为 gfx926、PyTorch 2.7.1/HIP 6.3.26093、FP32、704×64 权重，
且至少 65536 条边；其他形状继续使用原线性层。默认值为 `"torch"`。
FP32 归约次序会变化，参数对象、精度和 checkpoint keys 保持原约定。

两项新调度均要求 eager、materialized gates。C64/L2 的配置示例：

```python
configure_model(
    model, training=True, policy="generated", layout="fused_pack",
    message_schedule="roles", species_schedule="indexed",
    graph_reduction="csr", topology_backend="native",
    csr_accumulator_limit=24, csr_slice_search=True,
    geometry_schedule="streamed", message_chunk_edges=32768,
    training_gate_schedule="materialized", mixed_channel_tile=32,
    mixed_geometry_schedule="fused", training_gate_gemm_policy="k100_tiled",
)
```

C128/256、L3 使用 `mixed_geometry_schedule="split"` 和
`training_adjoint_channel_tile=64`；C256/L3 同时将 `csr_accumulator_limit` 设为 48。
K100 原生 CG/TP 的能量、力及混合伴随
继续使用同一组 CG 系数和实时训练参数。

### 消息线性层的节点／角分量行布局

generated eager training 默认使用 `training_message_linear_layout="node_component"`，将消息线性层的
`[N,Cin,2l+1]` 输入整理为连续的 `[N*(2l+1),Cin]`，共享同一份
`[Cout,Cin]` 实时权重执行 GEMM，再恢复原有输出分量顺序。这与已有通道
adapter 的共享 GEMM 布局一致，避免 `einsum` 在大图上整理
`[channel,node,angular]` 大张量产生的跨步复制。

```python
configure_model(
    model, training=True, policy="generated", layout="fused_pack",
)
# 显式恢复旧布局：
configure_model(model, training=True, training_message_linear_layout="einsum")
```

该开关可与上述 TP/gate 调度同时使用；它不替代这些调度。权重、输入和上游
特征的混合伴随通过普通可微 `linear` 计算，包含力损失及坐标二阶反传。
参数对象、trainability、state_dict keys 和对外特征顺序保持原样，FP32
归约顺序可能改变。该默认值同时适用于 `ForceTrainer` 和训练 CLI；CLI 可用
`--training-message-linear-layout einsum` 恢复旧布局。CUDA FP32 且未启用 autocast
时使用新布局，其他 dtype/设备沿用原计算。重新配置 eager generated training
时省略该选项，会重新启用默认布局；推理、`policy="reference"` 或 `auto` 的
reference 回退会清除该布局标志。`capture=True` 的训练在 `auto` 下沿用
reference 路径，显式 `policy="generated"` 仍因不支持训练 capture 而报错。

## 径向重算与逐层 gate 调度

`radial_checkpoint=True` 通过 non-reentrant checkpoint 按需重算径向 MLP
激活。在完整 gate 路径中覆盖原来的整个 `fc`；在流式 gate 路径中覆盖隐藏层，
末层线性投影仍由原 gate 调度执行。此选项同时在消息算子完成后及时释放
布局转换的临时输入，再分配返回张量。默认关闭，不修改参数、精度或模型公式。

`training_gate_schedule` 还可接受 tuple/list，长度必须等于 interaction 数量。
原来的单个字符串继续应用到全部层。两层 C128/L3 的速度/显存折中配置为：

```python
configure_model(
    model, training=True, policy="generated", layout="fused_pack",
    message_schedule="roles", species_schedule="indexed",
    graph_reduction="csr", topology_backend="device",
    csr_accumulator_limit=24, csr_slice_search=True,
    geometry_schedule="streamed", message_chunk_edges=1_048_576,
    mixed_channel_tile=64,
    training_gate_schedule=("cached", "materialized"),
    radial_checkpoint=True,
)
```

第一层保留 forward gates、按需重算 gate 梯度，第二层保留原速度路径。
两层径向 MLP 的中间激活按需重算。该方案仍有额外计算，不能把显存下降
表述为零成本；速度与显存应以同一工作负载的测量为准。

新选项限定 eager CUDA FP32、hidden/edge L<=3 的训练。请求 capture 会报错；
编译/代理张量、functorch、CPU 或非 FP32 的径向调用保留普通计算路径。
省略 `radial_checkpoint` 的再次配置、推理配置和 `policy="reference"`
均清除该选项。参数对象、可训练性、state_dict 和 checkpoint keys 保持不变。
推荐配置按两层模型验证，不能直接把两个 schedule 值用于不同层数的模型。

## 能量与力推理的消息布局

`inference_message_layout="degree"` 让 TP 直接按 L 分段生成适合后续线性层
的缓冲区，并让计算力的第一反向直接读取这个布局，省去大块消息的重排。
这是原生 ICTC 消息的存储布局调整，CG/SC、多项式、参数和精度都不变。
另外，特征的逐 L 独立切片改成一次 `split`，反向只合并一次不重叠的梯度。

A100 上两层、correlation=3、N8192、每节点 50 条边的速度配置：

```python
# 调用方在专用推理模型上明确固定参数；configure_model 本身不改变可训练性。
model.eval().requires_grad_(False)
configure_model(
    model, training=False, policy="generated", layout="fused_pack",
    message_schedule="roles", species_schedule="indexed",
    graph_reduction="csr", topology_backend="device",
    csr_accumulator_limit=24, csr_slice_search=True,
    geometry_schedule="streamed", message_chunk_edges=81_920,
    gate_cache_bytes=5 * 1024**3,
    inference_message_layout="degree",
)
```

`gate_cache_bytes` 是每个 interaction 的预算，不是网络的总显存上限。
分页大小、缓存预算和图规模共同决定峰值。预算为 0 时继续按页重算 gates，
可用于容量优先的部署；速度配置的结果不代表所有节点数下都达到相同收益。

新布局默认关闭，限定 eager CUDA FP32、hidden/edge L<=3 和 `fused_pack`。
显式请求训练、capture 或高 L 会报错。流式 frozen-gate 路径的既有条件仍然
适用，不支持的张量/参数上下文保留原计算路径。高阶导数使用可微表达式；
缓存不会被反向原地覆盖，因此保留图后再次求导仍然可用。

再次配置而省略此选项，或切换到 `policy="reference"` / 训练配置，会清除
新的消息布局。参数对象、可训练性和 state_dict keys 保持不变。

### 求力伴随通道分块

上述配置现在默认使用 `inference_adjoint_channel_tile="auto"`；也可显式固定为
`64`。此选项复用已有 CUDA
分块内核，只对需要输入特征梯度的 frozen-gate 第一伴随按通道切分工作，
第一层的纯几何/门控伴随和前向调度不变。不增加中间张量；球谐梯度的
分块归并使用原有输出缓冲。浮点归并顺序会变化，不保证逐位相同。

取值为 `"auto"`（默认）、`0`（原调度）、`32` 或 `64`。
训练时 auto 自动关闭，显式 32/64 不允许 `training=True`。
它不会冻结参数，也不改变高阶导数的可微回退。
再次配置省略此选项会恢复 auto；设为 `0` 或切换到 reference 清除分块设置。
容量优先保持 `gate_cache_bytes=0`，速度优先保持原缓存预算；两者均可
使用该选项。其他设备和配置应单独验证。

`"auto"` 使用有边界的结构规则，避免把固定 64 通道方案推广到所有模型。
它目前仅针对 A100 和 RTX 4090 D、degree 布局、streamed 几何、roles 伴随，要求边数至少
4096、C 在 128–256 内，且最大角输入/输出宽度均为 7（L3 路径）。
内核源级活跃值估计 `ceil(C/32) * max(2*输入宽度 + 输出宽度 + 3)`
至少为 144 时选择 64，其余保留原调度。该估计不是硬件寄存器计数。
两种设备共用同一结构规则；C192 在 4090 D 上可能小幅回退，验收上限为 5%。
显存和数值标准不放宽；A100 的选择范围不变。
普通 RTX 4090 和其他 Ada 卡尚未纳入自动启用范围。
它只读取形状/设备元数据，不进行在线调优、GPU 分配或同步；不同原子数
复用同一组内核，不按 N 单独编译。未知设备、低负载和未覆盖角宽度自动回退。
容量/速度模式共用同一个调度器，二者只由门控缓存预算区分。
这仍是有验证范围的经验策略，不保证所有图分布或通道数都达到最优速度。

## 非线性 MACE foundation checkpoint

`mace_ictc.interfaces.foundation_hp.FoundationHP` 支持 MH-1 使用的
`RealAgnosticResidualNonLinearInteractionBlock` 架构。节点与边通道可不同，
维数、路径、截断、元素及 head 从 checkpoint 读取。转换保留原始 Agnesi、
ZBL、元素径向网络、LayerNorm、密度归一化、非线性门控和全部 readout heads。
TP 与对称收缩使用 ICTC HP；径向网络分块重算，保留真实几何导数。
HP 对称收缩直接接收前级非线性层已有的 CG 基特征，消去等价的基变换往返与
大规模临时布局缓冲；`product_basis="bridge"` 可选择保留往返的对照路径。
两条路径均使用实际 checkpoint U 张量的重构校验和原始学习系数。

在指定远端计算资源上转换：

```bash
python -m mace_ictc.cli.convert_foundation \
  --mace-model mace-mh-1.model --head omat_pbe --out mace-mh-1-ictc-hp.pt
```

转换文件保存 FP64 模块。使用时显式设置精度与冻结权重，再启用 HP：

```python
import torch

torch.set_default_dtype(torch.float32)
model = torch.load("mace-mh-1-ictc-hp.pt", map_location="cpu", weights_only=False)
model = model.float().cuda().eval().requires_grad_(False)
plan = model.configure_inference(
    chunk_edges=262144, backward_edges=32768, radial_chunk=32768,
)
model.select_head("omat_pbe")
value = model.forward_vectors(vectors, species, src, dst)
edge_derivative = torch.autograd.grad(value["interaction_energy"].sum(), vectors)[0]
```

`vectors[E,3]` 为从 src 指向 dst 的实时位移，需开启梯度；`species[N]` 为
checkpoint 元素列表中的索引。结果包含每原子 `interaction_energy` 和含 E0
的 FP64 `node_energy`。采用 node-owned 分域时传入 `owned` 与
`exchange(layer, projected_state)`；回调应交换经过当前层 `linear_up` 的特征，
并实现其反向通信，返回与输入相同布局的 owned+ghost 状态。

此接口用于 eager、冻结权重的求力推理；额外电荷/自旋嵌入不在当前覆盖范围。
转换后需要对实际 checkpoint 做能量与力校验，容量随模型、材料密度和邻居数变化。
