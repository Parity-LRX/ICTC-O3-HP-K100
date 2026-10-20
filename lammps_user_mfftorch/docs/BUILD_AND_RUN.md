# LAMMPS + Kokkos + LibTorch 完整编译与运行指南

本文档描述如何从零编译带 **Kokkos GPU** 与 **USER-MFFTORCH**（LibTorch 势函数）的 LAMMPS，并完成端到端 MD 运行。

## 目录

1. [前置条件](#1-前置条件)
2. [获取 LAMMPS 源码](#2-获取-lammps-源码)
3. [安装 USER-MFFTORCH](#3-安装-user-mfftorch-到-lammps-源码树)
4. [修改 CMakeLists.txt](#4-修改-lammps-cmakeliststxt一次性)
5. [CMake 配置](#5-cmake-配置)
6. [编译](#6-编译)
7. [导出 LAMMPS 模型 core](#7-导出-lammps-模型-core)
8. [运行 LAMMPS](#8-运行-lammps)
9. [一键测试脚本](#9-一键测试脚本)
10. [故障排查](#10-故障排查)

---

## 1. 前置条件

### 1.1 系统与工具

| 依赖 | 要求 |
|------|------|
| 操作系统 | Linux（推荐，用于 CUDA） |
| CMake | ≥ 3.20 |
| C++ 编译器 | GCC 7+ 或 Clang，支持 C++17 |
| CUDA | 11+（与 PyTorch/LibTorch 版本匹配） |
| Python | 3.9+（用于 MACE-ICTC 导出 `.pt2` / `core.pt`） |

### 1.2 Python 环境

- 安装 PyTorch（含 CUDA 支持，与 GPU 驱动版本匹配）：
  ```bash
  pip install torch  # 或根据 CUDA 版本选择 torch+cu118 等
  ```
- 安装本仓库：
  ```bash
  pip install -e .  # 在 rebuild 仓库根目录
  ```

### 1.3 LibTorch 路径

USER-MFFTORCH 通过 `find_package(Torch)` 查找 LibTorch。推荐使用 **Python 内置的 LibTorch**：

```bash
python -c "import torch; print(torch.utils.cmake_prefix_path)"
```

输出示例：`/path/to/python/site-packages/torch/share/cmake/Torch`。将该路径传给 CMake 的 `CMAKE_PREFIX_PATH`。

---

## 2. 获取 LAMMPS 源码

```bash
# 方式 A：下载官方源码
wget https://download.lammps.org/tars/lammps-22Jul2025.tar.gz
tar xzf lammps-22Jul2025.tar.gz
cd lammps-22Jul2025

# 方式 B：Git 克隆
git clone https://github.com/lammps/lammps.git
cd lammps
git checkout 22Jul2025  # 或 develop 等
```

---

## 3. 安装 USER-MFFTORCH 到 LAMMPS 源码树

在 **ICTC-O3-HP 仓库根目录**，将接口源码复制到目标 LAMMPS 源码树：

```bash
lammps_source=/path/to/lammps
mkdir -p "$lammps_source/src/USER-MFFTORCH"
cp -R lammps_user_mfftorch/src/USER-MFFTORCH/. "$lammps_source/src/USER-MFFTORCH/"
```

然后根据目标 LAMMPS 的目录约定复制对应 CMake 文件：

```bash
cp lammps_user_mfftorch/cmake/Modules/Packages/USER-MFFTORCH.cmake \
  "$lammps_source/cmake/Modules/Packages/"
# 对采用 cmake/Packages 的版本，改用该目录下的同名文件。
```

本仓库不分发内部测试或安装任务脚本。完成上述复制后，按下一节配置 LAMMPS。

---

## 4. 修改 LAMMPS CMakeLists.txt（一次性）

LAMMPS 默认不识别 `USER-MFFTORCH`，需手动修改 `LAMMPS/cmake/CMakeLists.txt`。

### 4.1 添加 USER-MFFTORCH 到 STANDARD_PACKAGES

找到 `set(STANDARD_PACKAGES ...)`，在列表末尾添加 `USER-MFFTORCH`：

```cmake
set(STANDARD_PACKAGES
   ...
   VTK
   YAFF
   USER-MFFTORCH)  # 添加此行
```

### 4.2 添加 USER-MFFTORCH 到 PKG_WITH_INCL

找到 `foreach(PKG_WITH_INCL ...)` 的 include 列表（用于链接外部库的包，如 GRAPHICS、PYTHON、ML-IAP 等），在其中加入 `USER-MFFTORCH`。

**方式 A：加入现有 foreach 列表**

```cmake
foreach(PKG_WITH_INCL GRAPHICS KSPACE PYTHON ML-IAP ... USER-MFFTORCH)
  if(PKG_${PKG_WITH_INCL})
    include(Packages/${PKG_WITH_INCL})
  endif()
endforeach()
```

**方式 B：单独添加（若找不到合适列表）**

在 `foreach(PKG_WITH_INCL ...)` 附近添加：

```cmake
if(PKG_USER-MFFTORCH)
  include(Packages/USER-MFFTORCH)
  # 若报错找不到文件，改为：include(Modules/Packages/USER-MFFTORCH)
endif()
```

> `Packages/` 对应 `cmake/Packages/USER-MFFTORCH.cmake`；`Modules/Packages/` 对应 `cmake/Modules/Packages/USER-MFFTORCH.cmake`。安装脚本会拷贝到两个位置，按你 LAMMPS 版本的实际 include 路径选择其一。

---

## 5. CMake 配置

### 5.1 获取 LibTorch 路径

```bash
LIBTORCH_PREFIX=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
```

### 5.2 选择 GPU 架构

根据你的 GPU 选择 `Kokkos_ARCH_XXX`：

| GPU | Kokkos_ARCH |
|-----|-------------|
| NVIDIA A100 | AMPERE80 |
| NVIDIA A30 / RTX 30 系列 | AMPERE86 |
| NVIDIA V100 | VOLTA70 |
| NVIDIA T4 | TURING75 |
| AMD GPU | 参见 Kokkos 文档 |

### 5.3 编译选项：Virial/应力计算

| 选项 | 默认 | 说明 |
|------|------|------|
| `MFF_ENABLE_VIRIAL` | OFF | 设为 ON 时，在 GPU 上计算 virial，可输出正确的压力/应力张量 |

- **默认（OFF）**：不计算 virial，`thermo_style` 中的 `press` 等只有动能贡献，数值不完整；吞吐量最高。
- **ON**：在 GPU 上用 Kokkos 计算 fdotr virial，只拷贝 6 个 double 回 CPU，**几乎无性能损失**；`press`、`pxx`、`pyy`、`pzz`、`pxy`、`pxz`、`pyz` 输出正确。

### 5.4 配置命令

**不启用 virial（默认，最高吞吐）：**

```bash
cd /path/to/lammps

cmake -S cmake -B build-mfftorch \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_AMPERE86=ON \
  -D PKG_USER-MFFTORCH=ON \
  -D CMAKE_PREFIX_PATH="$LIBTORCH_PREFIX" \
  -D CMAKE_BUILD_TYPE=Release
```

**启用 virial（压力/应力正确）：**

```bash
cmake -S cmake -B build-mfftorch \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_AMPERE86=ON \
  -D PKG_USER-MFFTORCH=ON \
  -D MFF_ENABLE_VIRIAL=ON \
  -D CMAKE_PREFIX_PATH="$LIBTORCH_PREFIX" \
  -D CMAKE_BUILD_TYPE=Release
```

**若 LibTorch 未被自动找到**，可显式指定：

```bash
cmake -S cmake -B build-mfftorch \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_AMPERE86=ON \
  -D PKG_USER-MFFTORCH=ON \
  -D MFF_ENABLE_VIRIAL=ON \
  -D CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -D CMAKE_BUILD_TYPE=Release
```

---

## 6. 编译

```bash
cmake --build build-mfftorch -j
```

编译成功后，可执行文件位于：

```
build-mfftorch/lmp
```

---

## 7. 导出 LAMMPS 模型 core

当前推荐给 `pair_style mff/torch` / `pair_style mff/torch/kk` 使用的是
AOTInductor `.pt2` core。TorchScript `core.pt` 路径仍可用于 legacy LibTorch
部署，也适合调试导出签名和数值差异。

### 7.1 从 MACE-ICTC checkpoint 导出 `.pt2`

```bash
mff-export-aoti \
  --checkpoint /path/to/model.pth \
  --elements H,O \
  --out model.pt2 \
  --dynamic \
  --dtype float32 \
  --device cuda \
  --embed-e0
```

如果目标体系原子数固定，static-N 导出通常更稳，也方便 LAMMPS smoke test：

```bash
mff-export-aoti \
  --checkpoint /path/to/model.pth \
  --elements H,O \
  --atoms 256 \
  --degree 32 \
  --static-n \
  --dtype float32 \
  --device cuda \
  --embed-e0 \
  --out model_static256.pt2
```

常用规则：

- `--elements` 的顺序必须和后续 LAMMPS `pair_coeff` 里的元素顺序一致。
- `--embed-e0` 会把 atomic reference energy 嵌入导出 core，使 LAMMPS `pe` 返回绝对能量。
- `--dynamic` 适合变原子数/变邻接规模的部署；如果 dynamic export 失败，先用 `--static-n` 做固定形状验证。
- `--degree` 是静态导出时为邻居数预留的 directed degree 上限。实际运行不能超过导出时的容量。
- 导出日志会比较 eager 和 compiled `.pt2` 的能量/力；只有通过数值检查后才应进入 LAMMPS。

### 7.2 从原生 MACE 预训练模型导出

先把原生 `mace-torch` `ScaleShiftMACE` 对象转成 MACE-ICTC checkpoint：

```bash
mff-convert-mace \
  --mace-model /path/to/mace.model \
  --out mace_ictc.pth \
  --product-backend ictd-bridge-u \
  --dtype float64 \
  --device cpu
```

很老的公开预训练模型可能只能在对应历史 `mace-torch`/`e3nn` 环境里完成 load 和 conversion。
如果要做原生 MACE 数值对应审计，先保留 float64 checkpoint；如果要部署到 LAMMPS，
用同一条转换命令把 `--dtype` 改成 `float32` 再导出 `.pt2`：

```bash
mff-export-aoti \
  --checkpoint mace_ictc_f32.pth \
  --elements H,C,N,O,F,P,S,Cl,Br,I \
  --atoms 6 \
  --degree 5 \
  --static-n \
  --dtype float32 \
  --device cuda \
  --embed-e0 \
  --out mace_ictc_static6.pt2
```

### 7.3 Legacy TorchScript `core.pt`

TorchScript 仍可通过 `mff-export-core` 导出：

```bash
mff-export-core \
  --checkpoint /path/to/model.pth \
  --elements H O \
  --device cuda \
  --dtype float32 \
  --out core.pt
```

除非你明确需要 legacy TorchScript runtime，否则优先使用 7.1 的 `.pt2` 路径。

---

## 8. 运行 LAMMPS

### 8.1 设置环境变量（LibTorch 动态库）

```bash
export LD_LIBRARY_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):$LD_LIBRARY_PATH"
```

### 8.2 最小 LAMMPS 输入示例

创建 `in.mfftorch`：

```lammps
units metal
atom_style atomic
boundary p p p

region box block 0 40 0 40 0 40
create_box 2 box
create_atoms 1 random 200 12345 box
create_atoms 2 random 100 12346 box
mass 1 1.008
mass 2 15.999

neighbor 1.0 bin

pair_style mff/torch 5.0 cuda
pair_coeff * * /path/to/model.pt2 H O

velocity all create 300 42
fix 1 all nve
thermo 20
thermo_style custom step temp pe etotal fmax
run 200
```

当前 `.pt2` 部署路径支持能量和力输出。Physical tensor outputs 不是当前公开支持的
LAMMPS 接口；不要在生产输入中使用 `compute mff/torch/phys`。

若导出的模型 core 来自带外场架构的 checkpoint，可在运行时通过 LAMMPS equal-style 变量传入 rank-1 外场：

```lammps
variable Ex equal 0.0
variable Ey equal 0.0
variable Ez equal 0.01

pair_style mff/torch 5.0 cuda field v_Ex v_Ey v_Ez
pair_coeff * * /path/to/model.pt2 H O
```

说明：
- `field v_Ex v_Ey v_Ez` 会在每个 force call 重新求值，可用于时间相关外场。
- 也支持 rank-2 外场：
  - `field9`：全量 `3x3`，行主序 `xx xy xz yx yy yz zx zy zz`
  - `field6`：对称 `3x3` 简写，顺序 `xx yy zz xy xz yz`
- 若模型 core 需要外场但未提供 `field ...`，LAMMPS 初始化时会报错；反之，普通模型 core 也不能搭配 `field ...` 使用。

rank-2 示例：

```lammps
variable Txx equal 1.0
variable Txy equal 0.0
variable Txz equal 0.0
variable Tyx equal 0.0
variable Tyy equal 1.0
variable Tyz equal 0.0
variable Tzx equal 0.0
variable Tzy equal 0.0
variable Tzz equal 1.0

pair_style mff/torch 5.0 cuda field9 v_Txx v_Txy v_Txz v_Tyx v_Tyy v_Tyz v_Tzx v_Tzy v_Tzz
pair_coeff * * /path/to/model.pt2 H O
```

### 8.2.1 独立色散邻居表 / MBD

带 `--long-range-dispersion-mode mbd` 或 `--long-range-dispersion-mode mbd-slq`
训练并通过 `export_libtorch_core.py` 导出的 TorchScript `core.pt` 会接收第二套色散 edge list。
`mbd` 是 dense eigensolve oracle；`mbd-slq` 是 cutoff matrix-free stochastic-Lanczos 近似，
可用 `--dispersion-slq-num-probes` 和 `--dispersion-slq-lanczos-steps` 调节成本/精度。
LAMMPS 侧用
`dispersion <cutoff>` 指定色散邻居表 cutoff，主 cutoff 仍用于 message passing：

```lammps
pair_style mff/torch 4.0 cuda dispersion 6.0
pair_coeff * * /path/to/core.pt H C N O
```

这会让 LAMMPS 请求 `max(4.0, 6.0)` 的候选邻居，在 `mff/torch` / `mff/torch/kk`
内部拆成主 edge list 和 dispersion edge list。对 `mbd` / `mbd-slq` core，若省略
`dispersion <cutoff>`，初始化/运行会直接报错；不要让 MBD 静默复用短程 message-passing edge list。
这里的 `<cutoff>` 必须等于 checkpoint/export metadata 里的 `dispersion_cutoff`；MBD 的 cutoff
定义了耦合矩阵的稀疏图，部署时改 cutoff 等价于换了训练时的长程色散模型。
显式的 dispersion list 可以为空（例如低密度或单原子体系），但不能缺省。

AOTI `.pt2` 对 `mbd-slq` 会导出 `dispersion_edges 1` sidecar metadata，并接收第二套
dispersion edge list；`mff/torch` 和 `mff/torch/kk` build 都支持这一路径。导出时会把 SLQ 切到
atom-rademacher probes + Newton-Schulz quadrature，以避免有限 Cartesian probe 破坏旋转等变性，
并避开 AOTI 对本征分解的限制。dense `mbd` 仍只作为 dense eigensolve oracle，不建议也不会导出为
AOTI 部署 core。

### 8.3 输出压力/应力（virial）

若编译时启用了 `MFF_ENABLE_VIRIAL=ON`，可在 `thermo_style` 中加入压力与应力分量：

```lammps
thermo_style custom step temp pe ke etotal press pxx pyy pzz pxy pxz pyz
thermo 20
```

| 字段 | 含义 |
|------|------|
| `press` | 总压力（标量） |
| `pxx pyy pzz` | 压力张量对角分量 |
| `pxy pxz pyz` | 压力张量非对角分量（剪切应力） |

未启用 virial 时，`press` 等只有动能贡献，数值不完整。

### 8.4 使用 Kokkos GPU 运行

```bash
/path/to/lammps/build-mfftorch/lmp \
  -k on g 1 \
  -sf kk \
  -pk kokkos newton off neigh full \
  -in in.mfftorch
```

说明：

- `-k on g 1`：启用 Kokkos，使用 1 块 GPU
- `-sf kk`：将 `pair_style mff/torch` 映射到 Kokkos 变体 `mff/torch/kk`
- `-pk kokkos newton off neigh full`：Kokkos 使用 `neigh full` 时必须 `newton off`

### 8.5 多 MPI rank / 多 GPU 运行

`USER-MFFTORCH` 的多卡运行方式是 MPI domain decomposition：通常每张 GPU
对应一个 MPI rank。不要把它理解成一个 LAMMPS rank 同时调度多张 GPU。

普通 `mff/torch` 路径：

```bash
export MFF_DEBUG_BUNDLE=1
mpirun -np 2 /path/to/lammps/build-mfftorch/lmp -in in.mfftorch
```

LAMMPS input 中使用：

```lammps
pair_style mff/torch 5.0 cuda
pair_coeff * * /path/to/model.pt2 H O
```

普通路径会从本地 rank 环境变量选择 CUDA 设备，例如 `SLURM_LOCALID`、
`LOCAL_RANK`、`OMPI_COMM_WORLD_LOCAL_RANK`、`MV2_COMM_WORLD_LOCAL_RANK` 或
`MPI_LOCALRANKID`。设置 `MFF_DEBUG_BUNDLE=1` 后，初始化日志会打印 requested
device 和 selected device，生产前要确认不同 local rank 选择了不同 GPU。

Kokkos 路径：

```bash
export MFF_DEBUG_BUNDLE=1
mpirun -np 2 /path/to/lammps/build-mfftorch-kk/lmp \
  -k on g 2 \
  -sf kk \
  -pk kokkos newton off neigh full \
  -in in.mfftorch
```

input 中可以显式使用：

```lammps
pair_style mff/torch/kk 5.0 cuda
pair_coeff * * /path/to/model.pt2 H O
```

也可以写 `pair_style mff/torch`，让 `-sf kk` 在支持的 build 中映射到
`mff/torch/kk`。当前 Kokkos 设备映射对单节点、每 GPU 一个 MPI rank 的情况是目标路径；
多节点 Kokkos 运行必须先做 local-rank/device 映射验证。

Slurm 示例：

```bash
#!/bin/bash
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

source /path/to/conda.sh
conda activate mff
export LD_LIBRARY_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"
export MFF_DEBUG_BUNDLE=1

srun /path/to/lammps/build-mfftorch/lmp -in in.mfftorch
```

如果用 static-N `.pt2`，导出时的 `--atoms` 和 `--degree` 必须覆盖每个 MPI rank
上的 local atoms、ghost atoms 和边数。长 MD 前应先比较 `-np 1` 和 `-np N` 的
`run 0` energy/force。fp32 下不同 domain decomposition 可能造成很小的归约顺序差异；
如果能量或力出现明显偏差，先检查 cutoff、neighbor skin、ghost halo 和 `.pt2` 容量。

### 8.6 仅 CPU 运行（无 Kokkos）

```bash
/path/to/lammps/build-mfftorch/lmp -in in.mfftorch
```

---

## 9. 编译和运行 smoke test

### 9.1 编译检查

从 LAMMPS 源码根目录编译：

```bash
export LIBTORCH_PREFIX="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build-mfftorch -j 8
cmake --build build-mfftorch-kk -j 8
```

如果链接阶段出现 CUDA runtime 版本冲突 warning，需要核对 Python 环境中的 PyTorch CUDA
版本和系统 CUDA runtime。warning 不一定导致运行失败，但不应在生产环境中忽略。

### 9.2 OFF23 converted `.pt2` smoke

先按第 7 节导出一个 fixed-N `.pt2`。然后准备最小 LAMMPS input：

```lammps
units metal
atom_style atomic
boundary p p p

read_data system.data
neighbor 1.0 bin

pair_style mff/torch 4.5 cuda
pair_coeff * * /path/to/MACE-OFF23_small_ictd_bridge_u_f32_static6.pt2 H C N O

thermo 1
thermo_style custom step temp pe etotal fmax
run 0
```

运行：

```bash
export LD_LIBRARY_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"
/path/to/lammps/build-mfftorch/lmp -in in.off23_static6
/path/to/lammps/build-mfftorch-kk/lmp -in in.off23_static6
```

4090 验证记录中，普通 build 和 Kokkos build 均能加载 `.pt2`。fresh OFF23 static-6
导出的 `run 0` 与 Python checkpoint 对应到：

| 量 | LAMMPS `mff/torch` | Python checkpoint |
|---|---:|---:|
| energy (eV) | `-6633.036` | `-6633.03613281` |
| 最大绝对力分量 (eV/A) | `11.767612` | `11.76760674` |

这里 LAMMPS `fmax` 是最大绝对力分量。Python 侧应比较 `max(abs(forces))`，不要比较逐原子力向量范数最大值。

### 9.3 原生 MACE 到 ICTC 的数值对应

转换桥接本身应先在 Python 中验证，再进入 LAMMPS。OFF23 small 的 float64 验证记录为：

- native `mace-torch` vs converted ICTC，苯 same-frame 轨迹；
- 最大能量绝对差：`2.73e-12 eV`；
- 最大力分量差：`4.44e-15 eV/A`。

这个测试验证 checkpoint conversion；AOTI `.pt2` 仍要单独验证 eager-vs-compiled 和
LAMMPS-vs-Python，因为编译器 lowering 会改变浮点运算顺序。

---

## 10. 故障排查

### 10.1 `torch/torch.h: No such file or directory`

- 原因：LibTorch 未被找到。
- 解决：确保 `CMAKE_PREFIX_PATH` 包含 `torch.utils.cmake_prefix_path` 的路径，且 `USER-MFFTORCH` 已加入 `PKG_WITH_INCL` 的 include 列表。

### 10.2 `Unrecognized pair style 'mff/torch'`

- 原因：USER-MFFTORCH 未正确编译或未启用。
- 解决：确认 `-D PKG_USER-MFFTORCH=ON`，且 `USER-MFFTORCH` 已加入 `STANDARD_PACKAGES`，`src/USER-MFFTORCH/` 存在且包含 `pair_mff_torch.cpp` 等文件。

### 10.3 `error while loading shared libraries: libtorch.so`

- 原因：运行时找不到 LibTorch 动态库。
- 解决：设置 `LD_LIBRARY_PATH`（见 8.1 节）。

### 10.4 `Must use 'newton off' with KOKKOS package option 'neigh full'`

- 原因：Kokkos 配置要求。
- 解决：在 `lmp` 命令行加上 `-pk kokkos newton off neigh full`。

### 10.5 `Kokkos_ARCH` 与 GPU 不匹配

- 现象：运行时报错或 JIT 编译耗时很长。
- 解决：根据 GPU 型号选择正确的 `Kokkos_ARCH_XXX`（见 5.2 节）。

### 10.6 能量/温度恒不变

- 若使用 dummy 模型，输出常数为预期行为（dummy 模型梯度近似为 0）。
- 使用真实训练模型后，能量和温度会随时间变化。

### 10.7 需要压力/应力但未启用 virial

- 现象：`thermo_style` 含 `press` 时，数值只有动能贡献，不完整。
- 解决：用 `-D MFF_ENABLE_VIRIAL=ON` 重新配置并编译，virial 在 GPU 上计算，几乎无性能损失。

---

## 11. 附录：目录结构速览

```
ICTC-O3-HP/
├── lammps_user_mfftorch/
│   ├── src/USER-MFFTORCH/          # 源码
│   ├── cmake/Modules/Packages/USER-MFFTORCH.cmake
│   ├── cmake/Packages/USER-MFFTORCH.cmake  # 部分版本
│   ├── examples/
│   └── docs/BUILD_AND_RUN.md       # 本文档
└── mace_ictc/
│   └── cli/
│       ├── convert_mace.py          # 原生 MACE -> MACE-ICTC
│       ├── export_aoti_core.py      # 导出 .pt2
│       └── export_libtorch_core.py  # legacy TorchScript core
```
