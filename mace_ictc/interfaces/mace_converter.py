"""Load a real e3nn MACE (``mace-torch``) model into a MACE-ICTC model.

``convert_mace_to_ictd(mace_model, ictd_model)`` copies every learnable weight from a
``ScaleShiftMACE`` (built with the matching small config) into a
:class:`mace_ictc.models.pure_cartesian_ictd_fix.PureCartesianICTDFix`. The exact
default/recommended target is ``ictd_fix_product_backend="ictd-bridge-u"``, which uses
the MACE U tensors folded into the ICTC basis. ``native-mace`` is also supported as a
direct reference backend.

Why this is possible
--------------------
The two backbones correspond block-by-block (see ``docs``/the task table). The only
non-trivial pieces are:

1. **e3nn ``o3.Linear`` blocks** (``node_embedding``, ``linear_up``, the post-conv
   ``linear``, the product ``linear``) carry e3nn's path normalization and an internal
   flat-weight layout. Instead of reverse-engineering that layout, each such block is
   reduced to its *effective* per-``l`` channel-mixing matrix by probing it with an
   identity input (``W_eff = block(I)``). Because an SO(3)-equivariant linear is
   block-diagonal in ``l`` and m-diagonal, ``W_eff`` restricted to one ``l`` block is
   exactly ``M_l ⊗ I_(2l+1)`` in the channel-major ``(c, m)`` layout that ICTC also uses,
   so ``M_l`` drops straight into the ICTC adapter (``nn.Linear`` with ``weight = M_l.T``).
   This is exact to machine precision and version-independent.

2. **The convolution tensor product** (MACE ``conv_tp`` ↔ ICTC ``tp``). Both evaluate the
   same ``(l1, l2) -> l3`` paths but (a) in a different path order and (b) with a different
   per-path scalar (e3nn's element path-normalisation, which ICTC's CG basis does not
   carry). Both the path permutation and the per-path scalar are *calibrated empirically*
   at convert time: a few random inputs are pushed through MACE's isolated ``conv_tp`` and
   ICTC's ``tp`` and the per-path scalar ``c_p`` is recovered by least squares. The
   MACE radial MLP (``conv_tp_weights``) is then copied into ICTC's ``fc`` with the row for
   path ``p`` scaled by ``c_p`` and reordered into ICTC's path order. ICTC's own per-path
   base weight ``tp.weight`` is set to all-ones (MACE has no equivalent).

3. **The symmetric contraction** uses the bridge-U path with MACE U tensors pre-folded
   into the ICTC basis by default. The literal MACE code path
   (``ictd_fix_product_backend="native-mace"``) is kept as a reference backend. In both
   cases ``weights_max`` / ``weights.{k}`` copy 1:1. The product ``linear`` is an e3nn
   ``o3.Linear`` on both sides -> copied as in (1).

Everything is computed in float64. The angular basis differs by a fixed orthogonal ``Q``
per ``l`` (ICTC vs e3nn), but energy/forces are SO(3) invariants and therefore identical;
``Q`` is only used internally to bridge the calibration probes.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import torch
from e3nn import o3

from mace_ictc.models.ictd_irreps import direction_harmonics_all
from mace_ictc.mace_basis import orthogonal_Q_blocks


def _compute_normalize2mom_silu_k() -> float:
    """e3nn wraps the radial-MLP activation in ``normalize2mom`` (a 2nd-moment rescale), so the
    effective activation is ``K * silu(.)`` with a constant ``K``. Recover ``K`` exactly from the
    installed e3nn rather than hard-coding it (so the converter tracks the library)."""
    import torch as _t
    from e3nn.math import normalize2mom

    act = normalize2mom(_t.nn.functional.silu)
    x = _t.tensor([1.0, 2.0, -0.7, 3.3], dtype=_t.float64)
    ratio = act(x) / _t.nn.functional.silu(x)
    return float(ratio.mean().item())


_NORMALIZE2MOM_SILU_K = _compute_normalize2mom_silu_k()


# --------------------------------------------------------------------------------------
# helpers: e3nn o3.Linear -> ICTC per-l effective channel matrices
# --------------------------------------------------------------------------------------
def _e3nn_linear_effective_matrices(
    linear: o3.Linear,
    *,
    channels_in_by_l: Dict[int, int],
    channels_out_by_l: Dict[int, int],
) -> Dict[int, torch.Tensor]:
    """Return ``{l: M_l}`` with ``M_l`` of shape ``(C_in_l, C_out_l)`` such that the e3nn
    linear acts on l-block ``x`` (channel-major) as ``out[..., c', m] = sum_c x[..., c, m] M_l[c, c']``.

    Recovered by probing ``W_eff = linear(I)``; ``W_eff`` is block-diagonal in ``l`` and the
    per-l block is ``M_l ⊗ I_(2l+1)`` in the ``(c, m)`` channel-major layout, so reading the
    ``m == 0`` slice yields ``M_l`` exactly. No bias (e3nn linears here are bias-free).
    """
    irr_in = linear.irreps_in
    irr_out = linear.irreps_out
    dtype = next(linear.parameters()).dtype
    device = next(linear.parameters()).device
    with torch.no_grad():
        w_eff = linear(torch.eye(irr_in.dim, dtype=dtype, device=device))  # (in_dim, out_dim)

    # offsets of each l block on the input and output sides (channel-major within a block)
    in_off: Dict[int, int] = {}
    off = 0
    for mul, ir in irr_in:
        in_off.setdefault(ir.l, off)
        off += mul * ir.dim
    out_off: Dict[int, int] = {}
    off = 0
    for mul, ir in irr_out:
        out_off.setdefault(ir.l, off)
        off += mul * ir.dim

    mats: Dict[int, torch.Tensor] = {}
    for l in sorted(set(channels_in_by_l) & set(channels_out_by_l)):
        c_in = int(channels_in_by_l[l])
        c_out = int(channels_out_by_l[l])
        if c_in == 0 or c_out == 0:
            continue
        m = 2 * l + 1
        i0 = in_off[l]
        o0 = out_off[l]
        blk = w_eff[i0 : i0 + c_in * m, o0 : o0 + c_out * m]
        # channel-major (c, m); m-diagonal -> read m==0 slice
        blk = blk.reshape(c_in, m, c_out, m)
        mats[l] = blk[:, 0, :, 0].contiguous()  # (C_in_l, C_out_l)
    return mats


def _copy_into_so3_channel_linear(adapters_module, mats: Dict[int, torch.Tensor]) -> None:
    """Write ``{l: M_l}`` (shape ``(C_in_l, C_out_l)``) into an ``EquivariantChannelLinearSO3``
    / ``EquivariantChannelLinearSO3Rect``-style module whose ``.adapters[str(l)]`` is an
    ``nn.Linear(C_in, C_out, bias=False)``. The ICTC adapter computes ``y = x @ W.T`` over the
    channel axis, so ``W = M_l.T``."""
    adapters = adapters_module.adapters
    for l_str, lin in adapters.items():
        l = int(l_str)
        if l not in mats:
            continue
        with torch.no_grad():
            lin.weight.copy_(mats[l].T.to(dtype=lin.weight.dtype, device=lin.weight.device))
        if getattr(lin, "bias", None) is not None:
            with torch.no_grad():
                lin.bias.zero_()


def _copy_into_path_preserving_linear(message_linear, mats_by_l: Dict[int, torch.Tensor]) -> None:
    """Write per-l effective matrices into a ``PathPreservingLinearSO3`` whose
    ``.weights[str(l)]`` has shape ``(out_channels, in_channels_l)`` and computes
    ``out[..., o, m] = sum_c W[o, c] x[..., c, m]`` (``einsum('oc,ncm->nom')``). So
    ``W = M_l.T`` with ``M_l`` of shape ``(C_in_l, C_out_l)``."""
    for l_str, w in message_linear.weights.items():
        l = int(l_str)
        if l not in mats_by_l:
            continue
        with torch.no_grad():
            w.copy_(mats_by_l[l].T.to(dtype=w.dtype, device=w.device))


# --------------------------------------------------------------------------------------
# helpers: conv_tp path permutation + per-path scalar calibration
# --------------------------------------------------------------------------------------
def _mace_conv_tp_paths(conv_tp: o3.TensorProduct) -> List[Tuple[int, int, int]]:
    """``(l1, l2, l3)`` for each instruction of a MACE ``conv_tp``, in instruction order."""
    out = []
    for ins in conv_tp.instructions:
        l1 = conv_tp.irreps_in1[ins.i_in1][1].l
        l2 = conv_tp.irreps_in2[ins.i_in2][1].l
        l3 = conv_tp.irreps_out[ins.i_out][1].l
        out.append((int(l1), int(l2), int(l3)))
    return out


def _calibrate_conv_tp(
    conv_tp: o3.TensorProduct,
    ictd_tp,
    *,
    input_lmax: int,
    lmax: int,
    channels: int,
    n_probe: int = 6,
    seed: int = 12345,
) -> Tuple[List[int], Dict[Tuple[int, int, int], float]]:
    """Determine, for the MACE ``conv_tp`` / ICTC ``tp`` pair:

    - ``perm``: for each MACE path index, the corresponding ICTC path index.
    - ``c_by_path``: per-(l1,l2,l3) scalar with ``mace_path_out == c * (ictd_path_out @ Q_l3)``.

    Recovered by pushing random node features + a random direction through the *isolated*
    MACE ``conv_tp`` (with random per-path channel weights) and ICTC ``tp`` (gates = those
    same weights, ``tp.weight == 1``), then least-squares per path. Residuals are ~1e-15.
    """
    dtype = ictd_tp.weight.dtype
    device = ictd_tp.weight.device
    C = int(channels)
    Q = orthogonal_Q_blocks(lmax, dtype=dtype, device=device)

    mace_paths = _mace_conv_tp_paths(conv_tp)
    ictd_paths = [tuple(p) for p in ictd_tp.paths]
    perm = [ictd_paths.index(p) for p in mace_paths]

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    N = int(n_probe)
    # node features (ICTC basis) up to input_lmax; zero above
    h_ictd = {
        l: (
            torch.randn(N, C, 2 * l + 1, generator=g, dtype=dtype).to(device)
            if l <= input_lmax
            else torch.zeros(N, C, 2 * l + 1, dtype=dtype, device=device)
        )
        for l in range(lmax + 1)
    }
    h_e3nn_blocks = [torch.einsum("ncm,mp->ncp", h_ictd[l], Q[l]) for l in range(lmax + 1)]
    h_e3nn_flat = torch.cat([b.reshape(N, C * (2 * l + 1)) for l, b in enumerate(h_e3nn_blocks)], dim=-1)
    # MACE in1 layout matches irreps_in1.dim (input_lmax block)
    h_e3nn_in = h_e3nn_flat[:, : conv_tp.irreps_in1.dim]

    ndir = torch.randn(N, 3, generator=g, dtype=dtype).to(device)
    ndir = ndir / ndir.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    Y_ictd = direction_harmonics_all(ndir, lmax)
    Y_e3nn = o3.spherical_harmonics(
        o3.Irreps.spherical_harmonics(lmax), ndir, normalize=True, normalization="component"
    )

    num_paths = ictd_tp.num_paths
    wpath = torch.randn(N, num_paths, C, generator=g, dtype=dtype).to(device)
    w_e = torch.cat([wpath[:, ictd_paths.index(p), :] for p in mace_paths], dim=-1)  # (N, P*C)
    with torch.no_grad():
        out_e = conv_tp(h_e3nn_in, Y_e3nn, w_e)  # (N, expanded_out_dim)

    # split MACE output per path (instruction order, each path is a C*(2l3+1) block)
    oe_paths = []
    idx = 0
    for (_l1, _l2, l3) in mace_paths:
        d = C * (2 * l3 + 1)
        oe_paths.append(out_e[:, idx : idx + d].reshape(N, C, 2 * l3 + 1))
        idx += d

    gates = wpath.reshape(N, num_paths * C)
    edge_attrs = {l: Y_ictd[l].to(dtype=dtype, device=device).unsqueeze(-2) for l in range(lmax + 1)}
    with torch.no_grad():
        eb = ictd_tp(h_ictd, edge_attrs, gates)  # dict l -> (N, C*count_l, 2l+1)

    c_by_path: Dict[Tuple[int, int, int], float] = {}
    max_resid = 0.0
    for mi, (l1, l2, l3) in enumerate(mace_paths):
        pidx = ictd_paths.index((l1, l2, l3))
        off = int(ictd_tp.path_offset[pidx])
        block = eb[l3][:, off * C : (off + 1) * C, :]
        block_e = torch.einsum("ncm,mp->ncp", block, Q[l3])
        num = (oe_paths[mi] * block_e).sum()
        den = (block_e * block_e).sum().clamp_min(1e-30)
        c = float((num / den).item())
        c_by_path[(l1, l2, l3)] = c
        resid = (oe_paths[mi] - c * block_e).abs().max().item()
        max_resid = max(max_resid, resid)
    if max_resid > 2e-6:
        raise RuntimeError(
            f"conv_tp calibration residual too large ({max_resid:.2e}); the MACE/ICTC path "
            "bases do not correspond as expected."
        )
    return perm, c_by_path


# --------------------------------------------------------------------------------------
# main converter
# --------------------------------------------------------------------------------------
def convert_mace_to_ictd(mace_model, ictd_model) -> Dict[str, object]:
    """Copy weights from a built ``ScaleShiftMACE`` into a built ``PureCartesianICTDFix``
    so they produce the same interaction energy and forces for the supported MACE-style
    baseline configs. Operates in float64, in place on ``ictd_model``.

    Returns a small report dict (per-block notes + the calibrated conv_tp scalars) for
    logging / debugging. The MACE scale/shift is installed into ``ictd_model`` so the model
    output is already the scaled per-atom interaction energy. E0 (atomic energies) is returned
    in the report for the caller's total-energy assembly.
    """
    mace_model = mace_model.double().eval()
    ictd_model = ictd_model.double().eval()

    C = int(ictd_model.channels)
    report: Dict[str, object] = {"blocks": {}, "conv_tp": {}}
    product_backend = str(getattr(ictd_model, "ictd_fix_product_backend", ""))
    if product_backend not in {"native-mace", "ictd-bridge-u", "cueq"}:
        raise NotImplementedError(
            "convert_mace_to_ictd currently gives exact mace-torch parity only for "
            "ictd_fix_product_backend='native-mace', 'ictd-bridge-u', or 'cueq'. "
            f"Got {product_backend!r}; the pure-U backend is close but not bit-exact to mace-torch."
        )

    # ----- node embedding: e3nn Linear 4x0e -> 16x0e  ==>  nn.Linear(num_elements, C) -----
    ne_mats = _e3nn_linear_effective_matrices(
        mace_model.node_embedding.linear,
        channels_in_by_l={0: mace_model.node_embedding.linear.irreps_in.dim},
        channels_out_by_l={0: C},
    )
    M0 = ne_mats[0]  # (num_elements, C)
    with torch.no_grad():
        # ICTC node_embedding is nn.Linear(num_elements, C): out = onehot @ W.T => W = M0.T
        ictd_model.node_embedding.weight.copy_(
            M0.T.to(dtype=ictd_model.node_embedding.weight.dtype)
        )
    report["blocks"]["node_embedding"] = "e3nn Linear effective-matrix -> nn.Linear.weight = M.T"

    # ----- per interaction -----
    for i, (m_inter, ictd_inter) in enumerate(zip(mace_model.interactions, ictd_model.interactions)):
        input_lmax = int(ictd_inter.input_lmax)
        target_lmax = int(ictd_inter.target_lmax)
        edge_lmax = int(ictd_inter.lmax)

        # linear_up: e3nn Linear (in irreps over input_lmax) -> ICTC linear_up adapters
        lu_mats = _e3nn_linear_effective_matrices(
            m_inter.linear_up,
            channels_in_by_l={l: C for l in range(input_lmax + 1)},
            channels_out_by_l={l: C for l in range(input_lmax + 1)},
        )
        _copy_into_so3_channel_linear(ictd_inter.linear_up, lu_mats)

        # conv_tp: calibrate path permutation + per-path scalar
        perm, c_by_path = _calibrate_conv_tp(
            m_inter.conv_tp,
            ictd_inter.tp,
            input_lmax=input_lmax,
            lmax=edge_lmax,
            channels=C,
        )
        report["conv_tp"][f"interaction[{i}]"] = {
            "perm_mace_to_ictd": perm,
            "c_by_path": {str(k): round(v, 10) for k, v in c_by_path.items()},
        }

        # ICTC per-path base weight has no MACE equivalent -> all ones (path weight = fc alone)
        with torch.no_grad():
            ictd_inter.tp.weight.fill_(1.0)

        # radial MLP: MACE conv_tp_weights (e3nn FullyConnectedNet, no bias) -> ICTC fc.
        #
        # e3nn `_Layer`(var_in=var_out=1): with act -> out = K * silu(x @ (W_mace / sqrt(h_in)));
        # without act (last) -> out = x @ W_mace. Here W_mace is stored as (h_in, h_out) and
        # the activation is normalize2mom(silu) = K * silu(.) with the CONSTANT K below.
        # ICTC fc is [Linear, SiLU, Linear, SiLU, Linear, SiLU, Linear] (plain SiLU, bias).
        #
        # To make the two networks identical we (1) transpose W_mace to nn.Linear (out,in)
        # layout and divide by sqrt(h_in) so the SiLU arguments match, which leaves each
        # activated ICTC layer's output a factor 1/K below e3nn; (2) fold that 1/K into the
        # NEXT layer by multiplying its weight by K. The last (un-activated) layer therefore
        # carries one extra K (to undo the previous activated layer's 1/K). All biases -> 0.
        mace_layers = [layer for layer in m_inter.conv_tp_weights]  # e3nn _Layer modules
        ictd_fc_linears = [ictd_inter.fc[k] for k in (0, 2, 4, 6)]
        assert len(mace_layers) == len(ictd_fc_linears) == 4
        with torch.no_grad():
            for k in range(3):  # activated layers 0,1,2
                h_in = float(mace_layers[k].h_in)
                w = mace_layers[k].weight.to(dtype=ictd_fc_linears[k].weight.dtype)  # (h_in, h_out)
                scale = 1.0 / (h_in ** 0.5)
                ictd_fc_linears[k].weight.copy_((w * scale).T.contiguous())
                if ictd_fc_linears[k].bias is not None:
                    ictd_fc_linears[k].bias.zero_()
            # last layer (no act): e3nn computes out = x @ (W_mace / sqrt(h_in * var_in/var_out))
            # = x @ (W_mace / sqrt(h_in)) with var_in=var_out=1. ICTC now uses the same
            # normalize2mom(silu) activations, so no activation scale is folded into this layer.
            h_in_last = float(mace_layers[3].h_in)
            last_w = (
                mace_layers[3].weight * (1.0 / (h_in_last ** 0.5))
            ).to(dtype=ictd_fc_linears[3].weight.dtype).T.contiguous()  # (P*C, 64) MACE path order
            mace_paths = _mace_conv_tp_paths(m_inter.conv_tp)
            ictd_paths = [tuple(p) for p in ictd_inter.tp.paths]
            new_last = torch.zeros_like(ictd_fc_linears[3].weight)  # (P*C, 64) ICTC path order
            for mace_pos, path in enumerate(mace_paths):
                ictd_pos = ictd_paths.index(path)
                c = float(c_by_path[path])
                new_last[ictd_pos * C : (ictd_pos + 1) * C, :] = (
                    c * last_w[mace_pos * C : (mace_pos + 1) * C, :]
            )
            ictd_fc_linears[3].weight.copy_(new_last)
            if ictd_fc_linears[3].bias is not None:
                ictd_fc_linears[3].bias.zero_()
        report["blocks"][f"interaction[{i}].fc"] = (
            "e3nn FC -> ICTC fc: W.T/sqrt(h_in), normalize2mom(silu) K folded into next layer; "
            "last layer reordered to ICTC path order, scaled by c_p; biases zeroed"
        )

        # post-conv linear: MACE `linear` (expanded mul irreps -> hidden) maps each l3 block of
        # width (count_l3 * C) to C. ICTC message_linear is a PathPreservingLinearSO3 with the
        # SAME per-l input width (channels * path_counts_by_l[l]) -> C. Both mix the per-path
        # channel blocks of a given l3 into C outputs. Extract MACE linear's per-l effective
        # matrix (in: count_l3*C, out: C) and copy.
        mace_lin = m_inter.linear
        in_counts = Counter(l3 for _l1, _l2, l3 in _mace_conv_tp_paths(m_inter.conv_tp))
        ml_mats = _e3nn_linear_effective_matrices(
            mace_lin,
            channels_in_by_l={l: int(in_counts.get(l, 0)) * C for l in range(target_lmax + 1)},
            channels_out_by_l={l: C for l in range(target_lmax + 1)},
        )
        # MACE expands paths-to-l3 in conv_tp instruction order; ICTC orders them by path_offset.
        # Reorder the input-channel axis of each M_l from MACE order to ICTC order before copy.
        ml_mats_ictd = _reorder_linear_input_paths(
            ml_mats, m_inter.conv_tp, ictd_inter.tp, channels=C, target_lmax=target_lmax
        )
        _copy_into_path_preserving_linear(ictd_inter.message_linear, ml_mats_ictd)
        report["blocks"][f"interaction[{i}].message_linear"] = (
            "MACE post-conv linear effective-matrix, input paths reordered MACE->ICTC"
        )

        # element self-connection: MACE skip_tp (FullyConnectedTensorProduct, uvw, node x 4x0e)
        # is ALWAYS purely l=0 here (node 0e x attrs 0e -> 0e). It is ADDED to the product output.
        # ICTC layers > 0 own an additive `self_connection` -> direct l=0 per-element copy.
        # The FIRST ICTC interaction has no additive sc path (`self_connection=None`,
        # `message_selector` multiplies the message); MACE's first-layer l=0 self-connection is a
        # pure per-element constant (node_feats[l0] is the element embedding) -> captured as an
        # additive per-element scalar vector and injected by the caller before the layer-0 readout
        # AND into interaction[1]'s input (see report['_first_layer_sc']).
        num_elements = int(mace_model.node_embedding.linear.irreps_in.dim)
        if ictd_inter.self_connection is not None:
            _copy_skip_tp_all_l(
                m_inter.skip_tp,
                ictd_inter.self_connection,
                num_elements=num_elements,
                channels=C,
                lmax=int(ictd_inter.sc_lmax),
            )
            report["blocks"][f"interaction[{i}].self_connection"] = (
                "MACE skip_tp per-l per-element matrices -> self_connection.weights"
            )
        else:
            if m_inter.__class__.__name__ == "RealAgnosticInteractionBlock":
                _copy_skip_tp_all_l(
                    m_inter.skip_tp,
                    ictd_inter.message_selector,
                    num_elements=num_elements,
                    channels=C,
                    lmax=int(ictd_inter.target_lmax),
                )
                _install_first_layer_sc0(ictd_model, layer_idx=i, sc0_by_element=None)
                report["blocks"][f"interaction[{i}].message_selector"] = (
                    "MACE non-residual first-layer skip_tp selector copied per l into "
                    "message_selector.weights"
                )
            elif m_inter.__class__.__name__ == "RealAgnosticResidualInteractionBlock":
                _set_element_linear_identity(ictd_inter.message_selector)
                sc0 = _mace_first_layer_sc_l0(
                    m_inter.skip_tp, mace_model=mace_model, channels=C, num_elements=num_elements
                )
                report.setdefault("_first_layer_sc", {})[i] = sc0
                _install_first_layer_sc0(ictd_model, layer_idx=i, sc0_by_element=sc0)
                report["blocks"][f"interaction[{i}].message_selector"] = (
                    "identity passthrough; MACE first-layer l=0 element self-connection (which this "
                    "ICTC baseline's first interaction omits) injected as an additive per-element bias "
                    "on product[{}] output l=0 via an export-safe model buffer".format(i)
                )
            else:
                raise NotImplementedError(
                    "convert_mace_to_ictd supports first interaction blocks "
                    "RealAgnosticInteractionBlock and RealAgnosticResidualInteractionBlock, got "
                    f"{m_inter.__class__.__name__!r}"
                )

    # ----- products (native-mace backend: 1:1 symmetric-contraction weights + e3nn linear) -----
    for i, (m_prod, ictd_prod) in enumerate(zip(mace_model.products, ictd_model.products)):
        _copy_symmetric_contraction(m_prod.symmetric_contractions, ictd_prod.symmetric_contractions)
        # product linear is e3nn o3.Linear on BOTH sides -> direct flat-weight copy
        with torch.no_grad():
            ictd_prod.linear.weight.copy_(
                m_prod.linear.weight.to(dtype=ictd_prod.linear.weight.dtype)
            )
        report["blocks"][f"product[{i}]"] = (
            "symmetric_contraction weights copied 1:1; o3.Linear flat-weight copied"
        )

    # ----- readouts -----
    if len(mace_model.readouts) != int(ictd_model.num_interaction):
        raise NotImplementedError(
            "convert_mace_to_ictd expects one MACE readout per interaction "
            f"(use_last_readout_only=False); got {len(mace_model.readouts)} readouts for "
            f"{ictd_model.num_interaction} interactions."
        )
    if len(ictd_model.layer_energy_readouts) != int(ictd_model.num_interaction) - 1:
        raise RuntimeError("ICTC intermediate readout count does not match num_interaction")

    for ridx, (m_readout, i_readout) in enumerate(
        zip(mace_model.readouts[:-1], ictd_model.layer_energy_readouts)
    ):
        if not hasattr(m_readout, "linear"):
            raise NotImplementedError(
                f"expected MACE readout[{ridx}] to be LinearReadoutBlock-like, got {type(m_readout).__name__}"
            )
        ro_mats = _e3nn_linear_effective_matrices(
            m_readout.linear,
            channels_in_by_l={0: C},
            channels_out_by_l={0: 1},
        )
        with torch.no_grad():
            i_readout.readout.weight.copy_(ro_mats[0].T.to(dtype=i_readout.readout.weight.dtype))
            if i_readout.readout.bias is not None:
                i_readout.readout.bias.zero_()
        report["blocks"][f"readout[{ridx}]"] = "MACE LinearReadout l=0 effective-matrix -> nn.Linear(C,1)"

    last_idx = len(mace_model.readouts) - 1
    last_readout = mace_model.readouts[-1]
    if not (hasattr(last_readout, "linear_1") and hasattr(last_readout, "linear_2")):
        raise NotImplementedError(
            f"expected final MACE readout[{last_idx}] to be NonLinearReadoutBlock-like, "
            f"got {type(last_readout).__name__}"
        )
    # MACE final readout: NonLinearReadoutBlock linear_1 (Cx0e->H x0e) -> e3nn
    # Activation(silu) -> linear_2 (H x0e -> 1x0e). ICTC uses the same
    # normalize2mom(silu), so no activation scale is folded into linear_2.
    H = last_readout.linear_1.irreps_out.dim
    ro_l1 = _e3nn_linear_effective_matrices(
        last_readout.linear_1, channels_in_by_l={0: C}, channels_out_by_l={0: H}
    )[0]
    ro_l2 = _e3nn_linear_effective_matrices(
        last_readout.linear_2, channels_in_by_l={0: H}, channels_out_by_l={0: 1}
    )[0]
    with torch.no_grad():
        ictd_model.last_layer_energy_readout.linear_1.weight.copy_(
            ro_l1.T.to(dtype=ictd_model.last_layer_energy_readout.linear_1.weight.dtype)
        )
        if ictd_model.last_layer_energy_readout.linear_1.bias is not None:
            ictd_model.last_layer_energy_readout.linear_1.bias.zero_()
        ictd_model.last_layer_energy_readout.linear_2.weight.copy_(
            ro_l2.T.to(dtype=ictd_model.last_layer_energy_readout.linear_2.weight.dtype)
        )
        if ictd_model.last_layer_energy_readout.linear_2.bias is not None:
            ictd_model.last_layer_energy_readout.linear_2.bias.zero_()
    report["blocks"][f"readout[{last_idx}]"] = (
        "MACE NonLinearReadout linear_1/linear_2 effective-matrices; normalize2mom(silu) K "
        "folded into linear_2 (l=0 scalars)"
    )

    # ----- E0 + scale/shift -----
    atomic_energies = mace_model.atomic_energies_fn.atomic_energies.detach().double().clone()
    report["atomic_energies"] = atomic_energies
    report["atomic_numbers"] = list(ictd_model.atomic_numbers)
    report["scale"] = float(mace_model.scale_shift.scale.detach().double().item())
    report["shift"] = float(mace_model.scale_shift.shift.detach().double().item())
    _install_energy_scale_shift(ictd_model, scale=report["scale"], shift=report["shift"])
    report["blocks"]["scale_shift"] = (
        "MACE ScaleShiftBlock copied into ICTC energy_output_scale/energy_output_shift buffers"
    )
    report["avg_num_neighbors"] = float(ictd_model.avg_num_neighbors)
    return report


def _reorder_linear_input_paths(
    mats: Dict[int, torch.Tensor],
    conv_tp: o3.TensorProduct,
    ictd_tp,
    *,
    channels: int,
    target_lmax: int,
) -> Dict[int, torch.Tensor]:
    """Reorder the *input-channel* axis of each per-l matrix from MACE conv_tp path order to
    ICTC path order. MACE's post-conv linear sees the expanded irreps in conv_tp instruction
    order (paths to a given l3 concatenated in that order); ICTC's message_linear sees them in
    ``tp.path_offset`` order. ``mats[l]`` has shape ``(count_l * C, C)``."""
    C = int(channels)
    mace_paths = _mace_conv_tp_paths(conv_tp)
    ictd_paths = [tuple(p) for p in ictd_tp.paths]
    # for each l, list the MACE input-slot order (which path each C-block is) ...
    out: Dict[int, torch.Tensor] = {}
    for l in range(target_lmax + 1):
        if l not in mats:
            continue
        # MACE input slots for l3==l, in instruction order
        mace_slots = [p for p in mace_paths if p[2] == l]
        # ICTC target order: by path_offset
        ictd_slots_by_off = sorted(
            [p for p in ictd_paths if p[2] == l],
            key=lambda p: ictd_tp.path_offset[ictd_paths.index(p)],
        )
        M = mats[l]  # (len(mace_slots)*C, C)
        new = torch.zeros_like(M)
        for ictd_pos, path in enumerate(ictd_slots_by_off):
            mace_pos = mace_slots.index(path)
            new[ictd_pos * C : (ictd_pos + 1) * C, :] = M[mace_pos * C : (mace_pos + 1) * C, :]
        out[l] = new
    return out


def _l0_offset(irreps) -> int:
    off = 0
    for mul, ir in irreps:
        if ir.l == 0:
            return off
        off += mul * ir.dim
    raise ValueError("no l=0 block")


def _skip_tp_l0_matrices(skip_tp, *, channels: int, num_elements: int) -> torch.Tensor:
    """Per-element l=0 channel matrix of a MACE ``skip_tp`` (uvw, node_feats x Z one-hot).
    Returns ``M`` of shape ``(num_elements, C_in_l0, C_out=C)`` with
    ``out_l0[c'] = sum_c node_l0[c] * M[e, c, c']``. Recovered by probing per element."""
    dtype = next(skip_tp.parameters()).dtype
    device = next(skip_tp.parameters()).device
    C = int(channels)
    in_off0 = _l0_offset(skip_tp.irreps_in1)
    out_off0 = _l0_offset(skip_tp.irreps_out)
    mats = torch.zeros(num_elements, C, C, dtype=dtype, device=device)
    for e in range(num_elements):
        attr = torch.zeros(C, num_elements, dtype=dtype, device=device)
        attr[:, e] = 1.0
        x = torch.zeros(C, skip_tp.irreps_in1.dim, dtype=dtype, device=device)
        for c in range(C):
            x[c, in_off0 + c] = 1.0  # l=0 is 1 component, channel-major
        with torch.no_grad():
            y = skip_tp(x, attr)  # (C, out_dim)
        mats[e] = y[:, out_off0 : out_off0 + C]  # (c_in, c_out)
    return mats


def _skip_tp_l_matrices(skip_tp, *, l: int, channels: int, num_elements: int) -> torch.Tensor:
    """Per-element channel matrix for one l block of a MACE ``skip_tp``."""
    dtype = next(skip_tp.parameters()).dtype
    device = next(skip_tp.parameters()).device
    C = int(channels)
    l = int(l)
    dim_l = 2 * l + 1
    in_off = None
    out_off = None
    off = 0
    for mul, ir in skip_tp.irreps_in1:
        if int(ir.l) == l:
            in_off = off
            break
        off += mul * ir.dim
    off = 0
    for mul, ir in skip_tp.irreps_out:
        if int(ir.l) == l:
            out_off = off
            break
        off += mul * ir.dim
    if in_off is None or out_off is None:
        raise ValueError(f"skip_tp does not contain l={l} in both input and output")

    mats = torch.zeros(num_elements, C, C, dtype=dtype, device=device)
    for e in range(num_elements):
        attr = torch.zeros(C, num_elements, dtype=dtype, device=device)
        attr[:, e] = 1.0
        x = torch.zeros(C, skip_tp.irreps_in1.dim, dtype=dtype, device=device)
        for c in range(C):
            x[c, in_off + c * dim_l] = 1.0  # m=0 component in channel-major block
        with torch.no_grad():
            y = skip_tp(x, attr)
        mats[e] = y[:, out_off : out_off + C * dim_l : dim_l]
    return mats


def _copy_skip_tp_l0(skip_tp, self_connection, *, num_elements: int, channels: int) -> None:
    """Write the MACE skip_tp l=0 per-element matrix into an ``ElementConditionedLinearSO3``
    ``self_connection`` (l=0 only). ICTC computes ``out[c'] = sum_c W[e, c', c] x[c]`` so
    ``W[e] = M[e].T``."""
    mats = _skip_tp_l0_matrices(skip_tp, channels=channels, num_elements=num_elements)
    with torch.no_grad():
        for e in range(num_elements):
            self_connection.weights["0"][e].copy_(mats[e].T.to(dtype=self_connection.weights["0"].dtype))
        if getattr(self_connection, "bias", None) is not None:
            for v in self_connection.bias.values():
                v.zero_()


def _copy_skip_tp_all_l(skip_tp, module, *, num_elements: int, channels: int, lmax: int) -> None:
    """Copy a MACE ``skip_tp`` into an ``ElementConditionedLinearSO3`` for all available l blocks."""
    full_weights = {}
    for l in range(int(lmax) + 1):
        mats = _skip_tp_l_matrices(
            skip_tp,
            l=l,
            channels=channels,
            num_elements=num_elements,
        )
        full_weights[str(l)] = mats.transpose(-1, -2)
    set_from_full = getattr(module, "set_from_full_weights", None)
    if callable(set_from_full):
        set_from_full(full_weights)
        return
    with torch.no_grad():
        for l in range(int(lmax) + 1):
            module.weights[str(l)].copy_(
                full_weights[str(l)].to(dtype=module.weights[str(l)].dtype)
            )
        if getattr(module, "bias", None) is not None:
            for v in module.bias.values():
                v.zero_()


def _set_element_linear_identity(module) -> None:
    """Set an ``ElementConditionedLinearSO3`` to per-element identity (passthrough)."""
    if module is None:
        return
    reset = getattr(module, "reset_identity_parameters", None)
    if callable(reset):
        reset()
        return
    with torch.no_grad():
        for w in module.weights.values():
            eye = torch.eye(w.shape[-1], dtype=w.dtype, device=w.device)
            w.copy_(eye.unsqueeze(0).expand_as(w))
        if getattr(module, "bias", None) is not None:
            for v in module.bias.values():
                v.zero_()


def _mace_first_layer_sc_l0(skip_tp, *, mace_model, channels: int, num_elements: int) -> torch.Tensor:
    """MACE's first-layer self-connection ``sc[0]`` is a pure per-element constant l=0 vector:
    ``sc0_e = node_embed(onehot_e) @ skip_tp_l0[e]``. Returns ``(num_elements, C)``."""
    ref = next(mace_model.parameters())
    dtype = ref.dtype
    device = ref.device
    C = int(channels)
    mats = _skip_tp_l0_matrices(skip_tp, channels=C, num_elements=num_elements)  # (E,C,C)
    with torch.no_grad():
        onehot = torch.eye(num_elements, dtype=dtype, device=device)
        node_l0 = mace_model.node_embedding(onehot)  # (E, C) the l=0 element embedding
        out = torch.einsum("ec,ecd->ed", node_l0, mats.to(dtype=dtype, device=device))  # (E, C)
    return out


def _install_first_layer_sc0(ictd_model, *, layer_idx: int, sc0_by_element: torch.Tensor) -> None:
    """Inject MACE's first-layer element self-connection that the ICTC baseline's first interaction
    structurally omits (its first interaction multiplies the message via ``message_selector`` instead
    of adding an element ``skip`` to the product output).

    MACE computes ``h_1 = linear(symcontract(message)) + sc0(Z)`` where ``sc0(Z)`` is a PURE
    per-element constant in the l=0 channels. The ICTC baseline's ``product[0]`` is called with
    ``sc=None``, so this constant has no ordinary weight slot. Store it as a model buffer and let
    ``PureCartesianICTDFix.forward`` add it explicitly after product[0]. This avoids Python
    forward hooks, which are not reliable under ``torch.export`` / AOTI.
    """
    if int(layer_idx) != 0:
        raise NotImplementedError("only the first MACE interaction needs the exported sc0 buffer")
    if not hasattr(ictd_model, "install_mace_first_layer_sc0"):
        raise TypeError("target ICTC model does not expose install_mace_first_layer_sc0()")
    ictd_model.install_mace_first_layer_sc0(sc0_by_element)


def _install_energy_scale_shift(ictd_model, *, scale: float, shift: float) -> None:
    """Install MACE ScaleShiftMACE buffers into a converted ICTC model."""
    try:
        ref = next(ictd_model.parameters())
        device = ref.device
    except StopIteration:
        device = torch.device("cpu")
    with torch.no_grad():
        ictd_model.energy_output_scale_enabled = True
        ictd_model.energy_output_shift_enabled = True
        ictd_model.energy_output_scale = torch.tensor(
            float(scale), dtype=torch.float64, device=device
        )
        ictd_model.energy_output_shift = torch.tensor(
            float(shift), dtype=torch.float64, device=device
        )


def _copy_symmetric_contraction(mace_sc, ictd_sc) -> None:
    """Copy the per-contraction ``weights_max`` and ``weights.{k}`` 1:1. Both sides are the
    same MACE contraction parameterization. ``ictd_sc`` may be either the direct native-MACE
    contraction or the bridge-U wrapper whose inner ``.symmetric_contractions`` owns the weights."""
    i_sc = getattr(ictd_sc, "symmetric_contractions", ictd_sc)
    m_contr = mace_sc.contractions
    i_contr = i_sc.contractions
    assert len(m_contr) == len(i_contr), (len(m_contr), len(i_contr))
    with torch.no_grad():
        for mc, ic in zip(m_contr, i_contr):
            ic.weights_max.copy_(mc.weights_max.to(dtype=ic.weights_max.dtype))
            assert len(mc.weights) == len(ic.weights), (len(mc.weights), len(ic.weights))
            for mw, iw in zip(mc.weights, ic.weights):
                iw.copy_(mw.to(dtype=iw.dtype))
    if hasattr(ictd_sc, "refresh_cueq_weights"):
        ictd_sc.refresh_cueq_weights()
