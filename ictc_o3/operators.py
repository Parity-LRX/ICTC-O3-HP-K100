# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Documented Cartesian operators backed by the qualified CUDA/Triton kernels."""
from __future__ import annotations

import torch

__all__ = [
    "EdgeTensorProduct", "GraphMessage", "PackedGraphMessage", "direction_harmonics",
    "create_symmetric_contraction", "configure_symmetric_contraction",
    "channel_linear", "normalized_silu",
]


def _degree(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _eager(sample):
    from mace_ictc.models.fp32_policy import strict_fp32
    strict_fp32()
    from torch.fx.experimental.proxy_tensor import get_proxy_mode
    from torch._subclasses.fake_tensor import is_fake
    if (torch.compiler.is_compiling() or get_proxy_mode() is not None or is_fake(sample)
            or torch._C._are_functorch_transforms_active()
            or torch.is_autocast_enabled(sample.device.type)):
        raise ValueError("Standalone generated TP requires eager execution without autocast; use whole-model prepared export for AOTI")


def _tp_coverage(tp, input_lmax, edge_lmax):
    from mace_ictc.models.ictd_irreps import EdgeWeightedPathPreservingTensorProduct
    _degree(input_lmax, "input_lmax")
    _degree(edge_lmax, "edge_lmax")
    if not isinstance(tp, EdgeWeightedPathPreservingTensorProduct):
        raise TypeError("Expected an ICTC EdgeWeightedPathPreservingTensorProduct")
    if (tp.channels < 1 or tp.lmax < 0 or not tp.weight.is_cuda
            or tp.weight.dtype != torch.float32 or tp.internal_compute_dtype != torch.float32):
        raise ValueError("Generated TP requires CUDA FP32, internal FP32 and positive channels")
    if tp.weight.requires_grad:
        raise ValueError("TP base coefficients must already be fixed; HP does not freeze learned weights")
    if not tp.paths or any(l1 > input_lmax or l2 > edge_lmax for l1, l2, _ in tp.paths):
        raise ValueError("TP paths must be nonempty and fit the declared input/edge degrees")


def _operand(value, shape, sample, name):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if value.device != sample.device or value.dtype != torch.float32:
        raise ValueError(f"{name} must be FP32 on {sample.device}")


class _EdgeAdjoint(torch.autograd.Function):
    """Direct edge first/mixed adjoints, without graph-layout round trips."""
    @staticmethod
    def forward(ctx,x,sh,gates,weight,grad,owner,plan,projection,needed):
        from mace_ictc.models.generated_cartesian_message import _extension,_launch
        ctx.set_materialize_grads(False)
        ctx.owner,ctx.plan,ctx.projection,ctx.needed=owner,plan,projection,needed
        grad=grad.contiguous();ctx.save_for_backward(x,sh,gates,weight,grad)
        dx=torch.zeros_like(x) if needed[0] else x.new_empty(0)
        ids=owner._ids(x.shape[0])
        dh,dg=_extension(plan,projection).generated_adjoint(x,sh,weight,gates,ids,ids,
            owner._mask,grad,dx,bool(plan.get('unit_weights',False)),*needed,*_launch(plan,x,x.shape[0]),x.new_empty(0),x.new_empty(0))
        return dx if needed[0] else None,dh if needed[1] else None,dg if needed[2] else None

    @staticmethod
    def backward(ctx,hx,hs,hg):
        from mace_ictc.models.adjoint_requirements import required_gradients
        x,sh,gates,weight,grad=ctx.saved_tensors
        owner,plan,projection=ctx.owner,ctx.plan,ctx.projection
        needs=required_gradients((x,sh,gates,grad),(*ctx.needs_input_grad[:3],ctx.needs_input_grad[4]))
        higher=torch.is_grad_enabled()
        if not higher:
            from mace_ictc.models.generated_message_second import generated_second
            ids=owner._ids(x.shape[0])
            dx,ds,_,dg,dy=generated_second(x,sh,weight,gates,ids,ids,owner._mask,grad,
                plan,projection,(hx,hs,None,hg),(needs[0],needs[1],False,needs[2]),needs[3])
            return dx,ds,dg,None,dy,None,None,None,None
        # Higher derivatives retain the exact native
        # polynomial. No live weight or cotangent is detached in this route.
        from mace_ictc.models.rematerialized_message import _native
        from mace_ictc.models.fused_message import _normal_feature_gradient,_channel_cotangent
        with torch.enable_grad():
            local=tuple(v.clone() for v in (x,sh,gates));cotangent=grad.clone()
            ids=owner._ids(x.shape[0])
            output=_channel_cotangent(_native(_normal_feature_gradient(local[0],plan),local[1],
                weight,local[2],ids,ids,owner._mask,plan,projection),plan)
            first=torch.autograd.grad(output,[v for v,need in zip(local,ctx.needed) if need],
                cotangent,create_graph=True,allow_unused=True)
            pairs=[(v,h) for v,h in zip(first,[h for h,need in zip((hx,hs,hg),ctx.needed) if need])
                   if h is not None and v is not None and v.requires_grad]
            active=[v for v,need in zip((*local,cotangent),needs) if need]
            values=iter(torch.autograd.grad(tuple(v for v,_ in pairs),active,tuple(h for _,h in pairs),
                create_graph=higher,allow_unused=True))
            dx,dsh,dg,dy=[next(values) if need else None for need in needs]
        return dx,dsh,dg,None,dy,None,None,None,None


class _Edge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sh, gates, weight, owner):
        from mace_ictc.models.generated_cartesian_message import _extension, _launch
        plan, projection = owner._prepare()
        ctx.owner, ctx.plan, ctx.projection = owner, dict(plan), projection
        ctx.internal_layout = owner.execution == "legacy" and owner.layout == "mul_ir"
        ctx.save_for_backward(x, sh, gates, weight)
        if ctx.internal_layout:
            from mace_ictc.models.fused_message import _channel_node
            x = _channel_node(x, plan)
        ids = owner._ids(x.shape[0])
        output = x.new_empty(x.shape[0], plan["odim"])
        _extension(plan, projection).generated_forward(
            x, sh, weight, gates, ids, ids, owner._mask, output,
            bool(plan.get("unit_weights",False)), *_launch(plan, x, x.shape[0]),
        )
        if ctx.internal_layout:
            from mace_ictc.models.fused_message import _normal_message_output
            output = _normal_message_output(output, plan)
        return output

    @staticmethod
    def backward(ctx, grad):
        from mace_ictc.models.adjoint_requirements import required_gradients
        from mace_ictc.models.generated_cartesian_message import _extension, _launch
        x, sh, gates, weight = ctx.saved_tensors
        owner, plan, projection = ctx.owner, ctx.plan, ctx.projection
        ids = owner._ids(x.shape[0])
        needed = required_gradients((x, sh, gates), ctx.needs_input_grad[:3])
        grad = grad.contiguous()
        if ctx.internal_layout:
            from mace_ictc.models.fused_message import _channel_node, _channel_cotangent
            x, grad = _channel_node(x, plan), _channel_cotangent(grad, plan)
        if torch.is_grad_enabled():
            dx,dh,dg=_EdgeAdjoint.apply(x,sh,gates,weight,grad,owner,plan,projection,needed)
        else:
            # Restricted paths can leave some declared input blocks unused.
            dx = (torch.empty_like(x) if owner._all_inputs_used else torch.zeros_like(x)) if needed[0] else x.new_empty(0)
            dh, dg = _extension(plan, projection).generated_adjoint(
                x, sh, weight, gates, ids, ids, owner._mask, grad, dx,
                bool(plan.get("unit_weights",False)), *needed, *_launch(plan, x, x.shape[0]),x.new_empty(0),x.new_empty(0),
            )
            dx = dx if needed[0] else None
            dh = dh if needed[1] else None
            dg = dg if needed[2] else None
        if ctx.internal_layout and dx is not None:
            from mace_ictc.models.fused_message import _normal_feature_gradient
            dx = _normal_feature_gradient(dx, plan)
        return dx, dh, dg, None, None


class EdgeTensorProduct(torch.nn.Module):
    """Edge-wise TP with fixed base coefficients and live external gates.

    Wrap an existing ICTC TP without changing its coefficients/trainability.
    create constructs a new natural-parity TP with unit fixed coefficients.
    ir_mul uses [magnetic, channel] within each degree/path; mul_ir uses
    [channel, magnetic]. Output paths stay separate. Ordinary first and
    mixed/higher autograd are supported; eager execution is required.

    execution='direct' retains the current role-specialized adjoint. 'legacy'
    uses the earlier joint-adjoint schedule and rematerializes mul_ir layout
    conversions inside the TP autograd boundary. Both use current Cartesian
    coefficients, direct edge stores and the same public buffer contract.
    """
    def __init__(self, tp, *, input_lmax, edge_lmax, layout="ir_mul", execution="direct"):
        super().__init__()
        _tp_coverage(tp, input_lmax, edge_lmax)
        if layout not in ("ir_mul", "mul_ir"):
            raise ValueError("Edge layout must be 'ir_mul' or 'mul_ir'")
        if execution not in ("direct", "legacy"):
            raise ValueError("Edge execution must be 'direct' or 'legacy'")
        self.tp = tp
        self.input_lmax, self.edge_lmax, self.layout = input_lmax, edge_lmax, layout
        self.execution = execution
        self._plan, self._projection, self._prepared_device, self._weight_version = None, None, None, None
        self._all_inputs_used = {p[0] for p in tp.paths} == set(range(input_lmax + 1))
        self.register_buffer("_edge_ids", torch.empty(0, device=tp.weight.device, dtype=torch.int64), persistent=False)
        self.register_buffer("_mask", tp.weight.new_empty(0), persistent=False)

    @classmethod
    def create(cls, *, input_lmax, edge_lmax, output_lmax=None, channels=64, layout="ir_mul", execution="direct", device="cuda"):
        """Create a new FP32 natural-parity TP in native ICTC normalization.

        External gates follow paths. Wrap converted MACE modules directly to
        retain their calibrated coefficients; no MACE normalization is assumed
        by this fresh unit-coefficient factory.
        """
        _degree(input_lmax, "input_lmax")
        _positive_integer(channels, "channels")
        _degree(edge_lmax, "edge_lmax")
        output_lmax = input_lmax if output_lmax is None else output_lmax
        _degree(output_lmax, "output_lmax")
        if torch.device(device).type != "cuda":
            raise ValueError("Generated edge TP requires a CUDA device")
        from mace_ictc.models.ictd_irreps import EdgeWeightedPathPreservingTensorProduct
        paths = [(i, j, k) for i in range(input_lmax+1) for j in range(edge_lmax+1)
                 for k in range(output_lmax+1) if abs(i-j) <= k <= i+j and (i+j+k) % 2 == 0]
        tp = EdgeWeightedPathPreservingTensorProduct(
            channels=channels, lmax=max(input_lmax, edge_lmax, output_lmax),
            allowed_paths=paths, path_policy="full", internal_compute_dtype=torch.float32,
        ).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            tp.weight.fill_(1.)
        tp.weight.requires_grad_(False)
        return cls(tp, input_lmax=input_lmax, edge_lmax=edge_lmax, layout=layout, execution=execution)

    @property
    def paths(self):
        """Immutable external-gate order: (input_l, edge_l, output_l)."""
        return tuple(tuple(path) for path in self.tp.paths)

    @property
    def gate_width(self):
        return self.tp.num_paths * self.tp.channels

    @property
    def output_width(self):
        return sum(self.tp.channels*count*(2*l+1) for l, count in self.tp.path_counts_by_l.items())

    def _ids(self, rows):
        if self._edge_ids.numel() != rows:
            self._edge_ids = torch.arange(rows, device=self.tp.weight.device, dtype=torch.int64)
        return self._edge_ids

    def _prepare(self):
        _tp_coverage(self.tp, self.input_lmax, self.edge_lmax)
        device = str(self.tp.weight.device)
        if self._plan is None or self._prepared_device != device:
            from mace_ictc.models.fused_message import _plan
            from mace_ictc.models.bond_projection import projection_plan
            self._plan = dict(_plan(self.tp, list(range(self.input_lmax+1)), self.tp.weight.device, True))
            self._plan.update(generated_forward=True, generated_adjoint=True, generated_second=True,
                              generated_specialized_adjoint=self.execution=="direct" and max(self.input_lmax,self.edge_lmax,self.tp.lmax)<=3,
                              generated_second_roles=self.execution=="direct" and max(self.input_lmax,self.edge_lmax,self.tp.lmax)<=3,
                              generated_blocks=0, generated_schedule=0, edge_chunk_size=262144,
                              generated_layout="packed", edge_only=True)
            if (torch.version.hip and self.execution=="direct" and self.tp.channels>=64
                    and max(self.input_lmax,self.edge_lmax,self.tp.lmax)<=3
                    and getattr(torch.cuda.get_device_properties(self.tp.weight.device),'gcnArchName','').split(':')[0]=='gfx926'):
                # K100: one channel per lane avoids large unrolled adjoint live
                # ranges. Small channel counts need no cross-group reduction.
                self._plan.update(generated_lane_width=64,
                                  generated_channel_tile=64 if self.tp.channels>64 else 0,
                                  generated_split_adjoint=self.edge_lmax>=3 and len(self.tp.paths)>=16)
                from mace_ictc.models.k100_message_schedule import schedule
                self._plan=schedule(self._plan,self.tp.weight)
            self._projection = dict(projection_plan(
                self.tp, self.tp._get_bond_projectors(self.tp.weight.device, torch.float32),
                self.tp.weight.device, torch.float32,
            ))
            self._projection["generated_message_extensions"] = {}
            self._prepared_device = device
            self._weight_version = None
        version = None if self.tp.weight.is_inference() else self.tp.weight._version
        if version is None:
            self._plan["unit_weights"] = False
        elif self._weight_version != version:
            # Fixed-coefficient metadata is checked once per update, not each call.
            self._plan["unit_weights"] = bool(torch.all(self.tp.weight == 1).item())
            self._weight_version = version
        return self._plan, self._projection

    def __getstate__(self):
        state = super().__getstate__().copy()
        state.update(_plan=None, _projection=None, _prepared_device=None, _weight_version=None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "execution"):
            self.execution = "direct"

    def pack_input(self, blocks):
        """Pack {l: [E,C,2l+1]} using this operator's layout."""
        if set(blocks) != set(range(self.input_lmax+1)):
            raise ValueError("Input blocks must cover every declared degree")
        rows, parts = blocks[0].shape[0], []
        for l in range(self.input_lmax+1):
            block = blocks[l]
            _operand(block, (rows, self.tp.channels, 2*l+1), self.tp.weight, f"blocks[{l}]")
            if self.layout == "ir_mul":
                block = block.transpose(1, 2)
            parts.append(block.reshape(rows, self.tp.channels*(2*l+1)))
        return torch.cat(parts, -1).contiguous()

    def unpack_output(self, output):
        """Return {l: [E,C*path_count_l,2l+1]} in the ICTC convention."""
        rows = output.shape[0]
        _operand(output, (rows, self.output_width), self.tp.weight, "output")
        result, offset = {}, 0
        for l in range(self.tp.lmax+1):
            count, width = self.tp.path_counts_by_l.get(l, 0), 2*l+1
            value = output[:, offset:offset+count*self.tp.channels*width]
            if self.layout == "ir_mul":
                value = value.reshape(rows, count, width, self.tp.channels).transpose(2, 3)
            result[l] = value.reshape(rows, count*self.tp.channels, width)
            offset += count*self.tp.channels*width
        return result

    def forward(self, features, harmonics, gates):
        """features [E,C*(L_in+1)^2], harmonics [E,(L_edge+1)^2], gates [E,P*C] or [E,P,C]."""
        _tp_coverage(self.tp, self.input_lmax, self.edge_lmax)
        _eager(features)
        if features.ndim != 2:
            raise ValueError("features must be a flat two-dimensional row buffer")
        rows = features.shape[0]
        _operand(features, (rows, self.tp.channels*(self.input_lmax+1)**2), self.tp.weight, "features")
        _operand(harmonics, (rows, (self.edge_lmax+1)**2), self.tp.weight, "harmonics")
        _operand(gates, (rows, self.gate_width) if gates.ndim == 2 else (rows, self.tp.num_paths, self.tp.channels), self.tp.weight, "gates")
        if rows == 0:
            # Preserve the multilinear autograd dependency without empty -1 reshapes.
            empty = features[:, :1] * harmonics[:, :1] * gates.reshape(0, self.gate_width)[:, :1]
            return empty.expand(0, self.output_width)
        plan, projection = self._prepare()
        if self.layout == "mul_ir" and self.execution == "direct":
            from mace_ictc.models.fused_message import _channel_node
            features = _channel_node(features, plan)
        if projection["keys"] != list(range(self.edge_lmax+1)):
            harmonics = torch.cat([harmonics[:, l*l:(l+1)**2] for l in projection["keys"]], -1)
        if torch.is_grad_enabled() and any(v.requires_grad for v in (features,harmonics,gates)):
            # CUDA extension initialization belongs on the caller thread, before
            # the autograd worker requests a mixed derivative.
            from mace_ictc.models.generated_message_second import _extension as second_extension
            second_extension(plan,projection)
        output = _Edge.apply(features.contiguous(), harmonics.contiguous(),
                             gates.reshape(rows, self.tp.num_paths, self.tp.channels).contiguous(), self.tp.weight, self)
        if self.layout == "mul_ir" and self.execution == "direct":
            from mace_ictc.models.fused_message import _normal_message_output
            output = _normal_message_output(output, plan)
        return output


class PackedGraphMessage(torch.nn.Module):
    """Native packed gather/TP/scatter without dictionary layout round trips.

    Input/output layout matches EdgeTensorProduct(layout='ir_mul'). Topology
    stays live; no feature, gate or learned weight is cached. Coefficients must
    already be fixed, as for the other generated standalone TP APIs.
    """
    def __init__(self,tp,*,input_lmax,edge_lmax,cartesian_schedule='native',
                 reduction='atomic',edge_chunk_size=65536,topology_backend='native',
                 csr_accumulator_limit=None,csr_slice_search=False,geometry_schedule='auto'):
        super().__init__()
        if cartesian_schedule!='native':raise ValueError('Only the original ICTC basis is supported; the experimental STF-coordinate schedule is retired')
        if reduction not in ('atomic','csr'):raise ValueError('Packed graph reduction must be atomic or csr')
        if topology_backend not in ('native','device'):raise ValueError('Topology backend must be native or device')
        if csr_accumulator_limit is not None and (not isinstance(csr_accumulator_limit,int) or isinstance(csr_accumulator_limit,bool) or csr_accumulator_limit<1):
            raise ValueError('CSR accumulator limit must be a positive integer or None')
        if not isinstance(csr_slice_search,bool):raise ValueError('csr_slice_search must be a bool')
        if geometry_schedule not in ('auto','native','streamed'):raise ValueError('geometry_schedule must be auto, native or streamed')
        if not isinstance(edge_chunk_size,int) or isinstance(edge_chunk_size,bool) or edge_chunk_size<1:
            raise ValueError('edge_chunk_size must be a positive integer')
        self.edge=EdgeTensorProduct(tp,input_lmax=input_lmax,edge_lmax=edge_lmax)
        self.cartesian_schedule,self.reduction,self.edge_chunk_size=cartesian_schedule,reduction,edge_chunk_size
        self.options={'generated_topology':topology_backend,'csr_slice_search':csr_slice_search}
        if geometry_schedule!='auto':self.options['generated_geometry_streaming']=geometry_schedule=='streamed'
        if csr_accumulator_limit is not None:self.options['csr_accumulator_limit']=csr_accumulator_limit

    @classmethod
    def create(cls,*,channels=64,input_lmax=3,edge_lmax=3,output_lmax=None,device='cuda',**kwargs):
        edge=EdgeTensorProduct.create(channels=channels,input_lmax=input_lmax,edge_lmax=edge_lmax,
            output_lmax=output_lmax,device=device)
        return cls(edge.tp,input_lmax=input_lmax,edge_lmax=edge_lmax,**kwargs)

    @property
    def tp(self):return self.edge.tp
    @property
    def output_width(self):return self.edge.output_width
    @property
    def gate_width(self):return self.edge.gate_width

    def forward(self,features,harmonics,gates,src,dst,mask=None):
        if self.cartesian_schedule!='native':raise ValueError('Serialized STF-coordinate graph operators are retired; rebuild with cartesian_schedule=native')
        _eager(features);_tp_coverage(self.tp,self.edge.input_lmax,self.edge.edge_lmax)
        if features.ndim!=2:raise ValueError('Expected a flat two-dimensional packed node buffer')
        n,e=features.shape[0],src.numel()
        _operand(features,(n,self.tp.channels*(self.edge.input_lmax+1)**2),self.tp.weight,'features')
        _operand(harmonics,(e,(self.edge.edge_lmax+1)**2),self.tp.weight,'harmonics')
        _operand(gates,(e,self.gate_width) if gates.ndim==2 else (e,self.tp.num_paths,self.tp.channels),self.tp.weight,'gates')
        for name,value in (('src',src),('dst',dst)):
            if value.ndim!=1 or value.numel()!=e or value.device!=features.device or value.dtype!=torch.int64:
                raise ValueError(f'{name} must be an int64 CUDA vector of valid node indices')
        if mask is not None:
            if mask.requires_grad or tuple(mask.shape) not in ((e,),(e,1)):raise ValueError('Mask must be fixed [E] or [E,1]')
            _operand(mask,mask.shape,self.tp.weight,'mask')
        p,projection=self.edge._prepare()
        # Graph destinations may overlap; the direct-edge aliasing schedule is
        # invalid even when the graph reuses the same TP coefficients and plans.
        p={**p,'edge_only':False,'generated_restrict':False,
           'native_packed':True,'edge_chunk_size':self.edge_chunk_size,
           'generated_csr_forward':self.reduction=='csr',**self.options}
        sh=harmonics
        if projection['keys']!=list(range(self.edge.edge_lmax+1)):
            sh=torch.cat([sh[:,l*l:(l+1)**2] for l in projection['keys']],-1)
        g=gates.reshape(e,self.tp.num_paths,self.tp.channels).contiguous()
        m=features.new_empty(0) if mask is None else mask.reshape(e).contiguous()
        x,w=features.contiguous(),self.tp.weight
        from mace_ictc.models.rematerialized_message import rematerialized_message
        return rematerialized_message(x,sh.contiguous(),w,g,src.contiguous(),dst.contiguous(),m,p,projection)


class GraphMessage(torch.nn.Module):
    """Real gather/TP/gate/scatter with repeated destinations and live inputs.

    Input topology must contain valid int64 node indices. A mask is optional
    and fixed. The result is a dictionary of node path blocks. Radial MLP
    evaluation is separate; supply its live external gate output.
    """
    def __init__(self, tp, *, input_lmax, edge_lmax, layout="packed", edge_chunk_size=65536,
                 reduction="auto",cartesian_schedule="native"):
        super().__init__()
        _tp_coverage(tp, input_lmax, edge_lmax)
        if layout not in ("packed", "fused_pack",'canonical'):
            raise ValueError("Graph layout must be packed, fused_pack or canonical")
        if reduction not in ("auto","atomic","csr"):
            raise ValueError("Graph reduction must be auto, atomic or csr")
        if cartesian_schedule!='native':
            raise ValueError('Only the original ICTC basis is supported; the experimental STF-coordinate schedule is retired')
        qualified=max(input_lmax,edge_lmax,tp.lmax)<=3
        if reduction=="csr" and not qualified:
            raise ValueError("CSR graph reduction is qualified for L<=3")
        self.tp, self.input_lmax, self.edge_lmax = tp, input_lmax, edge_lmax
        self.layout, self.edge_chunk_size = layout, edge_chunk_size
        if not isinstance(edge_chunk_size, int) or isinstance(edge_chunk_size, bool) or edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be a positive integer")
        self._options = {"_generated_forward": True, "_generated_adjoint": True,
            "_generated_compact_cartesian":False,
            "_generated_second": True, "_generated_layout": layout,
            "_message_chunk_edges": edge_chunk_size, "_generated_blocks": 0,
            "_generated_schedule": 0, "_generated_reg_limit": 0,
            "_generated_csr_forward":True if reduction=="csr" else
                "auto" if reduction=="auto" and qualified and tp.channels>=96 else False}

    def forward(self, features, harmonics, gates, src, dst, mask=None):
        """features {l:[N,C,2l+1]}, harmonics {l:[E,1,2l+1]}, gates [E,P*C]."""
        _tp_coverage(self.tp, self.input_lmax, self.edge_lmax)
        if set(features) != set(range(self.input_lmax+1)) or set(harmonics) != set(range(self.edge_lmax+1)):
            raise ValueError("Degree dictionaries must cover declared input/edge degrees")
        _eager(features[0])
        n, e = features[0].shape[0], src.numel()
        for name, indices in (("src", src), ("dst", dst)):
            if indices.ndim != 1 or indices.numel() != e or indices.dtype != torch.int64 or indices.device != self.tp.weight.device:
                raise ValueError(f"{name} must be an int64 CUDA vector of E valid node indices")
        for l, value in features.items():
            _operand(value, (n, self.tp.channels, 2*l+1), self.tp.weight, f"features[{l}]")
        for l, value in harmonics.items():
            _operand(value, (e, 1, 2*l+1), self.tp.weight, f"harmonics[{l}]")
        _operand(gates, (e, self.tp.num_paths*self.tp.channels), self.tp.weight, "gates")
        if mask is not None:
            if mask.requires_grad or tuple(mask.shape) not in ((e,), (e, 1)):
                raise ValueError("mask must be fixed and have shape [E] or [E,1]")
            _operand(mask, mask.shape, self.tp.weight, "mask")
        from mace_ictc.models.fused_message import aggregate_message
        output = aggregate_message(self.tp, features, harmonics, gates, src, dst, mask,
                                   backend="triton_remat", generated_options=self._options)
        if output is None:
            raise ValueError("Requested generated graph operation is unsupported; use ICTC reference")
        return output


def direction_harmonics(directions, lmax, *, policy="auto"):
    """Component-normalized Cartesian harmonics: list of [E,2l+1] unit-direction blocks.

    Input is [E,3]; callers normalize their edge displacements. auto preserves
    reference fallbacks; generated requires the eager CUDA FP32 Triton route.
    """
    _degree(lmax, "lmax")
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape [E,3]")
    if policy not in ("auto", "generated", "reference"):
        raise ValueError("Harmonic policy must be auto, generated or reference")
    if policy != "reference":
        from mace_ictc.models.generated_direction_harmonics import generated_direction_harmonics
        output = generated_direction_harmonics(directions, lmax)
        if output is not None:
            return output
        if policy == "generated":
            raise ValueError("Generated harmonics require eager CUDA FP32 and no autocast")
    from mace_ictc.models.ictd_irreps import direction_harmonics_all
    return direction_harmonics_all(directions, lmax)


def create_symmetric_contraction(*, num_elements, lmax, target_lmax=None, channels=64,
                                correlation=3, use_reduced_cg=False, device="cuda", policy="generated",
                                high_l_policy="auto", training=False, reference_buffer_policy="auto",
                                species_schedule="grouped"):
    """Create a parameterized trainable bridge-U SC and select its plan.

    This creates initialized weights. To preserve existing trained coefficients,
    call configure_symmetric_contraction on the original module instead.
    """
    _degree(lmax, "lmax")
    _positive_integer(channels, "channels")
    _positive_integer(correlation, "correlation")
    if correlation not in (1,2,3):
        raise ValueError("HP correlation currently supports 1, 2, 3; cor4/5 experiments are separate")
    if not isinstance(use_reduced_cg, bool):
        raise TypeError("use_reduced_cg must be a boolean")
    if use_reduced_cg:
        raise ValueError("The independent HP factory uses native full CG; cuEq reduced-basis generation is not enabled")
    target_lmax = lmax if target_lmax is None else target_lmax
    _degree(target_lmax, "target_lmax")
    if not isinstance(num_elements, int) or isinstance(num_elements, bool) or num_elements < 1:
        raise ValueError("num_elements must be a positive integer")
    from mace_ictc.models.fp32_policy import strict_fp32
    strict_fp32()
    from mace_ictc.models.pure_cartesian_ictd_fix import ICTDBridgeUSymmetricContractionSO3
    module = ICTDBridgeUSymmetricContractionSO3(
        num_elements=num_elements, channels=channels, lmax=lmax, target_lmax=target_lmax,
        correlation=correlation, use_reduced_cg=use_reduced_cg,
    ).to(device=device, dtype=torch.float32)
    configure_symmetric_contraction(module, policy=policy, high_l_policy=high_l_policy,
                                   training=training,reference_buffer_policy=reference_buffer_policy,
                                   species_schedule=species_schedule)
    return module


def configure_symmetric_contraction(module, *, policy="auto", high_l_policy="auto", training=False,
                                    reference_buffer_policy="auto",species_schedule="grouped"):
    """Select factorized SC in place, preserving live weights and checkpoint keys.

    Coverage: CUDA FP32, positive channels, correlation 1/2/3, native full CG. Supply
    discrete node_type_idx in ordinary forward calls to use factorization;
    calls with only node_attrs retain the established route.
    High-L cubic auto residency keeps immutable dense U on CPU after planning;
    reference_buffer_policy=device keeps them on the parameter device.
    """
    from mace_ictc.models.fp32_policy import strict_fp32
    strict_fp32()
    from mace_ictc.models.pure_cartesian_ictd_fix import ICTDBridgeUSymmetricContractionSO3
    if not isinstance(module, ICTDBridgeUSymmetricContractionSO3):
        raise TypeError("Expected ICTDBridgeUSymmetricContractionSO3")
    if policy not in ("auto", "generated", "reference"):
        raise ValueError("SC policy must be auto, generated or reference")
    if high_l_policy not in ("auto","off"):
        raise ValueError("high_l_policy must be auto or off")
    if species_schedule not in ('grouped','indexed'):
        raise ValueError('species_schedule must be grouped or indexed')
    if not isinstance(training,bool):raise TypeError("training must be a boolean")
    if reference_buffer_policy not in ('auto','device'):
        raise ValueError('reference_buffer_policy must be auto or device')
    from mace_ictc.models.reference_u_residency import host_reference_unavailable,restore_reference_u
    parameter = next(module.parameters())
    supported = (parameter.is_cuda and parameter.dtype == torch.float32 and module.channels > 0 and not module.use_reduced_cg
                 and module.lmax >= 0 and module.target_lmax >= 0
                 and all(c.correlation in (1,2,3) for c in module.symmetric_contractions.contractions))
    if policy == "generated" and not supported:
        raise ValueError("Generated SC requires CUDA FP32 and a positive-order bridge-U contraction")
    if policy == "reference" or supported or policy=='auto':
        enabled = supported and policy != "reference"
        module._require_factorized_plan=policy=="generated"
        module._compact_contraction = enabled
        module._factorized_contraction = enabled
        module._factorized_circuit = enabled
        module._species_circuit=(enabled and species_schedule=='indexed' and max(module.lmax,module.target_lmax)<=3)
        module._partitioned_circuit=False
        module._compact_circuit_adjoints=False
        module._staged_circuit=False
        module._bounded_pair_circuit=False
        module._adaptive_sparse_tiling=False
        orders={c.correlation for c in module.symmetric_contractions.contractions}
        high=enabled and high_l_policy=="auto" and 4<=module.lmax<=5 and module.target_lmax<=5
        module._host_reference_u=(high and orders=={3} and reference_buffer_policy=='auto'
                                 and host_reference_unavailable(module) is None)
        if not module._host_reference_u:restore_reference_u(module)
        if high and orders=={3}:
            module._factorized_circuit=False
            module._adaptive_sparse_tiling=True
        if high and orders=={2} and module.target_lmax>0 and (training or (module.lmax==5 and module.channels%32)):
            module._partitioned_circuit=True
    return {"selected": "generated" if supported and policy != "reference" else "reference",
            "high_l_policy":high_l_policy,
            "species_schedule":species_schedule,"indexed_species_active":module._species_circuit,
            "reference_buffer_policy":reference_buffer_policy,
            "sc_schedule":"sparse" if supported and policy!="reference" and not module._factorized_circuit else
                "partitioned" if supported and policy!="reference" and module._partitioned_circuit else
                "circuit" if supported and policy!="reference" else "reference",
            "reasons": [] if supported or policy == "reference" else ["outside qualified standalone SC coverage"]}


def channel_linear(features, weight, bias=None):
    """Shared-weight channel GEMM: [*,Cin,m] x [Cout,Cin] -> [*,Cout,m].

    Matrices stay live and ordinary autograd includes their higher gradients.
    Non-scalar bias would break equivariance and is rejected.
    """
    from mace_ictc.models.fp32_policy import strict_fp32
    strict_fp32()
    if features.ndim < 2 or weight.ndim != 2 or features.shape[-2] != weight.shape[1]:
        raise ValueError("Expected features [*,Cin,m] and weight [Cout,Cin]")
    if features.device != weight.device or features.dtype != weight.dtype:
        raise ValueError("Features and weights must share device and dtype")
    if bias is not None:
        if features.shape[-1] != 1:
            raise ValueError("Channel bias is allowed only for the scalar angular block")
        if bias.shape != (weight.shape[0],) or bias.device != weight.device or bias.dtype != weight.dtype:
            raise ValueError("bias must match output channel count, device and dtype")
    angular = features.movedim(-2, -1).contiguous()
    output = torch.nn.functional.linear(angular.reshape(-1, weight.shape[1]), weight, bias)
    return output.reshape(*angular.shape[:-1], weight.shape[0]).movedim(-1, -2)


def normalized_silu(features, scale=1.0):
    """Fused FP32 CUDA SiLU times a fixed scale, with differentiable fallbacks."""
    if not isinstance(scale, (int, float)) or isinstance(scale, bool):
        raise TypeError("scale must be a fixed numeric scalar")
    from mace_ictc.models.fused_activation import normalized_silu as implementation
    return implementation(features, float(scale))
