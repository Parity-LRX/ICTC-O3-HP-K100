"""A linear geometry projection with a fused reciprocal autograd map.

The transpose reduces directly to harmonic components. Its double backward
returns one complete projected cotangent, avoiding per-slice full-bank fills
and additions from the native projection graph.
"""
import torch
from .monomial_operator import triton
if triton is not None:
    import triton.language as tl

    @triton.jit
    def _project_forward(X,J,A,O,E,BDIM:tl.constexpr,HDIM:tl.constexpr,K:tl.constexpr,BLOCK:tl.constexpr):
        index=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
        valid=index<E*BDIM;e=index//BDIM;column=index%BDIM
        value=tl.full((BLOCK,),0,tl.float32)
        for k in tl.static_range(K):
            h=tl.load(J+column*K+k,valid,0)
            coefficient=tl.load(A+column*K+k,valid,0.)
            x=tl.load(X+e*HDIM+h,valid,0.)
            value=value+x*coefficient
        tl.store(O+index,value,valid)

    @triton.jit
    def _project_column_transpose(V,J,A,O,E,HDIM:tl.constexpr,
                           K:tl.constexpr,BO:tl.constexpr,BK:tl.constexpr):
        e=tl.program_id(0).to(tl.int64)*BO+tl.arange(0,BO);h=tl.program_id(1)
        k=tl.arange(0,BK);active=(e[None,:]<E)&(k[:,None]<K)
        j=tl.load(J+h*K+k,k<K,0);a=tl.load(A+h*K+k,k<K,0.)
        v=tl.load(V+j[:,None]*E+e[None,:],active,0.)
        tl.store(O+e*HDIM+h,tl.sum(v*a[:,None],0),e<E)

    @triton.jit
    def _project_transpose(V,J,A,O,E,BDIM:tl.constexpr,HDIM:tl.constexpr,
                           K:tl.constexpr,BO:tl.constexpr,BK:tl.constexpr):
        row=tl.program_id(0).to(tl.int64)*BO+tl.arange(0,BO)
        e=row//HDIM;h=row%HDIM;valid=row<E*HDIM
        k=tl.arange(0,BK);active=valid[:,None]&(k[None,:]<K)
        j=tl.load(J+h[:,None]*K+k[None,:],active,0)
        a=tl.load(A+h[:,None]*K+k[None,:],active,0.)
        v=tl.load(V+e[:,None]*BDIM+j,active,0.)
        tl.store(O+row,tl.sum(v*a,1),valid)


def projection_plan(tp,projectors,device,dtype):
    keys=sorted({int(group["l2"]) for group in tp._groups})
    versions=tuple(None if value.is_inference() else value._version for value in projectors)
    key=(tuple(keys),tuple(id(value) for value in projectors),versions,device,dtype)
    previous=getattr(tp,"_bond_projection_adjoint_plan",None)
    if (None not in versions and previous is not None and previous[0]==key
            and "forward_indices" in previous[1]):return previous[1]
    offsets={};hdim=0
    for l in keys:offsets[l]=hdim;hdim+=2*l+1
    rows=[[] for _ in range(hdim)];forward_columns=[];groups=[];bdim=0
    for group,projector in zip(tp._groups,projectors):
        l=int(group["l2"]);h=2*l+1;size=projector.shape[1]
        groups.append((offsets[l],h,projector))
        # Match the existing native sparse TP's numerical-zero convention.
        # Float64 CG construction leaves cancellation residues (often 1e-17)
        # at structural zeros. Exact !=0 tests otherwise expand those residues
        # into runtime FP32 operations in every generated first/mixed kernel.
        # Keep the original projector/basis and its dense reference untouched.
        from .ictd_irreps import _SPARSE_ZERO_THRESHOLD
        constant=projector.detach().cpu()
        constant=torch.where(constant.abs()>_SPARSE_ZERO_THRESHOLD,constant,torch.zeros_like(constant))
        for column in range(size):
            positions=torch.nonzero(constant[:,column]!=0,as_tuple=True)[0].tolist()
            forward_columns.append([(offsets[l]+position,float(constant[position,column])) for position in positions])
        for magnetic in range(h):
            columns=torch.nonzero(constant[magnetic]!=0,as_tuple=True)[0].tolist()
            rows[offsets[l]+magnetic].extend((bdim+column,float(constant[magnetic,column])) for column in columns)
        bdim+=size
    k=max(map(len,rows),default=0)
    indices=torch.zeros(hdim,k,device=device,dtype=torch.int32)
    coefficient=torch.zeros(hdim,k,device=device,dtype=dtype)
    for h,values in enumerate(rows):
        if values:
            indices[h,:len(values)]=torch.tensor([j for j,_ in values],device=device,dtype=torch.int32)
            coefficient[h,:len(values)]=torch.tensor([v for _,v in values],device=device,dtype=dtype)
    fk=max(map(len,forward_columns),default=0)
    forward_indices=torch.zeros(bdim,fk,device=device,dtype=torch.int32)
    forward_coefficient=torch.zeros(bdim,fk,device=device,dtype=dtype)
    for column,values in enumerate(forward_columns):
        if values:
            forward_indices[column,:len(values)]=torch.tensor([h for h,_ in values],device=device,dtype=torch.int32)
            forward_coefficient[column,:len(values)]=torch.tensor([v for _,v in values],device=device,dtype=dtype)
    result={"keys":keys,"groups":groups,"bdim":bdim,"hdim":hdim,"indices":indices,"coefficient":coefficient,"k":k,
            "static_rows":tuple(tuple(values) for values in rows),
            "forward_indices":forward_indices,"forward_coefficient":forward_coefficient,"forward_k":fk}
    tp._bond_projection_adjoint_plan=(key,result)
    return result


def _forward_raw(sh,plan):
    sh=sh.contiguous();out=sh.new_empty(sh.shape[0],plan["bdim"])
    if out.numel():
        _project_forward[(triton.cdiv(out.numel(),256),)](sh,plan["forward_indices"],plan["forward_coefficient"],out,
            sh.shape[0],plan["bdim"],plan["hdim"],plan["forward_k"],256,num_warps=4,enable_fp_fusion=True)
    return out


def _transpose_raw(grad,plan):
    if grad.stride(0)==1 and grad.stride(1)==grad.shape[0] and grad.shape[0]>1:
        out=grad.new_empty(grad.shape[0],plan["hdim"])
        if out.numel():
            _project_column_transpose[(triton.cdiv(grad.shape[0],32),plan["hdim"])](grad,plan["indices"],plan["coefficient"],out,
                grad.shape[0],plan["hdim"],plan["k"],32,triton.next_power_of_2(max(plan["k"],1)),
                num_warps=4,enable_fp_fusion=True)
        return out
    grad=grad.contiguous();out=grad.new_empty(grad.shape[0],plan["hdim"])
    if out.numel():
        _project_transpose[(triton.cdiv(out.numel(),8),)](grad,plan["indices"],plan["coefficient"],out,
            grad.shape[0],plan["bdim"],plan["hdim"],plan["k"],8,triton.next_power_of_2(max(plan["k"],1)),
            num_warps=4,enable_fp_fusion=True)
    return out


class _ProjectionTranspose(torch.autograd.Function):
    @staticmethod
    def forward(ctx,grad,plan):
        ctx.plan=plan
        return _transpose_raw(grad,plan)

    @staticmethod
    def backward(ctx,h):
        return _Projection.apply(h,ctx.plan) if torch.is_grad_enabled() else _forward_raw(h,ctx.plan),None


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx,sh,plan):
        ctx.plan=plan
        return _forward_raw(sh,plan)

    @staticmethod
    def backward(ctx,grad):
        return _ProjectionTranspose.apply(grad,ctx.plan) if torch.is_grad_enabled() else _transpose_raw(grad,ctx.plan),None


def projected_geometry(tp,edge_attrs,projectors,sample):
    if any(value.requires_grad for value in projectors):return None
    plan=projection_plan(tp,projectors,sample.device,sample.dtype)
    sh=torch.cat([edge_attrs[l].squeeze(-2) for l in plan["keys"]],dim=-1).contiguous()
    return _Projection.apply(sh,plan)
