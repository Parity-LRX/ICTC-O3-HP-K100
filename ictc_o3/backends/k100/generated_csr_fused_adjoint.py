# SPDX-License-Identifier: MIT
"""Source-owned FP32 feature/gate adjoints with one shared neighbor traversal.

Each warp owns one node/input-degree/channel slice. Gate gradients are written
in original edge order; feature gradients accumulate locally. Harmonic
gradients use warp reductions followed by adds across disjoint input degrees.
The existing mixed/higher derivative expressions remain the autograd rule.
"""
from ictc_o3.backends.k100.generated_cartesian_message import _literal
from ictc_o3.backends.k100.generated_cartesian_message import _sum


def _source(p, projection):
    c, hd = p['c'], projection['hdim']
    tiles = (c + 31) // 32
    canonical = p.get('generated_layout', 'packed') == 'canonical'
    def xi(xb, ni, i):
        return f'{xb}+channel*{ni}+{i}' if canonical else f'{xb+i*c}+channel'
    def vi(ob, m, k):
        return f'{ob}+channel*{m}+{k}' if canonical else f'{ob+k*c}+channel'
    def read(value):
        return f'channel<{c}?({value}):0.0f'
    columns = [[] for _ in range(p['bdim'])]
    for h, rows in enumerate(projection['static_rows']):
        for col, a in rows:
            columns[col].append((h, a))
    by_input = {}
    for row in p['path_specs']:
        by_input.setdefault((row[2], row[1]), []).append(row)
    grouping=p.get('fused_source_grouping','degree')
    if grouping not in ('degree','all'):raise ValueError('Fused source grouping must be degree or all')
    groups=[p['path_specs']] if grouping=='all' else list(by_input.values())
    lines = ['#include <torch/extension.h>', '#include <ATen/cuda/CUDAContext.h>',
        '#include <c10/cuda/CUDAGuard.h>', '#include <c10/cuda/CUDAException.h>',
        '#include <cuda_runtime.h>', 'template<bool Unit,bool NeedH>',
        '__global__ void fused_kernel(const float* X,const float* SH,const float* W,const float* G,const int64_t* D,const float* M,const float* V,const int64_t* PTR,const int64_t* ORDER,float* DX,float* DH,float* DG,int64_t N,int64_t E,int64_t offset,bool masked,bool needx,bool needg) {',
        'const int lane=threadIdx.x%32;',
        'const int64_t task=(static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;',
        f'if(task>=N*{tiles*len(groups)})return;',
        f'const int channel=(task%{tiles})*32+lane;',
        f'const int group=(task/{tiles})%{len(groups)};',
        f'const int64_t src=task/{tiles*len(groups)};']
    # Tail lanes must participate in harmonic shuffles; all their operands and
    # stores are masked rather than returning before a warp collective.
    for number, paths in enumerate(groups):
        lines.append(f'if(group=={number}) {{')
        inputs=tuple(dict.fromkeys((row[2],row[1]) for row in paths))
        for xb,ni in inputs:
            for i in range(ni):
                value=read(f"X[src*{p['xdim']}+{xi(xb,ni,i)}]")
                lines.append(f'const float x{xb}_{i}=(needg||NeedH)?({value}):0.0f;')
                lines.append(f'float dx{xb}_{i}=0.0f;')
        for path, *_ in paths:
            lines.append(f'const float w{path}='+read(f'Unit?1.0f:W[{path*c}+channel]')+';')
        lines += ['for(int64_t index=PTR[src];index<PTR[src+1];++index) {',
            'const int64_t original=ORDER[index];',
            'if(original<offset||original>=offset+E)continue;',
            'const int64_t e=original-offset,dst=D[e];const float mask=masked?M[e]:1.0f;']
        used = sorted({h for _, n, _, bb, bs, _, m in paths
            for i in range(n) for k in range(m) for h, _ in columns[bb+i*bs+k]})
        for h in used:
            lines.append(f'const float h{h}=SH[e*{hd}+{h}];')
            lines.append(f'float dh{h}=0.0f;')
        for path, ni, xb, bb, bs, ob, m in paths:
            lines += ['{', 'const float gate=(needx||NeedH)?('+read(f'G[e*{p["p"]*c}+{path*c}+channel]')+f')*w{path}*mask:0.0f;']
            for k in range(m):
                lines.append(f'const float v{k}='+read(f'V[dst*{p["odim"]}+{vi(ob,m,k)}]')+';')
            for i in range(ni):
                terms = []
                for k in range(m):
                    coefficients = columns[bb+i*bs+k]
                    if coefficients:
                        terms.append(f'v{k}*('+_sum(f'{_literal(a)}*h{h}' for h,a in coefficients)+')')
                lines += [f'const float z{i}='+_sum(terms)+';', f'if(needx)dx{xb}_{i}+=gate*z{i};']
            value = _sum(f'x{xb}_{i}*z{i}' for i in range(ni))
            lines.append(f'if(needg&&channel<{c})DG[e*{p["p"]*c}+{path*c}+channel]=({value})*w{path}*mask;')
            lines.append('if(NeedH) {')
            for i in range(ni):
                for k in range(m):
                    coefficients = columns[bb+i*bs+k]
                    if not coefficients:
                        continue
                    lines.append(f'const float db{i}_{k}=x{xb}_{i}*(v{k}*gate);')
                    for h,a in coefficients:
                        lines.append(f'dh{h}+=({_literal(a)})*db{i}_{k};')
            lines.extend(['}', '}'])
        lines.append('if(NeedH) {')
        for h in used:
            lines += [f'for(int shift=16;shift>0;shift/=2)dh{h}+=__shfl_down_sync(0xffffffff,dh{h},shift);',
                f'if(lane==0)atomicAdd(DH+e*{hd}+{h},dh{h});']
        lines.extend(['}', '}'])
        for xb,ni in inputs:
            for i in range(ni):
                lines.append(f'if(needx&&channel<{c})DX[src*{p["xdim"]}+{xi(xb,ni,i)}]+=dx{xb}_{i};')
        lines.append('}')
    lines.append('}')
    lines.append(f'''
void fused_adjoint(torch::Tensor x,torch::Tensor sh,torch::Tensor w,torch::Tensor g,torch::Tensor d,torch::Tensor mask,torch::Tensor v,torch::Tensor ptr,torch::Tensor order,torch::Tensor dx,torch::Tensor dh,torch::Tensor dg,int64_t offset,bool unit,bool needx,bool needh,bool needg) {{
    const c10::cuda::CUDAGuard guard(x.device());
    for(const auto& t:{{x,sh,w,g,mask,v,dx,dh,dg}})TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.scalar_type()==at::kFloat&&t.is_contiguous(),"Invalid fused CSR FP32 operand");
    for(const auto& t:{{d,ptr,order}})TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.scalar_type()==at::kLong&&t.is_contiguous(),"Invalid fused CSR topology");
    TORCH_CHECK(x.dim()==2&&x.size(1)=={p['xdim']}&&v.dim()==2&&v.size(0)==x.size(0)&&v.size(1)=={p['odim']},"Invalid fused CSR node shape");
    TORCH_CHECK(sh.dim()==2&&sh.size(0)==d.numel()&&sh.size(1)=={hd}&&g.dim()==3&&g.size(0)==d.numel()&&g.size(1)=={p['p']}&&g.size(2)=={c},"Invalid fused CSR edge shape");
    TORCH_CHECK(w.dim()==2&&w.size(0)=={p['p']}&&w.size(1)=={c}&&ptr.numel()==x.size(0)+1,"Invalid fused CSR bank/pointer");
    TORCH_CHECK((!mask.numel()||mask.numel()==d.numel())&&offset>=0&&offset+d.numel()<=order.numel(),"Invalid fused CSR slice");
    TORCH_CHECK(!needx||dx.sizes()==x.sizes(),"Invalid fused CSR feature destination");
    TORCH_CHECK(!needh||dh.sizes()==sh.sizes(),"Invalid fused CSR harmonic destination");
    TORCH_CHECK(!needg||dg.sizes()==g.sizes(),"Invalid fused CSR gate destination");
    if(!d.numel()||!x.size(0))return;
    const int64_t blocks=(x.size(0)*{tiles*len(groups)}+3)/4;
    TORCH_CHECK(blocks<(1LL<<31),"Fused CSR grid exceeds supported range");
    auto stream=at::cuda::getCurrentCUDAStream();
    #define CALL(U,H) fused_kernel<U,H><<<blocks,128,0,stream>>>(x.data_ptr<float>(),sh.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),d.data_ptr<int64_t>(),mask.data_ptr<float>(),v.data_ptr<float>(),ptr.data_ptr<int64_t>(),order.data_ptr<int64_t>(),dx.data_ptr<float>(),dh.data_ptr<float>(),dg.data_ptr<float>(),x.size(0),d.numel(),offset,mask.numel()!=0,needx,needg)
    if(unit){{if(needh){{CALL(true,true);}}else{{CALL(true,false);}}}}
    else{{if(needh){{CALL(false,true);}}else{{CALL(false,false);}}}}
    #undef CALL
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
''')
    return '\n'.join(lines)+'\n'


def _extension(p, projection):
    cache = projection.setdefault('generated_fused_source_extensions', {})
    key = (p['path_specs'],p['xdim'],p['odim'],p['c'],p.get('generated_layout','packed'),bool(p.get('compact_cartesian',False)))
    key=(*key,p.get('fused_source_grouping','degree'))
    if key not in cache:
        from ictc_o3.backends.k100.native_jit import load_inline
        name = f'mace_ictc_fused_source_{p["xdim"]}_{p["odim"]}_{p["p"]}'
        if p.get('generated_layout')=='canonical':name+='_canonical'
        if p.get('compact_cartesian',False):name+='_compact'
        if p.get('fused_source_grouping')=='all':name+='_all'
        declaration = '#include <torch/extension.h>\nvoid fused_adjoint('+','.join(['torch::Tensor']*12)+',int64_t,bool,bool,bool,bool);\n'
        cache[key] = load_inline(name=name,cpp_sources=declaration,cuda_sources=_source(p,projection),
            functions=['fused_adjoint'],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--ptxas-options=-v'],verbose=False)
    return cache[key]


def adjoint(x,sh,w,g,dst,mask,v,dx,dh,dg,p,projection,csr,start,needed):
    empty=x.new_empty(0)
    _extension(p,projection).fused_adjoint(x,sh.contiguous(),w,g,dst,mask,v,*csr,
        empty if dx is None else dx,empty if dh is None else dh,empty if dg is None else dg,
        start,bool(p.get('unit_weights',False)),*needed)
