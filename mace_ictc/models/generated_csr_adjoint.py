# SPDX-License-Identifier: MIT
"""Source-owned exact Cartesian feature adjoint with no atomic feature writes."""
from .generated_cartesian_message import _literal,_sum


def _source(p,projection):
    c=p['c'];hd=projection['hdim'];tiles=(c+31)//32
    canonical=p.get('generated_layout','packed')=='canonical'
    neighbor_warps=int(p.get('csr_neighbor_warps',1))
    if neighbor_warps not in (1,4):raise ValueError('CSR neighbor warps must be 1 or 4')
    def read(expression):return f'channel<{c}?({expression}):0.0f' if neighbor_warps>1 and c%32 else expression
    def xi(xb,ni,i):return f'{xb}+channel*{ni}+{i}' if canonical else f'{xb+i*c}+channel'
    def vi(ob,m,k):return f'{ob}+channel*{m}+{k}' if canonical else f'{ob+k*c}+channel'
    columns=[[] for _ in range(p['bdim'])]
    for h,rows in enumerate(projection['static_rows']):
        for column,a in rows:columns[column].append((h,a))
    groups={}
    for row in p['path_specs']:groups.setdefault((row[2],row[1]),[]).append(row)
    lines=['#include <torch/extension.h>','#include <ATen/cuda/CUDAContext.h>',
        '#include <c10/cuda/CUDAGuard.h>','#include <c10/cuda/CUDAException.h>','#include <cuda_runtime.h>',
        'template<bool Unit>',
        '__global__ void source_kernel(const float* SH,const float* W,const float* G,const int64_t* D,const float* M,const float* V,const int64_t* PTR,const int64_t* ORDER,float* DX,int64_t N,int64_t E,int64_t offset,bool masked) {',
        'const int lane=threadIdx.x%32;',
        ('const int64_t task=blockIdx.x;const int shard=threadIdx.x/32;' if neighbor_warps>1 else
         'const int64_t task=(static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;'),
        f'if(task>=N*{tiles*len(groups)})return;',
        f'const int channel=(task%{tiles})*32+lane;',
        f'const int group=(task/{tiles})%{len(groups)};',
        f'const int64_t src=task/{tiles*len(groups)};',
        '' if neighbor_warps>1 else f'if(channel>={c})return;']
    if neighbor_warps>1:lines.append(f'__shared__ float partial[{neighbor_warps*32*max(ni for _,ni in groups)}];')
    for number,((xb,ni),paths) in enumerate(groups.items()):
        lines.append(f'if(group=={number}) {{')
        for i in range(ni):lines.append(f'float dx{i}=0.0f;')
        for path,*_ in paths:lines.append(f'const float w{path}='+read(f'Unit?1.0f:W[{path*c}+channel]')+';')
        lines += [f'for(int64_t index=PTR[src]'+('+shard' if neighbor_warps>1 else '')+f';index<PTR[src+1];index+={neighbor_warps}) {{',
            'const int64_t original=ORDER[index];',
            'if(original<offset||original>=offset+E)continue;',
            'const int64_t e=original-offset,dst=D[e];const float mask=masked?M[e]:1.0f;']
        for h in range(hd):lines.append(f'const float h{h}=SH[e*{hd}+{h}];')
        for path,ni,xb,bb,bs,ob,m in paths:
            lines.append('{')
            lines.append(f'const float g=('+read(f'G[e*{p["p"]*c}+{path*c}+channel]')+f')*w{path}*mask;')
            for mag in range(m):lines.append(f'const float v{mag}='+read(f'V[dst*{p["odim"]}+{vi(ob,m,mag)}]')+';')
            for i in range(ni):
                terms=[]
                for mag in range(m):
                    coefficients=columns[bb+i*bs+mag]
                    if coefficients:terms.append(f'v{mag}*('+_sum(f'{_literal(a)}*h{h}' for h,a in coefficients)+')')
                lines.append(f'dx{i}+=g*('+_sum(terms)+');')
            lines.append('}')
        lines.append('}')
        if neighbor_warps>1:
            for i in range(ni):lines.append(f'partial[(shard*{ni}+{i})*32+lane]=dx{i};')
            lines.append('__syncthreads();')
        for i in range(ni):
            value=_sum(f'partial[({shard*ni+i})*32+lane]' for shard in range(neighbor_warps)) if neighbor_warps>1 else f'dx{i}'
            lines.append((f'if(shard==0&&channel<{c}) ' if neighbor_warps>1 else '')+f'DX[src*{p["xdim"]}+{xi(xb,ni,i)}]+={value};')
        lines.append('}')
    lines.append('}')
    lines.append(f'''
void source_adjoint(torch::Tensor sh,torch::Tensor w,torch::Tensor g,torch::Tensor d,torch::Tensor mask,torch::Tensor v,torch::Tensor ptr,torch::Tensor order,torch::Tensor dx,int64_t offset,bool unit) {{
    const c10::cuda::CUDAGuard guard(dx.device());
    for(const auto& t:{{sh,w,g,mask,v,dx}}) TORCH_CHECK(t.is_cuda()&&t.device()==dx.device()&&t.scalar_type()==at::kFloat&&t.is_contiguous(),"Invalid source adjoint FP32 operand");
    for(const auto& t:{{d,ptr,order}}) TORCH_CHECK(t.is_cuda()&&t.device()==dx.device()&&t.scalar_type()==at::kLong&&t.is_contiguous(),"Invalid source adjoint topology");
    TORCH_CHECK(dx.dim()==2&&dx.size(1)=={p['xdim']}&&v.dim()==2&&v.size(0)==dx.size(0)&&v.size(1)=={p['odim']},"Invalid source adjoint node shape");
    TORCH_CHECK(sh.dim()==2&&sh.size(0)==d.numel()&&sh.size(1)=={hd}&&g.dim()==3&&g.size(0)==d.numel()&&g.size(1)=={p['p']}&&g.size(2)=={c},"Invalid source adjoint edge shape");
    TORCH_CHECK(w.dim()==2&&w.size(0)=={p['p']}&&w.size(1)=={c}&&ptr.numel()==dx.size(0)+1,"Invalid source adjoint pointer/bank");
    TORCH_CHECK((!mask.numel()||mask.numel()==d.numel())&&offset>=0&&offset+d.numel()<=order.numel(),"Invalid source adjoint slice");
    if(!d.numel()||!dx.size(0))return;
    const int64_t blocks={'dx.size(0)*'+str(tiles*len(groups)) if neighbor_warps>1 else '(dx.size(0)*'+str(tiles*len(groups))+'+3)/4'};
    TORCH_CHECK(blocks<(1LL<<31),"Source adjoint grid exceeds supported range");
    auto stream=at::cuda::getCurrentCUDAStream();
    #define CALL(U) source_kernel<U><<<blocks,128,0,stream>>>(sh.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),d.data_ptr<int64_t>(),mask.data_ptr<float>(),v.data_ptr<float>(),ptr.data_ptr<int64_t>(),order.data_ptr<int64_t>(),dx.data_ptr<float>(),dx.size(0),d.numel(),offset,mask.numel()!=0)
    if(unit){{CALL(true);}}else{{CALL(false);}}
    #undef CALL
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
''')
    return '\n'.join(lines)+'\n'


def _extension(p,projection):
    cache=projection.setdefault('generated_source_extensions',{})
    key=(p['path_specs'],p['xdim'],p['odim'],p['c'],p.get('generated_layout','packed'),bool(p.get('compact_cartesian',False)))
    neighbor_warps=int(p.get('csr_neighbor_warps',1));key=(*key,neighbor_warps)
    if key not in cache:
        from .native_jit import load_inline
        name=f'mace_ictc_source_adjoint_{p["xdim"]}_{p["odim"]}_{p["p"]}'
        if p.get('generated_layout')=='canonical':name+='_canonical'
        if p.get('compact_cartesian',False):name+='_compact'
        if neighbor_warps>1:name+=f'_nw{neighbor_warps}'
        declarations='#include <torch/extension.h>\nvoid source_adjoint(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,bool);\n'
        cache[key]=load_inline(name=name,cpp_sources=declarations,cuda_sources=_source(p,projection),functions=['source_adjoint'],
            extra_cflags=['-O3'],extra_cuda_cflags=['-O3'],verbose=False)
    return cache[key]


def source_adjoint(sh,w,g,dst,mask,v,dx,p,projection,csr,start):
    _extension(p,projection).source_adjoint(sh.contiguous(),w,g,dst,mask,v,*csr,dx,start,bool(p.get('unit_weights',False)))
