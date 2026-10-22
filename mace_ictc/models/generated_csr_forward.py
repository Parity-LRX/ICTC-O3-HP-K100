# SPDX-License-Identifier: MIT
"""Exact FP32 destination-owned TP reduction, with live unsorted edge inputs.

Each warp owns a destination/channel/path slice and sums its neighbors before
one output update. CSR contains only topology, constructed on device per call;
radial/geometry arrays retain their original order and no gate permutation is
materialized. Existing analytic first/mixed adjoints remain the derivative.
Small edge slices schedule candidate edges instead of every output node. The
first matching CSR edge owns each destination, including unsorted inputs and
empty-node gaps, while preserving the original neighbor accumulation order.
"""
import torch
from .generated_cartesian_message import _literal,_sum


def enabled(mode,edges,nodes):
    if mode=="auto":return nodes>0 and edges>=4096 and edges>=16*nodes
    return bool(mode)


def topology(dst,n,kind='native',with_order_hint=False):
    if kind=='device':
        from .message_topology import topology as device_topology
        return device_topology(dst,n,with_order_hint)
    if kind!='native':raise ValueError('CSR topology must be native or device')
    counts=torch.bincount(dst,minlength=n)
    result=(torch.cat((counts.new_zeros(1),counts.cumsum(0))),torch.argsort(dst,stable=True))
    # Stable sorting makes original edge IDs monotone inside each node even
    # for unsorted input. Empty hint means this is guaranteed by construction.
    return (*result,dst.new_empty(0)) if with_order_hint else result


def _source(p,projection):
    c=p['c'];hd=projection['hdim'];tiles=(c+31)//32
    limit=int(p.get('csr_accumulator_limit',48))
    canonical=p.get('generated_layout','packed')=='canonical'
    neighbor_warps=int(p.get('csr_neighbor_warps',1))
    slice_search=bool(p.get('csr_slice_search',False))
    if neighbor_warps not in (1,4):raise ValueError('CSR neighbor warps must be 1 or 4')
    def read(expression):return f'channel<{c}?({expression}):0.0f' if neighbor_warps>1 and c%32 else expression
    def xi(xb,ni,i):
        return f'{xb}+channel*{ni}+{i}' if canonical else f'{xb+i*c}+channel'
    def oi(ob,m,mag):
        if p.get('inference_degree_layout',False):
            degree=(m-1)//2;start=p['ooffset'][degree];width=p['counts'][degree]*c
            return f'{start}*N+(dst*{m}+{mag})*{width}+{(ob-start)//m}+channel'
        local=f'{ob}+channel*{m}+{mag}' if canonical else f'{ob+mag*c}+channel'
        return f'dst*{p["odim"]}+{local}'
    columns=[[] for _ in range(p['bdim'])]
    for h,rows in enumerate(projection['static_rows']):
        for col,a in rows:columns[col].append((h,a))
    specs=p['path_specs']
    if sum(row[-1] for row in specs)<=limit:groups=[specs]
    else:
        grouped={}
        for row in specs:grouped.setdefault(row[-1],[]).append(row)
        groups=[]
        for rows in grouped.values():
            current=[];live=0
            for row in rows:
                if current and live+row[-1]>limit:
                    groups.append(current);current=[];live=0
                current.append(row);live+=row[-1]
            if current:groups.append(current)
    lines=['#include <torch/extension.h>','#include <ATen/cuda/CUDAContext.h>',
        '#include <c10/cuda/CUDAGuard.h>','#include <c10/cuda/CUDAException.h>','#include <cuda_runtime.h>',
        '__device__ __forceinline__ int64_t edge_lower_bound(const int64_t* order,int64_t begin,int64_t end,int64_t value) { while(begin<end){const int64_t mid=begin+(end-begin)/2;if(order[mid]<value)begin=mid+1;else end=mid;}return begin;}',
        'template<bool Unit,bool EdgeOwned>',
        '__global__ void csr_kernel(const float* X,const float* SH,const float* W,const float* G,const int64_t* S,const float* MASK,const int64_t* PTR,const int64_t* ORDER,const int64_t* ORDER_HINT,float* O,const int64_t* D,int64_t N,int64_t E,int64_t offset,int64_t total_edges,bool masked) {',
        'const int lane=threadIdx.x%32;',
        ('const int64_t task=blockIdx.x;const int shard=threadIdx.x/32;' if neighbor_warps>1 else
         'const int64_t task=(static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;'),
        f'if(task>=(EdgeOwned?E:N)*{tiles*len(groups)})return;',
        f'const int channel=(task%{tiles})*32+lane;',
        f'const int group=(task/{tiles})%{len(groups)};',
        f'const int64_t candidate=task/{tiles*len(groups)};',
        'const int64_t dst=EdgeOwned?D[candidate]:candidate;',
        '' if neighbor_warps>1 else f'if(channel>={c})return;',
        'int64_t begin=PTR[dst],end=PTR[dst+1];']
    if slice_search:
        lines.extend(['if((offset||E<total_edges)&&(!ORDER_HINT||ORDER_HINT[0])) {',
            'const int64_t stop=edge_lower_bound(ORDER,begin,end,offset+E);',
            'begin=edge_lower_bound(ORDER,begin,stop,offset);end=stop;',
            '}', 'if(begin>=end)return;'])
    lines.extend(['if(EdgeOwned) {',
        'int64_t first=begin;',
        'while(first<end&&(ORDER[first]<offset||ORDER[first]>=offset+E))++first;',
        'if(first==end||ORDER[first]!=offset+candidate)return;',
        '}'])
    shared=max(sum(row[-1] for row in paths) for paths in groups)
    if neighbor_warps>1:lines.append(f'__shared__ float partial[{neighbor_warps*32*shared}];')
    for group,paths in enumerate(groups):
        lines.append(f'if(group=={group}) {{')
        for path,ni,xb,bb,bs,ob,m in paths:
            lines.append(f'const float w{path}='+read(f'Unit?1.0f:W[{path*c}+channel]')+';')
            for mag in range(m):lines.append(f'float y{path}_{mag}=0.0f;')
        lines.extend(['bool found=false;',f'for(int64_t index=begin'+('+shard' if neighbor_warps>1 else '')+f';index<end;index+={neighbor_warps}) {{',
            'const int64_t original=ORDER[index];',
            'if(original<offset||original>=offset+E)continue;',
            'found=true;const int64_t e=original-offset,src=S[e];',
            'const float mask=masked?MASK[e]:1.0f;'])
        for h in range(hd):lines.append(f'const float h{h}=SH[e*{hd}+{h}];')
        inputs={}
        for row in paths:inputs.setdefault((row[2],row[1]),[]).append(row)
        for (xb,ni),input_paths in inputs.items():
            lines.append('{')
            for i in range(ni):lines.append(f'const float x{i}='+read(f'X[src*{p["xdim"]}+{xi(xb,ni,i)}]')+';')
            for path,ni,xb,bb,bs,ob,m in input_paths:
                lines.append('{')
                lines.append(f'const float g=('+read(f'G[e*{p["p"]*c}+{path*c}+channel]')+f')*w{path}*mask;')
                for mag in range(m):
                    terms=[]
                    for i in range(ni):
                        coeff=columns[bb+i*bs+mag]
                        if coeff:terms.append(f'x{i}*('+_sum(f'{_literal(a)}*h{h}' for h,a in coeff)+')')
                    lines.append(f'y{path}_{mag}+=('+_sum(terms)+')*g;')
                lines.append('}')
            lines.append('}')
        lines.append('}')
        if neighbor_warps>1:
            slot=0
            for path,ni,xb,bb,bs,ob,m in paths:
                for mag in range(m):
                    lines.append(f'partial[(shard*{shared}+{slot})*32+lane]=y{path}_{mag};');slot+=1
            lines.append('__syncthreads();')
        lines.append(f'if(shard==0&&channel<{c}) {{' if neighbor_warps>1 else 'if(found) {')
        slot=0
        for path,ni,xb,bb,bs,ob,m in paths:
            for mag in range(m):
                value=_sum(f'partial[({shard*shared+slot})*32+lane]' for shard in range(neighbor_warps)) if neighbor_warps>1 else f'y{path}_{mag}'
                lines.append(f'O[{oi(ob,m,mag)}]+={value};');slot+=1
        lines.extend(['}','}'])
    lines.append('}')
    lines.append(f'''
void csr_forward(torch::Tensor x,torch::Tensor sh,torch::Tensor w,torch::Tensor g,torch::Tensor s,torch::Tensor mask,torch::Tensor ptr,torch::Tensor order,torch::Tensor order_hint,torch::Tensor out,torch::Tensor destinations,int64_t offset,bool unit) {{
    const c10::cuda::CUDAGuard guard(x.device());
    for(const auto& t:{{x,sh,w,g,mask,out}}) TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.scalar_type()==at::kFloat&&t.is_contiguous(),"Invalid CSR FP32 operand");
    for(const auto& t:{{s,ptr,order,order_hint,destinations}}) TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.scalar_type()==at::kLong&&t.is_contiguous(),"Invalid CSR topology");
    TORCH_CHECK(order_hint.numel()==0||order_hint.numel()==1,"Invalid CSR order hint");
    TORCH_CHECK(x.dim()==2&&x.size(1)=={p['xdim']}&&out.dim()==2&&out.size(1)=={p['odim']},"Invalid CSR node shapes");
    TORCH_CHECK(sh.dim()==2&&sh.size(0)==s.numel()&&sh.size(1)=={hd}&&g.dim()==3&&g.size(0)==s.numel()&&g.size(1)=={p['p']}&&g.size(2)=={c},"Invalid CSR edge shapes");
    TORCH_CHECK(w.dim()==2&&w.size(0)=={p['p']}&&w.size(1)=={c}&&ptr.numel()==out.size(0)+1,"Invalid CSR bank/pointer shape");
    TORCH_CHECK((mask.numel()==0||mask.numel()==s.numel())&&offset>=0&&offset+s.numel()<=order.numel(),"Invalid CSR edge slice");
    TORCH_CHECK(destinations.numel()==0||destinations.numel()==s.numel(),"Invalid CSR destination slice");
    if(!s.numel()||!out.size(0))return;
    const bool edge_owned=destinations.numel()&&s.numel()*2<out.size(0);
    const int64_t tasks=edge_owned?s.numel():out.size(0);
    const int64_t blocks={'tasks*'+str(tiles*len(groups)) if neighbor_warps>1 else '(tasks*'+str(tiles*len(groups))+'+3)/4'};
    TORCH_CHECK(blocks<(1LL<<31),"CSR grid exceeds supported size");
    auto stream=at::cuda::getCurrentCUDAStream();
    #define CALL(U,B) csr_kernel<U,B><<<blocks,128,0,stream>>>(x.data_ptr<float>(),sh.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),s.data_ptr<int64_t>(),mask.data_ptr<float>(),ptr.data_ptr<int64_t>(),order.data_ptr<int64_t>(),order_hint.numel()?order_hint.data_ptr<int64_t>():nullptr,out.data_ptr<float>(),destinations.data_ptr<int64_t>(),out.size(0),s.numel(),offset,order.numel(),mask.numel()!=0)
    if(edge_owned) {{ if(unit) {{ CALL(true,true); }} else {{ CALL(false,true); }} }}
    else {{ if(unit) {{ CALL(true,false); }} else {{ CALL(false,false); }} }}
    #undef CALL
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
''')
    return '\n'.join(lines)+'\n'


def _extension(p,projection):
    cache=projection.setdefault('generated_csr_extensions',{})
    limit=p.get('csr_accumulator_limit',48)
    if not isinstance(limit,int) or isinstance(limit,bool) or limit<1:
        raise ValueError('CSR accumulator limit must be a positive integer')
    key=(p['path_specs'],p['xdim'],p['odim'],p['c'],p.get('generated_layout','packed'),limit)
    neighbor_warps=int(p.get('csr_neighbor_warps',1));key=(*key,neighbor_warps)
    slice_search=bool(p.get('csr_slice_search',False));key=(*key,slice_search,'abi3')
    if p.get('compact_cartesian',False):key=(*key,'compact')
    if p.get('inference_degree_layout',False):key=(*key,'degree')
    if key not in cache:
        from .native_jit import load_inline
        name=f'mace_ictc_csr_{p["xdim"]}_{p["odim"]}_{p["p"]}_abi3'
        if limit!=48:name+=f'_a{limit}'
        if p.get('generated_layout')=='canonical':name+='_canonical'
        if p.get('compact_cartesian',False):name+='_compact'
        if p.get('inference_degree_layout',False):name+='_degree'
        if neighbor_warps>1:name+=f'_nw{neighbor_warps}'
        if slice_search:name+='_slice'
        declarations='#include <torch/extension.h>\nvoid csr_forward('+','.join(['torch::Tensor']*11)+',int64_t,bool);\n'
        cache[key]=load_inline(name=name,cpp_sources=declarations,cuda_sources=_source(p,projection),
            functions=['csr_forward'],extra_cflags=['-O3'],extra_cuda_cflags=['-O3'] if torch.version.hip else ['-O3','--ptxas-options=-v'],verbose=False)
    return cache[key]


def forward(x,sh,w,g,src,mask,out,p,projection,csr,start,dst=None):
    ptr,order=csr[:2]
    hint=csr[2] if len(csr)==3 else src.new_zeros(1)
    destinations=src.new_empty(0) if dst is None else dst.contiguous()
    _extension(p,projection).csr_forward(x,sh.contiguous(),w,g,src,mask,ptr,order,hint,out,
        destinations,start,bool(p.get('unit_weights',False)))
