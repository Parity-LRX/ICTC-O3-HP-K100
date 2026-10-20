# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Independent generated FP32 Cartesian TP/scatter and joint first adjoints.

The schedule borrows public OpenEquivariance ideas (warp-local work, constant
unrolling and joint adjoints), using this model's actual Cartesian projector
coefficients. No OpenEquivariance or cuEq runtime/kernel is called. Generated
source and compiled objects belong in the configured remote task cache.
"""
import torch


def _literal(value):
    value=float(value)
    text=format(value,".17g")
    if "." not in text and "e" not in text: text+=".0"
    return text+"f"


def _sum(parts):
    return " + ".join(f"({part})" for part in parts) if parts else "0.0f"


def _channel_read(expression, half, channels, tile=0, width=32):
    if tile:
        if channels%tile==0:return expression
        return f"(channel_base+{half*width}+lane<{channels})?({expression}):0.0f"
    if (half+1)*width <= channels:
        return expression
    return f"({half*width}+lane<{channels})?({expression}):0.0f"


def _channel_write(statement, half, channels, tile=0, width=32):
    if tile:
        if channels%tile==0:return statement
        return f"if(channel_base+{half*width}+lane<{channels}) {{ {statement} }}"
    if (half+1)*width <= channels:
        return statement
    return f"if({half*width}+lane<{channels}) {{ {statement} }}"


def _stream_geometry_path(p,columns,row,halves,tile,output_index,backward):
    path,ni,xb,bb,bs,ob,m=row;c=p['c']
    width=int(p.get('generated_lane_width',32))
    lines=[]
    def geometry(i,mag):
        terms=columns[bb+i*bs+mag]
        if terms:lines.append(f'const float b{i}_{mag}='+_sum(f'{_literal(a)}*h{h}' for h,a in terms)+';')
    if not backward:
        for mag in range(m):
            lines.append('{')
            for i in range(ni):geometry(i,mag)
            for half in halves:
                terms=[f'x{half}_{i}*b{i}_{mag}' for i in range(ni) if columns[bb+i*bs+mag]]
                lines.append(f'const float y{half}=('+_sum(terms)+f')*w{half}*g{half}*mask;')
                pointer=f'O+{output_index(ob,m,mag,half)}'
                statement=f'*({pointer})=y{half};' if p.get('edge_only',False) else f'atomicAdd({pointer},y{half});'
                lines.append(_channel_write(statement,half,c,tile,width))
            lines.append('}')
    else:
        for half in halves:lines.append(f'float dg{half}=0.0f;')
        for i in range(ni):
            lines.append('{')
            for mag in range(m):geometry(i,mag)
            for half in halves:
                terms=[f'b{i}_{mag}*v{half}_{mag}' for mag in range(m) if columns[bb+i*bs+mag]]
                lines.append(f'const float z{half}='+_sum(terms)+';')
                lines.append(f'if(needx) dx{half}_{i}+=z{half}*(w{half}*g{half});')
                lines.append(f'if(needg) dg{half}+=x{half}_{i}*z{half};')
            lines.append('}')
        base='channel_base+' if tile else ''
        for half in halves:
            lines.append(_channel_write(f'if(needg) DG[e*{p["p"]*c}+{path*c+half*width}+{base}lane]=dg{half}*w{half};',half,c,tile,width))
        lines.append('if(needh) {')
        for i in range(ni):
            for mag in range(m):
                terms=columns[bb+i*bs+mag]
                if not terms:continue
                value=_sum(f'x{half}_{i}*(v{half}_{mag}*(g{half}*w{half}))' for half in halves)
                lines.append(f'const float db{i}_{mag}={value};')
                for h,a in terms:lines.append(f'dh{h}+=({_literal(a)})*db{i}_{mag};')
        lines.append('}')
    return lines


def _source(p, projection):
    c=p["c"]; hd=projection["hdim"]
    width=int(p.get('generated_lane_width',32))
    tile=int(p.get("generated_channel_tile",0));tiles=(c+tile-1)//tile if tile else 1
    halves=range(((tile or c)+width-1)//width)
    edge_only=bool(p.get("edge_only",False))
    canonical=p.get("generated_layout","packed")=="canonical"
    degree_layout=bool(p.get('inference_degree_layout',False))
    streaming=bool(p.get('generated_geometry_streaming',False))
    specialized=bool(p.get('generated_specialized_adjoint',False))
    partition=p.get('generated_input_partition','')
    input_groups=list(dict.fromkeys((row[2],row[1]) for row in p['path_specs']))
    partitions=len(input_groups) if partition else 1
    def input_index(xb,ni,i,half):
        base="channel_base+" if tile else ""
        return f"{xb}+({base}{half*width}+lane)*{ni}+{i}" if canonical else f"{xb+i*c+half*width}+{base}lane"
    def output_index(ob,m,mag,half):
        base="channel_base+" if tile else ""
        if degree_layout:
            degree=(m-1)//2;start=p['ooffset'][degree];degree_width=p['counts'][degree]*c
            return f"{start}*N+(dst*{m}+{mag})*{degree_width}+{(ob-start)//m+half*width}+{base}lane"
        local=(f"{ob}+({base}{half*width}+lane)*{m}+{mag}" if canonical
               else f"{ob+mag*c+half*width}+{base}lane")
        return f"dst*{p['odim']}+{local}"
    columns=[[] for _ in range(p["bdim"])]
    for h,rows in enumerate(projection["static_rows"]):
        for col,a in rows: columns[col].append((h,a))
    common=["#include <torch/extension.h>", "#include <ATen/cuda/CUDAContext.h>",
            "#include <c10/cuda/CUDAGuard.h>", "#include <c10/cuda/CUDAException.h>",
            "#include <cuda_runtime.h>"]
    signature="const float* X,const float* SH,const float* W,const float* G,const int64_t* S,const int64_t* D,const float* MASK,const float* V,float* O,float* DX,float* DH,float* DG,int64_t E,bool masked,bool needx,bool needh,bool needg,int schedule"
    # Edge outputs are freshly allocated and disjoint from all read operands.
    # Restrict lets HIP keep shared geometry and addresses out of live vectors.
    if p.get('generated_restrict',False):signature=signature.replace('* ','* __restrict__ ')
    if degree_layout:signature+=",int64_t N"
    for backward in (False,True):
        name="adjoint_kernel" if backward else "forward_kernel"
        kernel_signature=(signature.replace('bool needx,bool needh,bool needg','bool requestx,bool requesth,bool requestg')
                          if backward and specialized else signature)
        template='template<bool Unit,bool NeedX,bool NeedH,bool NeedG>' if backward and specialized else 'template<bool Unit>'
        common.extend([template,f"__global__ void {name}({kernel_signature}) {{",
            'const bool needx=NeedX,needh=NeedH,needg=NeedG;' if backward and specialized else '',
            f"const int lane=threadIdx.x%{width};",
            f"const int64_t warp=(static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x)/{width};",
            f"const int64_t warps=static_cast<int64_t>(gridDim.x)*blockDim.x/{width};",
            f"const int64_t tasks=E*{tiles};" if not partition else f"const int64_t tasks=E*{tiles*partitions};",
            "const int64_t chunk=(tasks+warps-1)/warps;",
            "const int64_t begin=schedule?warp*chunk:warp;",
            "const int64_t end=schedule?min(begin+chunk,tasks):tasks;",
            "const int64_t step=schedule?1:warps;",
            "for(int64_t task=begin;task<end;task+=step) {",
            f"const int64_t e=task/{tiles};" if not partition else
            (f"const int group=task%{partitions}; const int64_t e=task/{tiles*partitions};" if partition=='interleave' else
             f"const int group=(task/{tiles})%{partitions}; const int64_t e=task/{tiles*partitions};" if partition=='tilefirst' else
             f"const int group=task/(E*{tiles}); const int64_t e=(task/{tiles})%E;"),
            f"const int64_t channel_base=((task/{partitions})%{tiles})*{tile};" if partition=='interleave' else f"const int64_t channel_base=(task%{tiles})*{tile};",
            "const int64_t src=e,dst=e;" if edge_only else "const int64_t src=S[e],dst=D[e];",
            "const float mask=masked?MASK[e]:1.0f;"])
        for h in range(hd):common.append(f"const float h{h}=SH[e*{hd}+{h}];")
        if backward:
            for h in range(hd):common.append(f"float dh{h}=0.0f;")
        by_input={}
        for row in p["path_specs"]:
            path,ni,xb,bb,bs,ob,m=row
            by_input.setdefault((xb,ni),[]).append(row)
        for (xb,ni),paths in by_input.items():
            # Each task owns one input block and its disjoint output/gate paths.
            # Only the small harmonic gradient needs cross-task accumulation.
            common.append(f"if(group=={input_groups.index((xb,ni))}) {{" if partition else "{")
            for i in range(ni):
                for half in halves:
                    value=_channel_read(f"X[src*{p['xdim']}+{input_index(xb,ni,i,half)}]",half,c,tile,width)
                    common.append(f"const float x{half}_{i}={value};")
                    if backward:common.append(f"float dx{half}_{i}=0.0f;")
            for path,ni,xb,bb,bs,ob,m in paths:
                common.append("{")
                for half in halves:
                    base="channel_base+" if tile else ""
                    value=_channel_read(f"Unit?1.0f:W[{path*c+half*width}+{base}lane]",half,c,tile,width)
                    common.append(f"const float w{half}={value};")
                    value=_channel_read(f"G[e*{p['p']*c}+{path*c+half*width}+{base}lane]",half,c,tile,width)
                    common.append(f"const float g{half}={value};")
                    if backward:
                        for mag in range(m):
                            value=_channel_read(f"V[{output_index(ob,m,mag,half)}]*mask",half,c,tile,width)
                            common.append(f"const float v{half}_{mag}={value};")
                if streaming:
                    common.extend(_stream_geometry_path(p,columns,(path,ni,xb,bb,bs,ob,m),halves,tile,output_index,backward))
                    common.append('}')
                    continue
                for i in range(ni):
                    for mag in range(m):
                        terms=columns[bb+i*bs+mag]
                        if terms:common.append(f"const float b{i}_{mag}="+_sum(f"{_literal(a)}*h{h}" for h,a in terms)+";")
                if not backward:
                    for half in halves:
                        for mag in range(m):
                            terms=[f"x{half}_{i}*b{i}_{mag}" for i in range(ni) if columns[bb+i*bs+mag]]
                            value=_sum(terms)
                            common.append(f"const float y{half}_{mag}=({value})*w{half}*g{half}*mask;")
                            pointer=f"O+{output_index(ob,m,mag,half)}"
                            statement=f"*({pointer})=y{half}_{mag};" if edge_only else f"atomicAdd({pointer},y{half}_{mag});"
                            common.append(_channel_write(statement,half,c,tile,width))
                else:
                    for half in halves:
                        for i in range(ni):
                            terms=[f"b{i}_{mag}*v{half}_{mag}" for mag in range(m) if columns[bb+i*bs+mag]]
                            common.append(f"const float z{half}_{i}="+_sum(terms)+";")
                            common.append(f"if(needx) dx{half}_{i}+=z{half}_{i}*(w{half}*g{half});")
                        dg=_sum(f"x{half}_{i}*z{half}_{i}" for i in range(ni))
                        base="channel_base+" if tile else ""
                        common.append(_channel_write(f"if(needg) DG[e*{p['p']*c}+{path*c+half*width}+{base}lane]=({dg})*w{half};",half,c,tile,width))
                    common.append("if(needh) {")
                    for i in range(ni):
                        for mag in range(m):
                            terms=columns[bb+i*bs+mag]
                            if not terms:continue
                            value="+".join(f"x{half}_{i}*(v{half}_{mag}*(g{half}*w{half}))" for half in halves)
                            common.append(f"const float db{i}_{mag}={value};")
                            for h,a in terms:common.append(f"dh{h}+=({_literal(a)})*db{i}_{mag};")
                    common.append("}")
                common.append("}")
            if backward:
                common.append("if(needx) {")
                for half in halves:
                    for i in range(ni):
                        pointer=f"DX+src*{p['xdim']}+{input_index(xb,ni,i,half)}"
                        statement=f"*({pointer})=dx{half}_{i};" if edge_only else f"atomicAdd({pointer},dx{half}_{i});"
                        common.append(_channel_write(statement,half,c,tile,width))
                common.append("}")
            common.append("}")
        if backward:
            common.append("if(needh) {")
            for h in range(hd):
                # Match the reduction to the logical group, including wave64 HIP.
                shuffle=(f"__shfl_down(dh{h},shift,{width})" if torch.version.hip else
                         f"__shfl_down_sync(0xffffffff,dh{h},shift,32)")
                common.append(f"for(int shift={width//2};shift>0;shift/=2) dh{h}+={shuffle};")
                common.append(f"if(lane==0) atomicAdd(DH+e*{hd}+{h},dh{h});" if tile or partition else f"if(lane==0) DH[e*{hd}+{h}]=dh{h};")
            common.append("}")
        common.extend(["}","}"])
    if specialized:
        calls=[]
        for unit in ('true','false'):
            branches=[]
            for nx in (False,True):
                for nh in (False,True):
                    for ng in (False,True):
                        if not (nx or nh or ng):continue
                        condition='&&'.join(name if value else '!'+name for name,value in zip(('needx','needh','needg'),(nx,nh,ng)))
                        flags=','.join(str(v).lower() for v in (nx,nh,ng))
                        branches.append(f'if({condition}){{CALL({unit},{flags});}}')
            calls.append(('if(unit)' if unit=='true' else 'else')+'{'+''.join(branches)+'}')
        launch_adjoint='''#define CALL(U,X,H,G) adjoint_kernel<U,X,H,G><<<blocks,128,0,stream>>>(x.data_ptr<float>(),sh.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),s.data_ptr<int64_t>(),d.data_ptr<int64_t>(),mask.data_ptr<float>(),v.data_ptr<float>(),nullptr,dx.data_ptr<float>(),dh.data_ptr<float>(),dg.data_ptr<float>(),s.numel(),mask.numel()!=0,needx,needh,needg,schedule)
'''+ '\n'.join(calls)
    else:
        launch_adjoint='''#define CALL(U) adjoint_kernel<U><<<blocks,128,0,stream>>>(x.data_ptr<float>(),sh.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),s.data_ptr<int64_t>(),d.data_ptr<int64_t>(),mask.data_ptr<float>(),v.data_ptr<float>(),nullptr,dx.data_ptr<float>(),dh.data_ptr<float>(),dg.data_ptr<float>(),s.numel(),mask.numel()!=0,needx,needh,needg,schedule)
    if(unit){CALL(true);}else{CALL(false);}'''
    if degree_layout:launch_adjoint=launch_adjoint.replace('needg,schedule)','needg,schedule,v.size(0))')
    if p.get('generated_split_adjoint',False):
        # Reuse the allocated outputs. Separating harmonic reduction from the
        # feature/gate adjoint shortens register live ranges without workspace.
        invoke=launch_adjoint+'\n    #undef CALL\n'
        launch_adjoint=('if(needh&&(needx||needg)) {\n'
                        'needh=false;\n'+invoke+
                        'needx=false;needh=true;needg=false;\n'+invoke+
                        '} else {\n'+invoke+'}\n')
    common.append(f"""
void validate(torch::Tensor x,torch::Tensor sh,torch::Tensor w,torch::Tensor g,torch::Tensor s,torch::Tensor d,torch::Tensor mask) {{
    for(const auto& t:{{x,sh,w,g,mask}}) TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.scalar_type()==at::kFloat&&t.is_contiguous(),"Invalid generated message FP32 operand");
    for(const auto& t:{{s,d}}) TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.scalar_type()==at::kLong&&t.is_contiguous(),"Invalid generated message topology");
    TORCH_CHECK(x.dim()==2&&x.size(1)=={p['xdim']}&&sh.dim()==2&&sh.size(1)=={hd}&&sh.size(0)==s.numel(),"Invalid generated message features");
    TORCH_CHECK(w.dim()==2&&w.size(0)=={p['p']}&&w.size(1)=={c}&&g.dim()==3&&g.size(0)==s.numel()&&g.size(1)=={p['p']}&&g.size(2)=={c}&&d.numel()==s.numel(),"Invalid generated message gates");
    TORCH_CHECK(mask.numel()==0||mask.numel()==s.numel(),"Invalid generated message mask");
}}
void generated_forward(torch::Tensor x,torch::Tensor sh,torch::Tensor w,torch::Tensor g,torch::Tensor s,torch::Tensor d,torch::Tensor mask,torch::Tensor out,bool unit,int64_t blocks,int64_t schedule) {{
    const c10::cuda::CUDAGuard guard(x.device()); validate(x,sh,w,g,s,d,mask);
    TORCH_CHECK(out.is_cuda()&&out.device()==x.device()&&out.scalar_type()==at::kFloat&&out.is_contiguous()&&out.dim()==2&&out.size(1)=={p['odim']},"Invalid generated message output");
    TORCH_CHECK(blocks>0&&blocks<(1LL<<31)&&(schedule==0||schedule==1),"Invalid generated launch");
    if(!s.numel())return;
    auto stream=at::cuda::getCurrentCUDAStream();
    #define CALL(U) forward_kernel<U><<<blocks,128,0,stream>>>(x.data_ptr<float>(),sh.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),s.data_ptr<int64_t>(),d.data_ptr<int64_t>(),mask.data_ptr<float>(),nullptr,out.data_ptr<float>(),nullptr,nullptr,nullptr,s.numel(),mask.numel()!=0,false,false,false,schedule{',out.size(0)' if degree_layout else ''})
    if(unit){{CALL(true);}}else{{CALL(false);}}
    #undef CALL
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
std::vector<torch::Tensor> generated_adjoint(torch::Tensor x,torch::Tensor sh,torch::Tensor w,torch::Tensor g,torch::Tensor s,torch::Tensor d,torch::Tensor mask,torch::Tensor v,torch::Tensor dx,bool unit,bool needx,bool needh,bool needg,int64_t blocks,int64_t schedule,torch::Tensor dh_out,torch::Tensor dg_out) {{
    const c10::cuda::CUDAGuard guard(x.device()); validate(x,sh,w,g,s,d,mask);
    TORCH_CHECK(v.is_cuda()&&v.device()==x.device()&&v.scalar_type()==at::kFloat&&v.is_contiguous()&&v.dim()==2&&v.size(1)=={p['odim']},"Invalid generated cotangent");
    TORCH_CHECK(!needx||(dx.is_cuda()&&dx.device()==x.device()&&dx.scalar_type()==at::kFloat&&dx.is_contiguous()&&dx.sizes()==x.sizes()),"Invalid generated feature gradient");
    TORCH_CHECK(blocks>0&&blocks<(1LL<<31)&&(schedule==0||schedule==1),"Invalid generated launch");
    for(const auto& t:{{dh_out,dg_out}}) TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.scalar_type()==at::kFloat&&t.is_contiguous(),"Invalid adjoint destination");
    TORCH_CHECK(!dh_out.numel()||(dh_out.dim()==2&&dh_out.size(0)==s.numel()&&dh_out.size(1)=={hd}),"Invalid harmonic destination");
    TORCH_CHECK(!dg_out.numel()||dg_out.sizes()==g.sizes(),"Invalid gate destination");
    auto dh=needh&&dh_out.numel()?dh_out:torch::empty({{needh?s.numel():0,{hd}}},x.options());
    auto dg=needg?(dg_out.numel()?dg_out:torch::empty_like(g)):torch::empty({{0}},x.options());
    {'if(needh)dh.zero_();' if (tile or partition) and not p.get('generated_accumulate_harmonic',False) else ''}
    if(!s.numel())return {{dh,dg}};
    auto stream=at::cuda::getCurrentCUDAStream();
    {launch_adjoint}
    #undef CALL
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {{dh,dg}};
}}
""")
    source="\n".join(common)+"\n"
    threads=int(p.get('generated_threads',128))
    return source.replace('<<<blocks,128,',f'<<<blocks,{threads},')


def _extension(p, projection):
    if p["c"]<1:raise ValueError("Generated Cartesian message requires positive channels")
    cache=projection.setdefault("generated_message_extensions",{})
    registers=int(p.get("generated_reg_limit",0))
    tile=int(p.get("generated_channel_tile",0))
    width=int(p.get('generated_lane_width',32))
    partition=p.get('generated_input_partition','')
    if partition not in ('','interleave','tilefirst','contiguous'):
        raise ValueError('Unsupported generated input partition order')
    if p.get('generated_restrict',False) and not p.get('edge_only',False):
        raise ValueError('Restrict scheduling requires disjoint edge outputs')
    threads=int(p.get('generated_threads',128))
    if threads not in (64,128,192,256) or (threads!=128 and not torch.version.hip):
        raise ValueError('Nondefault generated block sizes require HIP')
    if width not in (32,64) or (width==64 and not torch.version.hip):
        raise ValueError('A 64-lane generated group requires HIP')
    if tile and tile<width:raise ValueError('Channel tile must cover a full logical group')
    if tile not in (0,32,64):raise ValueError("Unsupported generated channel tile")
    if registers not in (0,64,96,128):raise ValueError("Unsupported generated register limit")
    streaming=bool(p.get('generated_geometry_streaming',False))
    key=(p["path_specs"],p["xdim"],p["odim"],p["c"],registers,tile,streaming)
    if width!=32:key=(*key,'wave64')
    if threads!=128:key=(*key,'threads',threads)
    if p.get('generated_split_adjoint',False):key=(*key,'split_adjoint')
    if partition:key=(*key,'partition',partition)
    if p.get('generated_restrict',False):key=(*key,'restrict')
    canonical=p.get("generated_layout","packed")=="canonical"
    if canonical:key=(*key,"canonical")
    edge_only=bool(p.get("edge_only",False))
    if edge_only:key=(*key,"edges")
    if p.get('compact_cartesian',False):key=(*key,'compact')
    if p.get('generated_specialized_adjoint',False):key=(*key,'roles')
    if p.get('inference_degree_layout',False):key=(*key,'degree')
    if 'generated_input_group' in p:key=(*key,p['generated_input_group'])
    if p.get('generated_accumulate_harmonic',False):key=(*key,'accumulate_harmonic')
    if key in cache:return cache[key]
    from .native_jit import load_inline
    # Source-based JIT versioning invalidates changes to actual constant values.
    name=f"mace_ictc_cartesian_message_{p['xdim']}_{p['odim']}_{p['p']}"+(f"_r{registers}" if registers else "")
    if tile:name+=f"_ct{tile}"
    if width!=32:name+='_w64'
    if threads!=128:name+=f'_t{threads}'
    if p.get('generated_split_adjoint',False):name+='_sa'
    if partition:name+='_part_'+partition
    if p.get('generated_restrict',False):name+='_restrict'
    if streaming:name+='_gs'
    if canonical:name+="_canonical"
    if edge_only:name+="_edges"
    if p.get('compact_cartesian',False):name+='_compact'
    if p.get('generated_specialized_adjoint',False):name+='_roles'
    if p.get('inference_degree_layout',False):name+='_degree'
    if 'generated_input_group' in p:name+=f'_ig{p["generated_input_group"]}'
    if p.get('generated_accumulate_harmonic',False):name+='_accumulate_harmonic'
    declarations="""#include <torch/extension.h>
void generated_forward(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,bool,int64_t,int64_t);
std::vector<torch::Tensor> generated_adjoint(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,bool,bool,bool,bool,int64_t,int64_t,torch::Tensor,torch::Tensor);
"""
    flags=["-O3"]
    if torch.version.hip:
        if registers:
            raise ValueError("CUDA register limits are not supported by the HIP backend")
    else:
        flags.append("--ptxas-options=-v")
        if registers:flags.append(f"--maxrregcount={registers}")
    extension=load_inline(name=name,cpp_sources=declarations,cuda_sources=_source(p,projection),
                          functions=["generated_forward","generated_adjoint"],
                          extra_cflags=["-O3"],extra_cuda_cflags=flags,verbose=False)
    cache[key]=extension
    return extension


def _launch(p,x,edges):
    blocks=p.get("generated_blocks")
    if blocks is None:blocks=torch.cuda.get_device_properties(x.device).multi_processor_count*4
    if blocks==0:
        tile=int(p.get("generated_channel_tile",0));tiles=(p["c"]+tile-1)//tile if tile else 1
        groups=int(p.get('generated_threads',128))//int(p.get('generated_lane_width',32))
        partitions=len(set((row[2],row[1]) for row in p['path_specs'])) if p.get('generated_input_partition') else 1
        blocks=max(1,(edges*tiles*partitions+groups-1)//groups)
    return int(blocks),int(p.get("generated_schedule",1))


def generated_forward(x,sh,w,g,src,dst,mask,out,p,projection):
    if src.numel():
        if p.get('generated_cuda_groups',False):
            for gp in _input_plans(p):
                generated_forward(x,sh,w,g,src,dst,mask,out,gp,projection)
            return
        _extension(p,projection).generated_forward(x,sh.contiguous(),w,g,src,dst,mask,out,
                                                   bool(p.get("unit_weights",False)),*_launch(p,x,src.numel()))


def generated_adjoint(x,sh,w,g,src,dst,mask,v,dx,p,projection,needed,*,harmonic_output=None,gate_output=None):
    if not src.numel():
        return sh.new_empty(sh.shape) if needed[1] else None,torch.empty_like(g) if needed[2] else None
    if p.get('generated_split_input_adjoint',False) and p.get('generated_input_partition'):
        dh=(harmonic_output if harmonic_output is not None else torch.empty_like(sh)) if needed[1] else None
        dg=(gate_output if gate_output is not None else torch.empty_like(g)) if needed[2] else None
        for index,gp in enumerate(_input_plans(p)):
            gp={**gp,'generated_split_input_adjoint':False,'generated_accumulate_harmonic':index>0}
            generated_adjoint(x,sh,w,g,src,dst,mask,v,dx,gp,projection,needed,
                              harmonic_output=dh,gate_output=dg)
        return dh,dg
    if p.get('generated_cuda_groups',False) and not needed[1]:
        dg=gate_output if gate_output is not None else torch.empty_like(g) if needed[2] else None
        for gp in _input_plans(p):
            generated_adjoint(x,sh,w,g,src,dst,mask,v,dx,gp,projection,needed,gate_output=dg)
        return None,dg
    rawdx=x.new_empty(0) if dx is None else dx
    dh_out=x.new_empty(0) if harmonic_output is None else harmonic_output
    dg_out=x.new_empty(0) if gate_output is None else gate_output
    dh,dg=_extension(p,projection).generated_adjoint(x,sh.contiguous(),w,g,src,dst,mask,v,rawdx,
        bool(p.get("unit_weights",False)),*needed,*_launch(p,x,src.numel()),dh_out,dg_out)
    return dh if needed[1] else None,dg if needed[2] else None


def _input_plans(p):
    groups={}
    for row in p['path_specs']:groups.setdefault((row[2],row[1]),[]).append(row)
    return [{**p,'path_specs':tuple(rows),'generated_input_group':number,'generated_cuda_groups':False}
            for number,rows in enumerate(groups.values())]
