"""Species/channel GEMMs for live correlation-one/two/three coefficients.

The highest coefficient bank is contracted with features before any node-sized
recursion state is formed. Values are freshly derived from learned weights.
"""
import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch._subclasses.fake_tensor import is_fake


def banked_contraction(module,x,idx):
    if (module.correlation>3 or torch.compiler.is_compiling() or get_proxy_mode() is not None
            or torch._C._are_functorch_transforms_active() or is_fake(x)):
        return None
    idx=idx.to(device=x.device,dtype=torch.long)
    correlation=module.correlation;n,c,i=x.shape;m=2*module.output_lmax+1
    weights=[module.weights_max,*module.weights]
    banks={}
    for order,weight in zip(range(correlation,0,-1),weights):
        u=module.U_tensors(order).to(x)
        if module.output_lmax==0:u=u.unsqueeze(0)
        banks[order]=torch.einsum("m...p,zpc->zcm...",u,weight.to(x))
    output=x.new_zeros(n,c,m)
    for species in range(module.weights_max.shape[0]):
        ids=torch.nonzero(idx==species,as_tuple=True)[0]
        if ids.numel()==0:continue
        features=x.index_select(0,ids)
        # C independent GEMMs: [C,N_species,I] @ [C,I,M*I^(nu-1)].
        matrix=banks[correlation][species].reshape(c,m*i**(correlation-1),i).transpose(1,2)
        state=torch.bmm(features.transpose(0,1),matrix).transpose(0,1)
        state=state.reshape(ids.numel(),c,m,*([i]*(correlation-1)))
        for order in range(correlation-1,0,-1):
            state=state+banks[order][species].unsqueeze(0)
            state=torch.einsum("nc...i,nci->nc...",state,features)
        output=output.index_add(0,ids,state)
    return output.reshape(n,c*m)
