# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Species-conditioned GEMMs without an [N,C_out,C_in] weight expansion.

Grouped GEMMs support arbitrary species banks. The preceding shared FP32
schedule remains available for small banks. All operations are ordinary PyTorch,
so learned banks and mixed/higher derivatives remain live.
"""
import torch
import torch.nn.functional as F


def species_groups(index, num_species):
    """Return a row permutation and spans for arbitrary integer species labels.

    Only the small list of occupied species/counts crosses to the host. Reuse
    this plan across angular blocks, but rebuild it when the node list changes
    (in particular, after a distributed halo/owner update).
    """
    if index.ndim != 1 or index.dtype != torch.long:
        raise ValueError("Species indices must be a one-dimensional long tensor")
    labels, order = torch.sort(index)
    species, counts = torch.unique_consecutive(labels, return_counts=True)
    spans = torch.stack((species, counts)).cpu().tolist()
    if any(label < 0 or label >= num_species for label in spans[0]):
        raise IndexError("Species index outside the weight bank")
    inverse = torch.empty_like(order)
    inverse.scatter_(0, order, torch.arange(order.numel(), device=order.device))
    return order, inverse, tuple(zip(*spans))


def grouped_species_linear(block, weight, index, bias=None, *, groups=None):
    """Apply a species bank without materializing per-node channel matrices.

    block[N,I,M], weight[S,O,I], optional bias[S,O]. Each occupied species
    performs one ordinary GEMM on its own rows. Shapes, species count, angular
    size and floating dtype are unrestricted; input/parameter gradients and
    higher derivatives remain ordinary PyTorch autograd operations. Eager only:
    extracting the group sizes deliberately synchronizes the small metadata.
    """
    if block.ndim != 3 or weight.ndim != 3 or index.shape != (block.shape[0],):
        raise ValueError("Expected block[N,I,M], weight[S,O,I], index[N]")
    if block.shape[1] != weight.shape[2]:
        raise ValueError("Input channels do not match the species weight bank")
    if bias is not None and bias.shape != weight.shape[:2]:
        raise ValueError("Expected bias[S,O]")
    if groups is None:
        groups = species_groups(index, weight.shape[0])
    order, inverse, spans = groups
    angular = block.index_select(0, order).movedim(-2, -1)
    if not spans:
        # Preserve zero gradients for every operand on an empty node list.
        zero = block.sum() * 0 + weight.sum() * 0
        if bias is not None:
            zero = zero + bias.sum() * 0
        return block.new_empty((0, weight.shape[1], block.shape[2])) + zero
    values = []
    start = 0
    for species, count in spans:
        rows = angular.narrow(0, start, count).contiguous()
        value = F.linear(rows.reshape(-1, rows.shape[-1]), weight[species],
                         None if bias is None else bias[species])
        values.append(value.reshape(count, block.shape[2], weight.shape[1]))
        start += count
    return torch.cat(values, dim=0).index_select(0, inverse).movedim(-1, -2)


def shared_species_linear(block, weight, index, bias=None):
    """Apply weight[S,O,I] to block[N,I,M] selected by fixed index[N]."""
    torch._assert_async(((index>=0)&(index<weight.shape[0])).all(),"Species index outside the weight bank")
    angular=block.movedim(-2,-1).contiguous()
    flat=angular.reshape(-1,angular.shape[-1])
    result=None
    for species in range(weight.shape[0]):
        value=F.linear(flat,weight[species],None if bias is None else bias[species])
        value=value.reshape(*angular.shape[:-1],weight.shape[1])
        selected=torch.where(index[:,None,None]==species,value,0.)
        result=selected if result is None else result+selected
    return result.movedim(-1,-2)


def supported(block, weight):
    return (block.is_cuda and block.dtype==torch.float32 and weight.dtype==block.dtype
            and weight.device==block.device and 1<=weight.shape[0]<=4
            and not torch.is_autocast_enabled(block.device.type))
