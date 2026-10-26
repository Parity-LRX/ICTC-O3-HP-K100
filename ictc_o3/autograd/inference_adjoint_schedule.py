# SPDX-License-Identifier: MIT
"""Bounded, metadata-only dispatch for frozen-gate coordinate adjoints.

No shape-specific timing, device allocations or synchronization are performed.
The score is a source-level live-value proxy, not a measured register count.
"""


def _qualified_device(name, capability):
    """Keep unmeasured devices on the original schedule."""
    return (capability == (8, 0) and 'A100' in name) or (
        capability == (8, 9) and name == 'NVIDIA GeForce RTX 4090 D')


def choose_channel_tile(plan, edges, capability, need_features=True):
    """Return 0 (original schedule) or 64 for a qualified workload."""
    if not need_features or capability not in ((8, 0), (8, 9)) or edges < 4096:
        return 0
    if not (plan.get('inference_degree_layout', False)
            and plan.get('generated_geometry_streaming', False)
            and plan.get('generated_specialized_adjoint', False)):
        return 0
    channels = plan['c']
    paths = plan['path_specs']
    if not 128 <= channels <= 256 or not paths:
        return 0
    # x, dx, output cotangents, gate/weight/gate-gradient values per lane.
    # Large angular inputs and wide channels jointly create live-value pressure.
    width = max(2*row[1]+row[-1]+3 for row in paths)
    max_input = max(row[1] for row in paths)
    max_output = max(row[-1] for row in paths)
    # L2/C256 is a measured counterexample: the live-value score alone does not
    # predict whether splitting improves end-to-end force latency.
    if max_input != 7 or max_output != 7:
        return 0
    score = ((channels+31)//32)*width
    return 64 if score >= 144 else 0


def adjoint_plan(plan, x, edges, need_features, *, need_geometry=True, need_gate=True):
    """Resolve the requested policy once per streamed backward invocation."""
    requested = plan.get('inference_adjoint_channel_tile', 0)
    if not need_features or requested == 0:
        return plan
    if requested == 'auto':
        if not (need_geometry and need_gate):
            return plan
        # Reject small/light work using host metadata before querying a device.
        tile = choose_channel_tile(plan, edges, (8, 0))
        if not tile:
            return plan
        import torch
        properties = torch.cuda.get_device_properties(x.device)
        capability = (properties.major, properties.minor)
        if not _qualified_device(properties.name, capability):
            return plan
    else:
        tile = requested
    return {**plan, 'generated_channel_tile': tile} if tile else plan
