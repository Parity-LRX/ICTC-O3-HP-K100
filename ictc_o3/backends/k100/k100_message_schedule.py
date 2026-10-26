"""Shared K100 scheduling for direct edge TP and streamed graph inference."""
import torch


def schedule(plan,tensor):
    if not (torch.version.hip and tensor.is_cuda and plan['c']>=64):return plan
    if getattr(torch.cuda.get_device_properties(tensor.device),'gcnArchName','').split(':')[0]!='gfx926':return plan
    # The qualified degree range is expressed by actual magnetic block widths.
    specs=plan['path_specs']
    if not specs or any(row[1]>7 or row[-1]>7 for row in specs):return plan
    groups={(row[2],row[1]) for row in specs}
    result={**plan,'generated_lane_width':64,
            'generated_channel_tile':64 if plan['c']>64 else 0}
    if len(groups)>=3:
        result.update(generated_input_partition='interleave',generated_split_adjoint=False,
                      generated_restrict=bool(plan.get('edge_only',False)))
        # Whole-network qualification covers frozen C256/L3 degree-packed
        # graphs. Keep standalone edge operators and unmeasured shapes intact.
        if (plan['c']==256 and plan.get('inference_degree_layout',False)
                and not plan.get('edge_only',False)
                and max(row[1] for row in specs)==7 and max(row[-1] for row in specs)==7):
            result['generated_split_input_adjoint']=True
    return result
