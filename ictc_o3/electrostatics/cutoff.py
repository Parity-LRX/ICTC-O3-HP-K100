"""Scalar geometric cutoff convention."""
import math

def derive_long_range_far_max_radius_multiplier(
    far_num_shells: int,
    far_shell_growth: float,
) -> float:
    far_num_shells = int(far_num_shells)
    far_shell_growth = float(far_shell_growth)
    if far_num_shells < 1:
        raise ValueError("long_range_far_num_shells must be >= 1")
    if far_shell_growth <= 1.0:
        raise ValueError("long_range_far_shell_growth must be > 1")
    return float(math.pow(far_shell_growth, far_num_shells))

