"""
Utility helper functions
"""

from ..vision.cam import clamp, sign

MAX_ROTATION = 0.9

def clamp_lerp(x, a, b):
    """Clamped linear interpolation"""
    return clamp((x - a) / (b - a), 0, 1)
        
def smooth_linear(x: float, a = 674.1) -> float:
    """Smooth linear function"""
    return pow(x, 3) / (a + pow(x, 2))
    
def in_range(x, x_range) -> bool:
    """Check if x is within x_range"""
    if x_range[0] > x_range[1]:
        x_range = (x_range[1], x_range[0])
    return x_range[0] <= x and x <= x_range[1]

def out_of_bounds_function(d_back, d_front, box_width = 182.0, field_length = 183.0):
    """Calculate out of bounds function"""
    return max(0, d_back * d_front - (pow(box_width / 2, 2) + pow(field_length / 2, 2)))
