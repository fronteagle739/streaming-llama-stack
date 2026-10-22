# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/domainshuttle_vendor/videox_fun/models/cache_utils.py
# Machine fallback paths cleared; no model data included.
import numpy as np
import torch

def get_teacache_coefficients(model_name):
    if 'wan2.1-t2v-1.3b' in model_name.lower() or 'wan2.1-fun-1.3b' in model_name.lower() or 'wan2.1-fun-v1.1-1.3b' in model_name.lower() or ('wan2.1-vace-1.3b' in model_name.lower()):
        return [-52186.2437, 9230.41404, -528.275948, 13.6987616, -0.0499875664]
    elif 'wan2.1-t2v-14b' in model_name.lower():
        return [-303318.725, 49053.7029, -2655.30556, 58.7365115, -0.315583525]
    elif 'wan2.1-i2v-14b-480p' in model_name.lower():
        return [257151.496, -35422.9917, 1402.86849, -13.5890334, 0.132517977]
    elif 'wan2.1-i2v-14b-720p' in model_name.lower() or 'wan2.1-fun-14b' in model_name.lower() or 'wan2.2-fun' in model_name.lower() or ('wan2.2-i2v-a14b' in model_name.lower()) or ('wan2.2-t2v-a14b' in model_name.lower()) or ('wan2.2-ti2v-5b' in model_name.lower()) or ('wan2.2-s2v' in model_name.lower()) or ('wan2.1-vace-14b' in model_name.lower()) or ('wan2.2-vace-fun' in model_name.lower()) or ('wan2.2-animate' in model_name.lower()):
        return [8107.0546, 2133.93892, -372.934672, 16.6203073, -0.0417769401]
    elif 'qwen-image' in model_name.lower():
        return [-450.0, 280.0, -45.0, 3.2, -0.02]
    else:
        print(f'The model {model_name} is not supported by TeaCache.')
        return None

class TeaCache:

    def __init__(self, coefficients: list[float], num_steps: int, rel_l1_thresh: float=0.0, num_skip_start_steps: int=0, offload: bool=True):
        if num_steps < 1:
            raise ValueError(f'`num_steps` must be greater than 0 but is {num_steps}.')
        if rel_l1_thresh < 0:
            raise ValueError(f'`rel_l1_thresh` must be greater than or equal to 0 but is {rel_l1_thresh}.')
        if num_skip_start_steps < 0 or num_skip_start_steps > num_steps:
            raise ValueError(f'`num_skip_start_steps` must be great than or equal to 0 and less than or equal to `num_steps={num_steps}` but is {num_skip_start_steps}.')
        self.coefficients = coefficients
        self.num_steps = num_steps
        self.rel_l1_thresh = rel_l1_thresh
        self.num_skip_start_steps = num_skip_start_steps
        self.offload = offload
        self.rescale_func = np.poly1d(self.coefficients)
        self.cnt = 0
        self.should_calc = True
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.previous_residual_cond = None
        self.previous_residual_uncond = None

    @staticmethod
    def compute_rel_l1_distance(prev: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        rel_l1_distance = torch.abs(cur - prev).mean() / torch.abs(prev).mean()
        return rel_l1_distance.cpu().item()

    def reset(self):
        self.cnt = 0
        self.should_calc = True
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.previous_residual_cond = None
        self.previous_residual_uncond = None
