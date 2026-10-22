# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/domainshuttle_vendor/videox_fun/utils/discrete_sampler.py
# Machine fallback paths cleared; no model data included.
import torch

class DiscreteSampling:

    def __init__(self, num_idx, uniform_sampling=False, start_num_idx=0, sp_size=1):
        self.num_idx = num_idx
        self.start_num_idx = start_num_idx
        self.uniform_sampling = uniform_sampling
        self.is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if self.is_distributed and self.uniform_sampling:
            world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
            i = 1
            while True:
                if world_size % i != 0 or num_idx % (world_size // i) != 0:
                    i += 1
                else:
                    if i >= sp_size:
                        self.group_num = world_size // i
                    elif sp_size > world_size:
                        self.group_num = 1
                    else:
                        self.group_num = world_size // sp_size
                    break
            assert self.group_num > 0
            assert world_size % self.group_num == 0
            self.group_width = world_size // self.group_num
            self.sigma_interval = self.num_idx // self.group_num
            print('rank=%d world_size=%d group_num=%d group_width=%d sigma_interval=%s' % (self.rank, world_size, self.group_num, self.group_width, self.sigma_interval))

    def __call__(self, n_samples, generator=None, device=None):
        if self.is_distributed and self.uniform_sampling:
            group_index = self.rank // self.group_width
            idx = torch.randint(self.start_num_idx + group_index * self.sigma_interval, self.start_num_idx + (group_index + 1) * self.sigma_interval, (n_samples,), generator=generator, device=device)
            print('proc[%d] idx=%s' % (self.rank, idx))
        else:
            idx = torch.randint(self.start_num_idx, self.start_num_idx + self.num_idx, (n_samples,), generator=generator, device=device)
        return idx
