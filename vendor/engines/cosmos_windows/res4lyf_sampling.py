# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/res4lyf_sampling.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import math
from typing import Iterable
RES4LYF_SAMPLERS: list[str] = ['none', 'multistep/res_2m', 'multistep/res_3m', 'multistep/dpmpp_2m', 'multistep/dpmpp_3m', 'multistep/abnorsett_2m', 'multistep/abnorsett_3m', 'multistep/abnorsett_4m', 'multistep/deis_2m', 'multistep/deis_3m', 'multistep/deis_4m', 'exponential/res_2s_rkmk2e', 'exponential/res_2s', 'exponential/res_2s_stable', 'exponential/res_3s', 'exponential/res_3s_non-monotonic', 'exponential/res_3s_alt', 'exponential/res_3s_cox_matthews', 'exponential/res_3s_lie', 'exponential/res_3s_sunstar', 'exponential/res_3s_strehmel_weiner', 'exponential/res_4s_krogstad', 'exponential/res_4s_krogstad_alt', 'exponential/res_4s_strehmel_weiner', 'exponential/res_4s_strehmel_weiner_alt', 'exponential/res_4s_cox_matthews', 'exponential/res_4s_cfree4', 'exponential/res_4s_friedli', 'exponential/res_4s_minchev', 'exponential/res_4s_munthe-kaas', 'exponential/res_5s', 'exponential/res_5s_hochbruck-ostermann', 'exponential/res_6s', 'exponential/res_8s', 'exponential/res_8s_alt', 'exponential/res_10s', 'exponential/res_15s', 'exponential/res_16s', 'exponential/etdrk2_2s', 'exponential/etdrk3_a_3s', 'exponential/etdrk3_b_3s', 'exponential/etdrk4_4s', 'exponential/etdrk4_4s_alt', 'exponential/dpmpp_2s', 'exponential/dpmpp_sde_2s', 'exponential/dpmpp_3s', 'exponential/lawson2a_2s', 'exponential/lawson2b_2s', 'exponential/lawson4_4s', 'exponential/lawson41-gen_4s', 'exponential/lawson41-gen-mod_4s', 'exponential/ddim', 'hybrid/pec423_2h2s', 'hybrid/pec433_2h3s', 'hybrid/abnorsett2_1h2s', 'hybrid/abnorsett3_2h2s', 'hybrid/abnorsett4_3h2s', 'hybrid/lawson42-gen-mod_1h4s', 'hybrid/lawson43-gen-mod_2h4s', 'hybrid/lawson44-gen-mod_3h4s', 'hybrid/lawson45-gen-mod_4h4s', 'linear/ralston_2s', 'linear/ralston_3s', 'linear/ralston_4s', 'linear/midpoint_2s', 'linear/heun_2s', 'linear/heun_3s', 'linear/houwen-wray_3s', 'linear/kutta_3s', 'linear/ssprk3_3s', 'linear/ssprk4_4s', 'linear/rk38_4s', 'linear/rk4_4s', 'linear/rk5_7s', 'linear/rk6_7s', 'linear/bogacki-shampine_4s', 'linear/bogacki-shampine_7s', 'linear/dormand-prince_6s', 'linear/dormand-prince_13s', 'linear/tsi_7s', 'linear/euler', 'diag_implicit/irk_exp_diag_2s', 'diag_implicit/kraaijevanger_spijker_2s', 'diag_implicit/qin_zhang_2s', 'diag_implicit/pareschi_russo_2s', 'diag_implicit/pareschi_russo_alt_2s', 'diag_implicit/crouzeix_2s', 'diag_implicit/crouzeix_3s', 'diag_implicit/crouzeix_3s_alt', 'fully_implicit/gauss-legendre_2s', 'fully_implicit/gauss-legendre_3s', 'fully_implicit/gauss-legendre_4s', 'fully_implicit/gauss-legendre_4s_alternating_a', 'fully_implicit/gauss-legendre_4s_ascending_a', 'fully_implicit/gauss-legendre_4s_alt', 'fully_implicit/gauss-legendre_5s', 'fully_implicit/gauss-legendre_5s_ascending', 'fully_implicit/radau_ia_2s', 'fully_implicit/radau_ia_3s', 'fully_implicit/radau_iia_2s', 'fully_implicit/radau_iia_3s', 'fully_implicit/radau_iia_3s_alt', 'fully_implicit/radau_iia_5s', 'fully_implicit/radau_iia_7s', 'fully_implicit/radau_iia_9s', 'fully_implicit/radau_iia_11s', 'fully_implicit/lobatto_iiia_2s', 'fully_implicit/lobatto_iiia_3s', 'fully_implicit/lobatto_iiia_4s', 'fully_implicit/lobatto_iiib_2s', 'fully_implicit/lobatto_iiib_3s', 'fully_implicit/lobatto_iiib_4s', 'fully_implicit/lobatto_iiic_2s', 'fully_implicit/lobatto_iiic_3s', 'fully_implicit/lobatto_iiic_4s', 'fully_implicit/lobatto_iiic_star_2s', 'fully_implicit/lobatto_iiic_star_3s', 'fully_implicit/lobatto_iiid_2s', 'fully_implicit/lobatto_iiid_3s']
RES4LYF_EXTRA_SAMPLERS: list[str] = ['er_sde', 'res_2m', 'res_3m', 'res_2s', 'res_3s', 'res_5s', 'res_6s', 'res_2m_ode', 'res_3m_ode', 'res_2s_ode', 'res_3s_ode', 'res_5s_ode', 'res_6s_ode', 'res_2m_sde', 'res_3m_sde', 'res_2s_sde', 'res_3s_sde']
RES4LYF_SAMPLERS = []
RES4LYF_EXTRA_SAMPLERS = ['er_sde']
RES4LYF_CUSTOM_SIGMAS: list[str] = ['beta', 'beta57', 'bong_tangent']
RES4LYF_SIGMAS: list[str] = [*RES4LYF_CUSTOM_SIGMAS, 'native_beta']
RES4LYF_SCHEDULERS: list[str] = RES4LYF_SIGMAS
_RES4LYF_NAME_SET = {name.lower() for name in RES4LYF_EXTRA_SAMPLERS}

def res4lyf_sampler_options(*, include_plain_aliases: bool=True) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for name in RES4LYF_EXTRA_SAMPLERS:
        if name == 'none':
            continue
        if name.lower() not in seen:
            out.append(name)
            seen.add(name.lower())
    return out

def _key(value: str | None) -> str:
    return (value or '').strip().lower().replace('-', '_')

def sampler_base(value: str | None) -> str:
    key = _key(value)
    return key.rsplit('/', 1)[-1]

def is_res4lyf_sampler(value: str | None) -> bool:
    key = _key(value)
    if not key:
        return False
    return key in _RES4LYF_NAME_SET or sampler_base(key) in _RES4LYF_NAME_SET

def res4lyf_stage_count(value: str | None) -> int:
    return 1

def is_res4lyf_sigma(value: str | None) -> bool:
    return _key(value) in {name.lower() for name in RES4LYF_SIGMAS}

def is_custom_flow_sigma(value: str | None) -> bool:
    return _key(value) in {name.lower() for name in RES4LYF_CUSTOM_SIGMAS}

def _res4lyf_bong_tangent_raw(steps: int, *, start: float=1.0, middle: float=0.5, end: float=0.0, pivot_1: float=0.6, pivot_2: float=0.6, slope_1: float=0.2, slope_2: float=0.2) -> list[float]:

    def section(n: int, slope: float, pivot: float, section_start: float, section_end: float) -> list[float]:
        n = max(2, int(n))
        smax = (2 / math.pi * math.atan(-slope * (0 - pivot)) + 1) / 2
        smin = (2 / math.pi * math.atan(-slope * (n - 1 - pivot)) + 1) / 2
        srange = smax - smin
        if abs(srange) < 1e-09:
            return [section_start for _ in range(n)]
        scale = section_start - section_end
        return [((2 / math.pi * math.atan(-slope * (x - pivot)) + 1) / 2 - smin) * (1 / srange) * scale + section_end for x in range(n)]
    total = max(1, int(steps)) + 2
    midpoint = int((total * pivot_1 + total * pivot_2) / 2)
    pivot_1_i = int(total * pivot_1)
    pivot_2_i = int(total * pivot_2)
    slope_1 = slope_1 / (total / 40)
    slope_2 = slope_2 / (total / 40)
    stage_2_len = total - midpoint
    stage_1_len = total - stage_2_len
    first = section(stage_1_len, slope_1, pivot_1_i, start, middle)[:-1]
    second = section(stage_2_len, slope_2, pivot_2_i - stage_1_len, middle, end)
    return first + second

def bong_tangent_sigmas(steps: int) -> list[float]:
    n = max(1, int(steps))
    raw = _res4lyf_bong_tangent_raw(n)
    if len(raw) > n:
        raw = raw[:n]
    if len(raw) < n:
        raw.extend([raw[-1] if raw else 1.0] * (n - len(raw)))
    return [float(max(0.0, min(1.0, x))) for x in raw]

def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3e-14
    fp_min = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fp_min:
        d = fp_min
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fp_min:
            d = fp_min
        c = 1.0 + aa / c
        if abs(c) < fp_min:
            c = fp_min
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fp_min:
            d = fp_min
        c = 1.0 + aa / c
        if abs(c) < fp_min:
            c = fp_min
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h

def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _beta_continued_fraction(a, b, x) / a
    return 1.0 - bt * _beta_continued_fraction(b, a, 1.0 - x) / b

def _beta_ppf(p: float, a: float, b: float) -> float:
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo = 0.0
    hi = 1.0
    for _ in range(80):
        mid = (lo + hi) * 0.5
        if _regularized_incomplete_beta(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5

def comfy_beta_sigmas(steps: int, *, alpha: float=0.6, beta: float=0.6, table_size: int=1000) -> list[float]:
    n = max(1, int(steps))
    total = max(1, int(table_size))
    out: list[float] = []
    last_idx: int | None = None
    for i in range(n):
        p = 1.0 - i / float(n)
        q = _beta_ppf(p, float(alpha), float(beta))
        idx = int(round(max(0.0, min(1.0, q)) * total))
        idx = max(0, min(total, idx))
        if idx != last_idx:
            out.append(float(idx) / float(total))
        last_idx = idx
    return out or [1.0]

def beta_sigmas(steps: int) -> list[float]:
    return comfy_beta_sigmas(steps, alpha=0.6, beta=0.6)

def beta57_sigmas(steps: int) -> list[float]:
    return comfy_beta_sigmas(steps, alpha=0.5, beta=0.7)

def custom_flow_sigmas(value: str | None, steps: int) -> list[float] | None:
    key = _key(value)
    if key in {'', 'default', 'flow', 'normal', 'simple', 'linspace', 'linear'}:
        return None
    if key == 'beta':
        return beta_sigmas(steps)
    if key == 'bong_tangent':
        return bong_tangent_sigmas(steps)
    if key == 'beta57':
        return beta57_sigmas(steps)
    return None

def _clean_flow_config(base: dict) -> dict:
    out = dict(base or {})
    for key in ('use_karras_sigmas', 'use_exponential_sigmas', 'use_beta_sigmas'):
        if key in out:
            out[key] = False
    return out

def _from_config(cls, base: dict, **kwargs):
    try:
        return cls.from_config(base, **kwargs)
    except Exception:
        cfg = dict(base)
        cfg.update(kwargs)
        try:
            return cls(**cfg)
        except Exception:
            return cls.from_config(base)

def _flow_kwargs_for_sigmas(sigmas: str | None) -> dict:
    key = _key(sigmas)
    if key == 'karras':
        return {'use_karras_sigmas': True}
    if key == 'exponential':
        return {'use_exponential_sigmas': True}
    if key in {'native_beta', 'diffusers_beta'}:
        return {'use_beta_sigmas': True}
    return {}

def _is_removed_res4lyf_sampler(value: str | None) -> bool:
    key = _key(value)
    base = sampler_base(value)
    if not key:
        return False
    return '/' in key or base.startswith('res_') or base.startswith('rk_')

def make_flow_scheduler(base_config: dict, *, sampler: str | None=None, sigmas: str | None=None, shift: float | None=None, log_prefix: str='[RES4LYF]'):
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    try:
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput
    except Exception:
        FlowMatchEulerDiscreteSchedulerOutput = None
    try:
        from diffusers.utils.torch_utils import randn_tensor
    except Exception:
        randn_tensor = None
    base = _clean_flow_config(base_config)
    if shift is not None:
        if 'shift' in base:
            base['shift'] = float(shift)
        else:
            base['shift'] = float(shift)
    sampler_key = sampler_base(sampler)
    use_custom = is_custom_flow_sigma(sigmas)

    class ComfyFlowERSDEScheduler(FlowMatchEulerDiscreteScheduler):

        def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None):
            result = super().set_timesteps(num_inference_steps=num_inference_steps, device=device, sigmas=sigmas, mu=mu, timesteps=timesteps)
            self._er_old_denoised = None
            self._er_old_denoised_d = None
            self._er_point_indices = None
            return result

        def _er_lambdas(self, device, dtype):
            sigmas = self.sigmas.to(device=device, dtype=torch.float32)
            if sigmas.numel() > 0 and sigmas[0] >= 1.0:
                sigmas = sigmas.clone()
                sigmas[0] = 1.0 - 0.0001
            sigmas = sigmas.clamp(0.0, 1.0 - 1e-07)
            lambdas = sigmas / (1.0 - sigmas)
            return lambdas.to(dtype=dtype)

        @staticmethod
        def _noise_scaler(x):
            return x * ((x ** 0.3).exp() + 10.0)

        def _noise(self, sample, generator):
            if randn_tensor is not None:
                try:
                    return randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
                except Exception:
                    pass
            try:
                return torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            except Exception:
                return torch.randn_like(sample)

        def step(self, model_output, timestep, sample, s_churn: float=0.0, s_tmin: float=0.0, s_tmax: float=float('inf'), s_noise: float=1.0, generator=None, per_token_timesteps=None, return_dict: bool=True):
            if per_token_timesteps is not None:
                raise NotImplementedError('er_sde does not support per-token timesteps yet')
            if isinstance(timestep, int) or isinstance(timestep, torch.IntTensor) or isinstance(timestep, torch.LongTensor):
                raise ValueError('Pass a scheduler timestep value to er_sde.step(), not an integer step index')
            if self.step_index is None:
                self._init_step_index(timestep)
            sigma_idx = int(self.step_index)
            sigma = self.sigmas[sigma_idx].to(device=sample.device, dtype=torch.float32)
            sigma_next = self.sigmas[sigma_idx + 1].to(device=sample.device, dtype=torch.float32)
            sample_f = sample.to(torch.float32)
            model_output_f = model_output.to(torch.float32)
            denoised = sample_f - sigma * model_output_f
            if sigma_next == 0:
                prev_sample = denoised
            else:
                er_lambdas = self._er_lambdas(sample.device, torch.float32)
                er_lambda_s = er_lambdas[sigma_idx]
                er_lambda_t = er_lambdas[sigma_idx + 1]
                alpha_s = sigma / er_lambda_s
                alpha_t = sigma_next / er_lambda_t
                r_alpha = alpha_t / alpha_s
                r = self._noise_scaler(er_lambda_t) / self._noise_scaler(er_lambda_s)
                prev_sample = r_alpha * r * sample_f + alpha_t * (1.0 - r) * denoised
                stage_used = min(3, sigma_idx + 1)
                if stage_used >= 2 and self._er_old_denoised is not None:
                    num_integration_points = 200
                    if self._er_point_indices is None or self._er_point_indices.device != sample.device:
                        self._er_point_indices = torch.arange(0, num_integration_points, dtype=torch.float32, device=sample.device)
                    dt = er_lambda_t - er_lambda_s
                    lambda_step_size = -dt / float(num_integration_points)
                    lambda_pos = er_lambda_t + self._er_point_indices * lambda_step_size
                    scaled_pos = self._noise_scaler(lambda_pos)
                    s = torch.sum(1.0 / scaled_pos) * lambda_step_size
                    denoised_d = (denoised - self._er_old_denoised) / (er_lambda_s - er_lambdas[sigma_idx - 1])
                    prev_sample = prev_sample + alpha_t * (dt + s * self._noise_scaler(er_lambda_t)) * denoised_d
                    if stage_used >= 3 and self._er_old_denoised_d is not None:
                        s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                        denom = (er_lambda_s - er_lambdas[sigma_idx - 2]) / 2.0
                        denoised_u = (denoised_d - self._er_old_denoised_d) / denom
                        prev_sample = prev_sample + alpha_t * (dt ** 2 / 2.0 + s_u * self._noise_scaler(er_lambda_t)) * denoised_u
                    self._er_old_denoised_d = denoised_d
                if s_noise > 0:
                    variance = (er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2).clamp_min(0.0).sqrt().nan_to_num(0.0)
                    prev_sample = prev_sample + alpha_t * self._noise(sample_f, generator) * float(s_noise) * variance
            self._er_old_denoised = denoised
            self._step_index += 1
            prev_sample = prev_sample.to(model_output.dtype)
            if not return_dict:
                return (prev_sample,)
            if FlowMatchEulerDiscreteSchedulerOutput is None:
                return (prev_sample,)
            return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
    if _is_removed_res4lyf_sampler(sampler):
        raise ValueError(f'sampler={sampler!r} is not implemented in this backend')
    if sampler_key == 'er_sde':
        scheduler = _from_config(ComfyFlowERSDEScheduler, base, **_flow_kwargs_for_sigmas(sigmas))
        print(f'{log_prefix} sampler=er_sde (Comfy ER-SDE flow solver)', flush=True)
    else:
        kwargs = _flow_kwargs_for_sigmas(sigmas)
        scheduler = _from_config(FlowMatchEulerDiscreteScheduler, base, **kwargs)
    return (scheduler, [] if use_custom else None)

def configure_diffusers_flow_scheduler(pipe, *, sampler: str | None=None, sigmas: str | None=None, num_inference_steps: int=8, shift: float | None=None, log_prefix: str='[RES4LYF]') -> list[float] | None:
    base = dict(getattr(getattr(pipe, 'scheduler', None), 'config', {}) or {})
    scheduler, custom = make_flow_scheduler(base, sampler=sampler, sigmas=sigmas, shift=shift, log_prefix=log_prefix)
    pipe.scheduler = scheduler
    custom = custom_flow_sigmas(sigmas, int(num_inference_steps or 1))
    if custom is not None:
        print(f'{log_prefix} sigmas={sigmas} ({len(custom)} custom flow sigmas)', flush=True)
    return custom

def flow_scheduler_for_manual_loop(base_scheduler, *, sampler: str | None=None, sigmas: str | None=None, num_inference_steps: int=8, shift: float | None=None, log_prefix: str='[RES4LYF]'):
    base = dict(getattr(base_scheduler, 'config', {}) or {})
    scheduler, _ = make_flow_scheduler(base, sampler=sampler, sigmas=sigmas, shift=shift, log_prefix=log_prefix)
    custom = custom_flow_sigmas(sigmas, int(num_inference_steps or 1))
    if custom is not None:
        print(f'{log_prefix} sigmas={sigmas} ({len(custom)} custom flow sigmas)', flush=True)
    return (scheduler, custom)

def extend_unique(values: Iterable[str], additions: Iterable[str]) -> list[str]:
    out = list(values)
    seen = {str(v).lower() for v in out}
    for value in additions:
        key = str(value).lower()
        if key not in seen:
            out.append(value)
            seen.add(key)
    return out
