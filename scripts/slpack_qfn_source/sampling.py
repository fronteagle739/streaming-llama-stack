# Derived from Basalt (MIT); see LICENSE and UPSTREAM.json for attribution.
import math,struct

GGUF_SAMPLING = {"general.sampling.temp": "temperature", "general.sampling.top_p": "top_p",
                 "general.sampling.top_k": "top_k", "general.sampling.min_p": "min_p",
                 "general.sampling.penalty_repeat": "repetition_penalty",
                 "general.sampling.penalty_last_n": "penalty_last_n",
                 "general.sampling.penalty_present": "presence_penalty",
                 "general.sampling.penalty_freq": "frequency_penalty",
                 "general.sampling.seed": "seed"}

def _short(v):
    """A GGUF float32 as the shortest decimal that is the same float32 (0.949999988079071 -> 0.95)."""
    if not isinstance(v, float):
        return v
    bits = struct.pack("<f", v)
    for digits in range(1, 10):
        s = float(f"{v:.{digits}g}")
        if struct.pack("<f", s) == bits:
            return s
    return v

def sampling_defaults(meta: dict, source: dict | None = None) -> dict:
    """GGUF sampling defaults with any explicit legacy metadata defaults taking precedence."""

    kv = meta.get("gguf_kv") or {}
    out = {}
    aliases = {"temp": "temperature", "penalty_repeat": "repetition_penalty", "repeat_penalty": "repetition_penalty",
               "rep_pen": "repetition_penalty", "repeat_last_n": "penalty_last_n", "penalty_present": "presence_penalty",
               "penalty_freq": "frequency_penalty"}
    if source is not None:
        if not isinstance(source, dict):
            raise ValueError("checkpoint generation_config must be an object")
        for key, value in source.items():
            if key in ("eos_token_id", "bos_token_id", "pad_token_id"):
                continue  # Native authenticated reader owns token roles, independently from sampling.
            if key == "_from_model_config":
                if not isinstance(value, bool): raise ValueError("_from_model_config must be boolean")
                continue
            if key == "transformers_version":
                if not isinstance(value, str): raise ValueError("transformers_version must be text")
                continue
            if value is None: continue
            canonical = aliases.get(key, key)
            if canonical not in SAMPLING_KEYS:
                raise ValueError(f"Unsupported checkpoint generation default {key}")
            if key != canonical and source.get(canonical) is not None: continue
            if key == "rep_pen" and source.get("repetition_penalty") is not None: continue
            out[canonical] = value
    unknown = [key for key, value in kv.items() if key.startswith("general.sampling.") and key not in GGUF_SAMPLING and value is not None]
    if unknown: raise ValueError(f"Unsupported GGUF sampling {unknown}")
    out.update({field: _short(kv[key]) for key, field in GGUF_SAMPLING.items() if kv.get(key) is not None})
    legacy = meta.get("sampling") or {}
    if not isinstance(legacy, dict):
        raise ValueError("model sampling defaults must be an object")
    for key, value in legacy.items():
        if value is None:
            continue
        canonical = aliases.get(key, key)
        if canonical not in SAMPLING_KEYS and canonical != "experimental_speed_projection":
            raise ValueError(f"unsupported model sampling default {key}={value!r}")
        if canonical == "experimental_speed_projection" and not isinstance(value, bool):
            raise ValueError("model sampling.experimental_speed_projection must be true or false")
        if key != canonical and legacy.get(canonical) is not None: continue
        if key == "rep_pen" and legacy.get("repetition_penalty") is not None: continue
        out[canonical] = value
    checked_sampling(out)
    return out

def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

def _whole(v):
    return _num(v) and v == int(v)

PENALTY_WINDOW_MAX = 4096

SAMPLING_KEYS = {
    "do_sample": (bool, lambda v: isinstance(v, bool), "true or false"),
    "temperature": (float, lambda v: _num(v) and v >= 0, "a number >= 0 (0 = greedy)"),
    "top_p": (float, lambda v: _num(v) and 0 < v <= 1, "0 < top_p <= 1"),
    "min_p": (float, lambda v: _num(v) and 0 <= v <= 1, "0 <= min_p <= 1"),
    "top_k": (int, lambda v: _whole(v) and v >= -1, "an integer >= 0 (0 or -1 = off: every token)"),
    "presence_penalty": (float, _num, "a number"),
    "frequency_penalty": (float, _num, "a number"),
    "repetition_penalty": (float, lambda v: _num(v) and v > 0, "a number > 0 (1 = off)"),
    "penalty_last_n": (int, lambda v: _whole(v) and 0 <= v <= PENALTY_WINDOW_MAX,
                       f"an integer 0..{PENALTY_WINDOW_MAX} (the engine's penalty window)"),
    "seed": (int, lambda v: _whole(v) and -2 ** 63 <= v < 2 ** 64, "an integer"),
}

def checked_sampling(req: dict) -> None:
    """A request's sampling fields against SAMPLING_KEYS (the model file's rules): ValueError naming the first one out
    of range, so the request gets a 400 before streaming; a field left out or null takes the defaults.  A value is
    never changed: what passes reaches the engine as sent."""
    for key, (_, ok, expected) in SAMPLING_KEYS.items():
        value = req.get(key)
        if value is not None and not ok(value):
            raise ValueError(f"{key}: expected {expected}, not {value!r}")
    rd = req.get("relaxed_drafting")
    if rd is not None and rd is not False and not (_num(rd) and 0 <= rd <= 1):
        raise ValueError(f"relaxed_drafting: expected false or a number 0..1, not {rd!r}")
