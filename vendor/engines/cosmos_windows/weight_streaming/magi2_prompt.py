# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_prompt.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
MAGI2_MODES: Tuple[str, ...] = ('t2v', 'i2v')
MAGI2_CLIP_SECONDS: float = 10.0
MAGI2_OUTPUT_FPS: float = 12.5
_NEG_VIDEO = 'Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards'
_NEG_AUDIO = ', low quality, worst quality, poor quality, noise, background noise, hiss, hum, buzz, crackle, static, compression artifacts, MP3 artifacts, digital clipping, distortion, muffled, muddy, unclear, echo, reverb, room echo, over-reverberated, hollow sound, distant, washed out, harsh, shrill, piercing, grating, tinny, thin sound, boomy, bass-heavy, flat EQ, over-compressed, abrupt cut, jarring transition, sudden silence, looping artifact, music, instrumental, sirens, alarms, crowd noise, unrelated sound effects, chaotic, disorganized, messy, cheap sound'
_NEG_VOICE = ', emotionless, flat delivery, deadpan, lifeless, apathetic, robotic, mechanical, monotone, flat intonation, undynamic, boring, reading from a script, AI voice, synthetic, text-to-speech, TTS, insincere, fake emotion, exaggerated, overly dramatic, melodramatic, cheesy, cringey, hesitant, unconfident, tired, weak voice, stuttering, stammering, mumbling, slurred speech, mispronounced, bad articulation, lisp, vocal fry, creaky voice, mouth clicks, lip smacks, wet mouth sounds, heavy breathing, audible inhales, plosives, p-pops, coughing, clearing throat, sneezing, speaking too fast, rushed, speaking too slow, dragged out, unnatural pauses, awkward silence, choppy, disconnected, multiple speakers, two voices, background talking, out of tune, off-key, autotune artifacts'
DEFAULT_NEGATIVE_PROMPT: str = _NEG_VIDEO + _NEG_AUDIO + _NEG_VOICE

@dataclass(frozen=True)
class ResolutionTier:
    name: str
    preview_width: int
    preview_height: int
    refiner_width: Optional[int] = None
    refiner_height: Optional[int] = None
    uses_refiner: bool = False
RESOLUTION_TIERS: Dict[str, ResolutionTier] = {'272p': ResolutionTier('272p', 448, 256), '540p': ResolutionTier('540p', 896, 512), '1080p': ResolutionTier('1080p', 896, 512, 1920, 1088, uses_refiner=True)}

@dataclass(frozen=True)
class Magi2Defaults:
    preview_num_inference_steps: int = 100
    refiner_num_inference_steps: int = 5
    preview_shift: float = 7.0
    refiner_shift: float = 5.0
    video_txt_guidance_scale: float = 5.0
    audio_txt_guidance_scale: float = 7.0
    refiner_video_txt_guidance_scale: float = 2.0
    refiner_audio_txt_guidance_scale: float = 5.0
    refiner_noise_value: int = 220
    fps: float = MAGI2_OUTPUT_FPS
    use_negative_prompt: bool = True
    use_turbo_vae: bool = True
    use_ref_for_uncond: bool = True
MAGI2_DEFAULTS = Magi2Defaults()

def resolution_tier(resolution: str) -> ResolutionTier:
    if resolution not in RESOLUTION_TIERS:
        raise ValueError(f'resolution must be one of {tuple(RESOLUTION_TIERS)}; got {resolution!r}')
    return RESOLUTION_TIERS[resolution]

def mode_config_defaults(mode: str, resolution: str) -> Dict[str, object]:
    if mode not in MAGI2_MODES:
        raise ValueError(f'unsupported MAGI-2 mode: {mode!r} (expected one of {MAGI2_MODES})')
    tier = resolution_tier(resolution)
    d = MAGI2_DEFAULTS
    return {'mode': mode, 'resolution': resolution, 'seconds': MAGI2_CLIP_SECONDS, 'fps': d.fps, 'preview_width': tier.preview_width, 'preview_height': tier.preview_height, 'refiner_width': tier.refiner_width, 'refiner_height': tier.refiner_height, 'uses_refiner': tier.uses_refiner, 'steps': d.preview_num_inference_steps, 'refiner_steps': d.refiner_num_inference_steps if tier.uses_refiner else None, 'preview_shift': d.preview_shift, 'refiner_shift': d.refiner_shift, 'video_txt_guidance_scale': d.video_txt_guidance_scale, 'audio_txt_guidance_scale': d.audio_txt_guidance_scale, 'refiner_video_txt_guidance_scale': d.refiner_video_txt_guidance_scale, 'refiner_audio_txt_guidance_scale': d.refiner_audio_txt_guidance_scale, 'use_negative_prompt': d.use_negative_prompt, 'use_turbo_vae': d.use_turbo_vae}

def load_prompt_text(prompt: Optional[str], prompt_file: Optional[str | Path]) -> str:
    if prompt is not None and prompt.strip():
        return prompt
    if prompt_file is not None:
        p = Path(prompt_file)
        if not p.is_file():
            raise FileNotFoundError(f'prompt_file does not exist: {p}')
        text = p.read_text(encoding='utf-8').strip()
        if not text:
            raise ValueError(f'prompt_file is empty: {p}')
        return text
    raise ValueError('a non-empty --prompt or --prompt-file is required')

def is_enhanced_caption(text: str) -> Optional[str]:
    s = text.lstrip()
    if s.startswith('{'):
        try:
            obj = json.loads(text)
        except (ValueError, TypeError):
            obj = None
        if isinstance(obj, dict):
            if 'global_layer' in obj or 'dynamic_layer' in obj:
                return 't2v'
            if 'global_caption' in obj or 'shot_timeline' in obj:
                return 'i2v'
    lowered = s.lower()
    if lowered.startswith('# global_caption') or '\n# shot_timeline' in '\n' + lowered:
        return 'i2v'
    if lowered.startswith('# global_layer') or '\n# dynamic_layer' in '\n' + lowered:
        return 't2v'
    return None
ENHANCE_SYSTEM_PROMPT_FILES: Dict[str, str] = {'t2v': 'prompts/t2v.md', 'i2v': 'prompts/i2v.md'}

@dataclass(frozen=True)
class Magi2EnhancePreset:
    mode: str
    system_prompt_file: str
    caption_schema: str
    needs_image: bool
    note: str

def magi2_enhance_presets(repo_root: Optional[Path]=None) -> Dict[str, Magi2EnhancePreset]:
    presets: Dict[str, Magi2EnhancePreset] = {}
    for mode, rel in ENHANCE_SYSTEM_PROMPT_FILES.items():
        path = str(Path(repo_root) / rel) if repo_root is not None else rel
        presets[f'magi2_{mode}'] = Magi2EnhancePreset(mode=mode, system_prompt_file=path, caption_schema='global_layer/dynamic_layer' if mode == 't2v' else 'global_caption/shot_timeline', needs_image=mode == 'i2v', note='Use as the system prompt of an instruction-following LLM; pass the raw prompt (and still, for i2v) as the message; feed the JSON caption it returns to the pipeline in place of the prompt.')
    return presets

def load_enhance_system_prompt(mode: str, repo_root: Path) -> str:
    if mode not in ENHANCE_SYSTEM_PROMPT_FILES:
        raise ValueError(f'no enhance preset for mode {mode!r}')
    p = Path(repo_root) / ENHANCE_SYSTEM_PROMPT_FILES[mode]
    if not p.is_file():
        raise FileNotFoundError(f'enhance system prompt not found: {p}')
    return p.read_text(encoding='utf-8')
