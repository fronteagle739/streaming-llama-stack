# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
IMAGE_MODALITIES: dict[str, str] = {'t2i': 't2v', 'i2i': 'i2v', 'image_edit': 'i2v'}
DEFAULT_IMAGE_FRAMES = 22
DEFAULT_IMAGE_FPS = 24
MIN_IMAGE_FRAMES = 8
AUDIO_MODALITIES: dict[str, str] = {'t2a': 't2v', 'audio': 't2v', 'ref2a': 'ref2v'}
DEFAULT_AUDIO_SIZE = 32
DEFAULT_AUDIO_STEPS = 50

def build_video_prompt(prompt: str) -> str:
    """Keep Studio's labelled H3 captions; format a plain chat prompt in that IR."""
    text=(prompt or '').strip()
    markers=('integrated_multimodal_description','overall_soundscape','non_diegetic_music',
             'subject_definitions','retention_analysis','detailed_description')
    if any(marker in text.lower() for marker in markers):return prompt
    return (f'integrated_multimodal_description: [Shot 1] {text}\n'
            'overall_soundscape: Natural ambience and action sounds matching the described scene.\n'
            'non_diegetic_music: N/A')

def build_image_prompt(prompt: str) -> str:
    subject = (prompt or '').strip().rstrip('.')
    return f'integrated_multimodal_description: [Shot 1] The scene immediately cuts to a single still photograph of {subject}. The photograph fills the entire frame and is completely static: the camera does not move, pan, zoom or shake, and nothing within the image moves. Sharp focus, even exposure, natural colour, high detail throughout, held unchanged for the whole duration like a photograph on screen.\noverall_soundscape: Silence, with no ambience, no action sound and no human sound.\nnon_diegetic_music: N/A'

def build_edit_prompt(edit_instruction: str) -> str:
    edit = (edit_instruction or '').strip().rstrip('.')
    return f'For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] The scene immediately cuts to show the image from <Picture 1> modified so that {edit}. Everything else from <Picture 1> is preserved exactly: the same subject, framing, composition, lighting, colour palette and background. The result is a single still photograph that fills the frame and is completely static: the camera does not move, pan, zoom or shake, and nothing within the image moves. Sharp focus, even exposure, natural colour, held unchanged for the whole duration.\noverall_soundscape: Silence, with no ambience, no action sound and no human sound.\nnon_diegetic_music: N/A'

def build_audio_prompt(prompt: str) -> str:
    want = (prompt or '').strip().rstrip('.')
    return f'integrated_multimodal_description: [Shot 1] A simple, evenly lit scene that plainly shows the source of the following sound: {want}. The camera is static and the framing is close on that source, held for the whole duration.\noverall_soundscape: {want}. Clean, well recorded and prominent in the mix, present for the entire duration with a natural acoustic space and no unrelated background noise.\nnon_diegetic_music: N/A'

def build_prompt_for(modality: str, prompt: str) -> str:
    if modality in AUDIO_MODALITIES:
        return build_audio_prompt(prompt)
    if modality not in IMAGE_MODALITIES:
        raise ValueError(f'{modality!r} is not an H3 image/audio modality (expected one of {sorted(set(IMAGE_MODALITIES) | set(AUDIO_MODALITIES))})')
    return build_image_prompt(prompt) if modality == 't2i' else build_edit_prompt(prompt)

def extract_audio(video_path: str | Path, out_path: str | Path) -> Path:
    src, dst = (Path(video_path), Path(out_path))
    if not src.is_file():
        raise FileNotFoundError(f'H3 audio mode: source clip missing: {src}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    argv = [_ffmpeg(), '-v', 'error', '-i', str(src), '-vn', '-acodec', 'pcm_s16le', '-y', str(dst)]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    if r.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        raise RuntimeError(f"H3 audio mode: no audio track could be extracted from {src} ({(r.stderr or '').strip()[:300]}). The clip may have been rendered without audio.")
    return dst

def _ffmpeg() -> str:
    exe = shutil.which('ffmpeg')
    if not exe:
        raise RuntimeError('ffmpeg is required to extract the still frame but was not found on PATH')
    return exe

def extract_last_frame(video_path: str | Path, out_path: str | Path) -> Path:
    src, dst = (Path(video_path), Path(out_path))
    if not src.is_file():
        raise FileNotFoundError(f'H3 image mode: source clip missing: {src}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    ff = _ffmpeg()
    attempts = ([ff, '-v', 'error', '-sseof', '-0.2', '-i', str(src), '-update', '1', '-y', str(dst)], [ff, '-v', 'error', '-i', str(src), '-update', '1', '-y', str(dst)])
    last_err = ''
    for argv in attempts:
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=180)
            if r.returncode == 0 and dst.is_file() and (dst.stat().st_size > 0):
                return dst
            last_err = (r.stderr or '').strip()[:300]
        except Exception as exc:
            last_err = f'{type(exc).__name__}: {exc}'
    raise RuntimeError(f'H3 image mode: could not extract the final frame from {src} ({last_err})')
