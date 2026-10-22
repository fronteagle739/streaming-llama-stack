# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/video_contact_sheet.py
# Machine fallback paths cleared; no model data included.
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from PIL import Image, ImageStat
GUTTER = 8

def check_binary(name: str) -> None:
    if not shutil_which(name):
        print(f'Error: {name} not found in PATH.', file=sys.stderr)
        sys.exit(2)

def shutil_which(name: str) -> bool:
    import shutil
    return shutil.which(name) is not None

def probe_video(video_path: str) -> dict:
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=width,height,nb_frames', '-of', 'json', video_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f'Error probing {video_path}: {e.stderr}', file=sys.stderr)
        sys.exit(3)
    data = json.loads(result.stdout)
    duration = float(data.get('format', {}).get('duration', 0))
    stream = data.get('streams', [{}])[0]
    width = int(stream.get('width', 0))
    height = int(stream.get('height', 0))
    return {'duration': duration, 'width': width, 'height': height}

def extract_frame(video_path: str, timestamp: float, width: int, out_path: str) -> bool:
    cmd = ['ffmpeg', '-v', 'error', '-ss', f'{timestamp:.6f}', '-i', video_path, '-frames:v', '1', '-vf', f'scale={width}:-1', '-y', out_path]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def compute_stats(img: Image.Image) -> tuple:
    rgb = img.convert('RGB')
    stat = ImageStat.Stat(rgb)
    mean = sum(stat.mean) / 3.0
    min_val = min(stat.extrema[0][0], stat.extrema[1][0], stat.extrema[2][0])
    max_val = max(stat.extrema[0][1], stat.extrema[1][1], stat.extrema[2][1])
    return (mean, min_val, max_val)

def tile_images(images: list, cols: int, width: int) -> Image.Image:
    if not images:
        return None
    rows = (len(images) + cols - 1) // cols
    max_h = max((img.height for img in images))
    canvas_w = cols * width + (cols - 1) * GUTTER
    canvas_h = rows * max_h + (rows - 1) * GUTTER
    canvas = Image.new('RGB', (canvas_w, canvas_h), (0, 0, 0))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        x = c * (width + GUTTER)
        y = r * (max_h + GUTTER)
        offset_y = (max_h - img.height) // 2
        canvas.paste(img, (x, y + offset_y))
    return canvas

def process_video(video_path: str, args) -> int:
    probe = probe_video(video_path)
    duration = probe['duration']
    if duration <= 0:
        print(f'Error: {video_path} has zero duration.', file=sys.stderr)
        return 3
    timestamps = [duration * (i + 0.5) / args.frames for i in range(args.frames)]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        images = []
        stats = []
        for i, ts in enumerate(timestamps):
            out_file = tmp / f'f_{i}.png'
            if not extract_frame(video_path, ts, args.width, str(out_file)):
                print(f'Warning: failed to extract frame {i} at {ts:.2f}s', file=sys.stderr)
                continue
            try:
                img = Image.open(out_file)
                img.load()
                mean, min_val, max_val = compute_stats(img)
                stats.append((i, ts, mean, min_val, max_val))
                images.append(img)
            except Exception as e:
                print(f'Warning: could not process frame {i}: {e}', file=sys.stderr)
                continue
        if not images:
            print(f'Error: no frames extracted from {video_path}', file=sys.stderr)
            return 3
        print(f'\nVideo: {video_path}')
        print(f"{'Idx':>3} {'Time(s)':>8} {'Mean':>6} {'Min':>4} {'Max':>4}  Flag")
        black_count = 0
        for idx, ts, mean, min_val, max_val in stats:
            is_black = max_val <= 2
            if is_black:
                black_count += 1
            flag = 'BLACK' if is_black else ''
            print(f'{idx:3d} {ts:8.2f} {mean:6.1f} {min_val:4d} {max_val:4d}  {flag}')
        overall_mean = sum((s[2] for s in stats)) / len(stats)
        overall_max = max((s[4] for s in stats))
        print(f'{len(stats)} frames, {black_count} black, overall mean={overall_mean:.1f}, max={overall_max}')
        if black_count == len(stats):
            verdict = 'BLACK'
        elif black_count >= len(stats) / 2:
            verdict = 'MOSTLY BLACK'
        else:
            verdict = 'CONTENT'
        print(f'VERDICT: {verdict}')
        sheet = tile_images(images, args.cols, args.width)
        if sheet is None:
            return 3
        out_path = args.out
        if out_path is None:
            stem = Path(video_path).stem
            out_path = str(Path(video_path).with_name(stem + '_contact.png'))
        sheet.save(out_path)
        print(f'Saved contact sheet to {out_path}')
    return 0

def main():
    parser = argparse.ArgumentParser(description='Create contact sheet from video(s).')
    parser.add_argument('videos', nargs='+', help='Input video files')
    parser.add_argument('--frames', type=int, default=9, help='Number of frames to extract (default: 9)')
    parser.add_argument('--cols', type=int, default=3, help='Number of columns in grid (default: 3)')
    parser.add_argument('--out', type=str, default=None, help='Output PNG path (default: input_contact.png)')
    parser.add_argument('--width', type=int, default=320, help='Tile width in pixels (default: 320)')
    args = parser.parse_args()
    check_binary('ffprobe')
    check_binary('ffmpeg')
    exit_code = 0
    for video in args.videos:
        if not os.path.isfile(video):
            print(f'Error: {video} does not exist.', file=sys.stderr)
            exit_code = 3
            continue
        rc = process_video(video, args)
        if rc != 0:
            exit_code = rc
    sys.exit(exit_code)
if __name__ == '__main__':
    main()
