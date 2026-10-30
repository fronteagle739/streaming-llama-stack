"""Render the launcher SVG icons and export .ico (Windows) and .icns (macOS).

Uses tools already on the machine: a Chromium browser in headless mode rasterises the
SVG once at 1024 px, Pillow downsamples and writes the icon containers.

    python -B launcher/icons/build_icons.py [--browser <chrome.exe or msedge.exe>]
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / 'built'
CANDIDATES = (r'C:\Program Files\Google\Chrome\Application\chrome.exe',
              r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
              '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')


def render(browser, svg, png, profile):
    subprocess.run([browser, '--headless=new', '--disable-gpu', '--hide-scrollbars', '--no-first-run',
                    '--user-data-dir=' + str(profile), '--default-background-color=00000000',
                    '--window-size=1024,1024', '--screenshot=' + str(png), svg.resolve().as_uri()],
                   check=True, capture_output=True, timeout=120)
    image = Image.open(png).convert('RGBA')
    if image.size != (1024, 1024):
        raise RuntimeError(f'{png} rendered at {image.size}, expected 1024x1024')
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser', default=next((p for p in CANDIDATES if Path(p).is_file()), None))
    args = parser.parse_args()
    if not args.browser:
        raise SystemExit('No Chromium browser found; pass --browser')
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        for name in ('single', 'dual'):
            big = render(args.browser, HERE / (name + '.svg'), Path(temp) / (name + '.png'), Path(temp) / 'profile')
            big.save(OUT / (name + '-1024.png'))
            for size in (16, 32, 256):
                big.resize((size, size), Image.LANCZOS).save(OUT / f'{name}-{size}.png')
            big.save(OUT / (name + '.ico'), sizes=[(s, s) for s in (16, 20, 24, 32, 40, 48, 64, 128, 256)])
            big.save(OUT / (name + '.icns'))
            print('built', OUT / (name + '.ico'), OUT / (name + '.icns'))
        shutil.rmtree(Path(temp) / 'profile', ignore_errors=True)


if __name__ == '__main__':
    main()
