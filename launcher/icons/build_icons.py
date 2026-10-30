"""Render the launcher SVG icons and export .ico (Windows) and .icns (macOS) into icons/built/.

Uses tools already on the machine: a headless Chromium browser (Windows) or Quick Look
(`qlmanage`, macOS) rasterises each SVG once at 1024 px; Pillow masks the rounded tile,
downsamples and writes the icon containers. The packers run this when an icon is missing.

    python -B launcher/icons/build_icons.py [--browser <chrome.exe or msedge.exe>] [single dual]
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT = HERE / 'built'
CANDIDATES = (r'C:\Program Files\Google\Chrome\Application\chrome.exe',
              r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
              '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')


def render(browser, svg, temp):
    png = temp / (svg.stem + '.png')
    if browser:
        subprocess.run([browser, '--headless=new', '--disable-gpu', '--hide-scrollbars', '--no-first-run',
                        '--user-data-dir=' + str(temp / 'profile'), '--default-background-color=00000000',
                        '--window-size=1024,1024', '--screenshot=' + str(png), svg.resolve().as_uri()],
                       check=True, capture_output=True, timeout=120)
    elif sys.platform == 'darwin':
        subprocess.run(['qlmanage', '-t', '-s', '1024', '-o', str(temp), str(svg)], check=True, capture_output=True,
                       timeout=120)
        png = temp / (svg.name + '.png')
    else:
        raise SystemExit('No renderer: pass --browser <Chromium executable>')
    image = Image.open(png).convert('RGBA')
    if image.size != (1024, 1024):
        raise RuntimeError(f'{png} rendered at {image.size}, expected 1024x1024')
    # The SVG tile is the rounded square x=y=8, size 240, radius 52 in a 256 viewBox.
    mask = Image.new('L', image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((32, 32, 991, 991), radius=208, fill=255)
    image.putalpha(Image.composite(image.getchannel('A'), mask, mask))
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser', default=next((p for p in CANDIDATES if Path(p).is_file()), None))
    parser.add_argument('names', nargs='*', default=['single', 'dual'])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        for name in args.names:
            big = render(args.browser, HERE / (name + '.svg'), Path(temp))
            big.save(OUT / (name + '-1024.png'))
            big.save(OUT / (name + '.ico'), sizes=[(s, s) for s in (16, 20, 24, 32, 40, 48, 64, 128, 256)])
            big.save(OUT / (name + '.icns'))
            print('built', OUT / (name + '.ico'), OUT / (name + '.icns'))


if __name__ == '__main__':
    main()
