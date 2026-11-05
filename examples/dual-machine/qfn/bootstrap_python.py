"""Create an independent task environment from installed offline dependencies.

Run with the required base Python. --source-site-packages is only a source of
package files during installation; the resulting environment never imports it.
No network operation or system-site-packages dependency is introduced.
"""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import venv


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--target',type=Path,required=True)
    p.add_argument('--source-site-packages',type=Path,required=True)
    p.add_argument('--package',action='append',default=[])
    a=p.parse_args()
    if a.target.exists():raise FileExistsError('Choose a new independent environment; nothing is deleted')
    selected=a.package or ['psutil','numpy']
    installed={d.metadata['Name'].lower():d for d in metadata.distributions(path=[str(a.source_site_packages)])}
    missing=set(name.lower() for name in selected)-installed.keys()
    if missing:raise ValueError('Offline distributions missing: '+str(missing))
    venv.EnvBuilder(with_pip=False,symlinks=sys.platform!='win32').create(a.target)
    site=a.target/('Lib/site-packages' if sys.platform=='win32' else f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages')
    receipt=dict(python=sys.version,base_python=sys._base_executable,packages={})
    for name in selected:
        distribution=installed[name.lower()];copied=0
        for file in distribution.files or []:
            if '..' in file.parts or str(file).endswith('.pyc'):continue
            source=Path(distribution.locate_file(file));target=site/str(file)
            if not source.is_file():raise ValueError('Offline distribution file missing: '+str(source))
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);copied+=1
        receipt['packages'][name]=dict(version=distribution.version,files=copied,source=str(a.source_site_packages))
    (a.target/'offline-install.json').write_text(json.dumps(receipt,indent=2))
    python=a.target/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    subprocess.run([str(python),'-B','-c','import sys,psutil;print(sys.prefix,psutil.__version__)'],check=True)


if __name__=='__main__':main()
