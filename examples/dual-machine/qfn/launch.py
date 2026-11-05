"""Select the Q4 or Q8 deployment, preserving the complete native preset."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--quant',choices=['q4','q8'],required=True)
    p.add_argument('--config',type=Path,required=True)
    a,extra=p.parse_known_args()
    cfg=json.loads(a.config.read_text(encoding='utf8'))
    if set(cfg)!={'common','formats'} or set(cfg['formats'])!={'q4','q8'}:
        raise ValueError('Deployment requires common settings and both q4/q8 formats')
    common=dict(cfg['common']);selected=cfg['formats'][a.quant]
    if (not {'mac-model','mac-receipt'}<=set(selected) or
            set(selected)-{'mac-model','mac-receipt','profile'} or set(common)&set(selected)):
        raise ValueError('Format entries select the model, verified receipt and optional complete profile without overriding common flags')
    settings={**common,**selected}
    settings.setdefault('profile',str(Path(__file__).with_name('native.q4.settings.json' if a.quant=='q4' else 'native.settings.json')))
    missing = [key for key,value in settings.items() if isinstance(value,str) and value.startswith(('SET_', 'REPLACE_'))]
    if missing:raise ValueError('Fill the deployment inputs before launching: ' + ', '.join(sorted(missing)))
    profile=Path(settings['profile'])
    if not profile.is_absolute():settings['profile']=str(Path(__file__).resolve().parents[3]/profile)
    command=[sys.executable,'-B',str(Path(__file__).with_name('native_pair.py'))]
    for key,value in settings.items():
        if not isinstance(value,(str,int,float)) or isinstance(value,bool):
            raise ValueError('Deployment values are explicit native-pair flag values: '+key)
        command+=['--'+key,str(value)]
    return subprocess.call([*command,*extra])


if __name__=='__main__':raise SystemExit(main())
