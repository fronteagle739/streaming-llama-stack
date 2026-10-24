"""Pass an existing guarded command its bounded stdin without buffering stdout."""
import json
from pathlib import Path
import subprocess
import sys

def main():
    from slstack.envvars import restore_secrets
    spec = restore_secrets(json.loads(Path(sys.argv[1]).read_text()))
    child = subprocess.Popen(spec['command'], cwd=spec['cwd'],env=spec['env'],
                             stdin=subprocess.PIPE if 'stdin' in spec else subprocess.DEVNULL)
    child.communicate(spec['stdin'].encode() if 'stdin' in spec else None)
    return child.returncode


if __name__ == '__main__':
    sys.exit(main())
