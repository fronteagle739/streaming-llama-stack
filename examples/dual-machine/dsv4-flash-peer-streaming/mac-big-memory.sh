#!/bin/sh
set -eu
stack_root=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
if [ -n "${SLSTACK_PYTHON:-}" ]; then
    exec "$SLSTACK_PYTHON" -B "$stack_root/examples/run_preset.py" --preset dsv4-flash-peer-streaming --machine mac "$@"
fi
for stack_python in python3.12 python3.13 python3.14 python3; do
    if command -v "$stack_python" >/dev/null 2>&1 && "$stack_python" -c 'import sys; sys.exit(sys.version_info < (3,12))' 2>/dev/null; then
        exec "$stack_python" -B "$stack_root/examples/run_preset.py" --preset dsv4-flash-peer-streaming --machine mac "$@"
    fi
done
echo 'Install Python 3.12+ or set SLSTACK_PYTHON.' >&2
exit 1
