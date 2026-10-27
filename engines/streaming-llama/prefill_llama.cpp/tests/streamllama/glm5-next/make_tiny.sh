#!/bin/sh
# usage: make_tiny.sh <work-dir> <upstream-src>  (upstream's gguf-py knows the glm5-next tensor names)
set -e
PYTHONPATH="$2/gguf-py" python3 "$(dirname "$0")/make_tiny_glm5next.py" "$1"
cat > "$1/models.txt" <<EOM
up-shared.gguf 1 -
up-full.gguf 1 -
ds4-full.gguf 0 up-full.gguf
EOM
