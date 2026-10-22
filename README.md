# ds4-stack

Private glue repository for pinned ds4-win, RTX streaming llama and Studio engines.
Work in progress; no model execution or end-to-end qualification is claimed.

```powershell
git clone --branch agent/ds4stack-20261022 --recurse-submodules https://github.com/fronteagle739/streaming-llama-stack-clean.git
cd ds4-stack
python -m pip install -r requirements.txt
Copy-Item config.example.json config.local.json
# Set model paths; all runtime/build/cache writes on this RTX stay under runtime/ds4stack.
./ds4stack.ps1 --config config.local.json check
./build.ps1 -Config config.local.json
```

Source pins and provenance are in `.gitmodules`, gitlinks and `ds4stack/check.py`.
Studio is a pinned submodule, not a vendored source copy. No GitHub workflows.
The llama build uses its own CMake targets; ds4 uses its own Windows build script.
`check` fails loudly for missing models, builds, pins, live bias service or unsafe journal state.
Every engine process requires an explicit RSS budget, a physical-memory floor, a
Windows Job private-commit ceiling and an independent process-tree guardian.

The dependency result files explicitly defer live bias MCP control and the
Windows/CUDA ULTRA state adapter. Their interfaces must not be mistaken for
implemented engine features. See `docs/STATUS.md` as integration proceeds.
