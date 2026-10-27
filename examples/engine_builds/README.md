# Building native engines

`slstack install --all` builds Windows CUDA/native targets, streaming targets, producer-profile checks and the chat UI. On macOS it builds the native receiver. Python 3.12, CMake/Ninja and the platform C++ compiler are required; Windows also needs the CUDA toolkit and Node.js/npm for UI assets.

`install --all --check` checks packages and outputs without rebuilding. The source includes the CUDA provider headers needed for standalone builds. UI assets build before the server embeds them.

`RESULTS.json` records measured build durations, binary identities and resource usage. Model-output comparisons are described in [engine parity](../engine_parity/README.md).
