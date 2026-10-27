# Native GLM extension

Build with the serving Python on Apple Silicon. The stack setup script installs
this customised package into its isolated decode environment, then runs:

```sh
python -m mlx_vlm.custom_kernels.glm_moe_dsa.build \
  --build-dir runtime/build/glm-native --fetch-build-deps \
  --install-dir <decode-environment>/site-packages/mlx_vlm/custom_kernels/glm_moe_dsa
```

CMake and Xcode's Metal compiler must be available. The builder selects isolated
nanobind headers to match the installed MLX binary's ABI, evaluates native scoring
and top-k with real MLX arrays, and preserves previous installed artifacts before
replacement. Run setup while that decode environment is stopped.

The pure MLX implementation remains available with `MLX_GLM_DSA_NATIVE_KERNELS=0`.
An import-only check does not establish native Metal correctness or performance.
