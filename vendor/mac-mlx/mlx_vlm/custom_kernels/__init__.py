# SPDX-License-Identifier: Apache-2.0
"""Optional native custom kernels bundled with mlx-vlm.

Currently ships the GLM-5.2 ``glm_moe_dsa`` sparse-MLA / DSA Metal kernels
vendored from oMLX (github.com/jundot/omlx PR #1984). The kernels are optional:
without a native build the model runs on the pure-MLX path unchanged.
"""
