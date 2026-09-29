# Experimental hand-contact repair

Based on RobotFlow-Labs/Mujoco-mlx commit
`5de472e576e5233d159c1a2c46bb4f70a75c2db4`.
Changes are confined to the Python MLX port; MuJoCo C/C++ and official JAX MJX
are unchanged. This branch is **not ready for hand RL training**.

## Compatibility baseline (`repair-baseline` tag)

- Restore collision dispatch for flattened MLX data.
- Preserve convex mesh data during model conversion.
- Convert NumPy indices/scalars and convex inputs where MLX requires arrays.
- Correct batched constraint parameter shapes and quaternion integration.

## Collision optimization

- Conservative bounding-sphere rejection for box/mesh pairs. Keep all allocated
  contact slots and fill rejected pairs with inactive distances, zero positions,
  and identity frames. Positive distances for rejected pairs are sentinels, not
  exact geometric distances. Plane and heightfield pairs are not culled.
- Batch Gauss-map edge-pair normal gathering, arc intersection and edge-axis
  computations rather than constructing tens of thousands of scalar operations.
- The broadphase currently synchronizes positions to NumPy: it is an eager-mode
  optimization, not an end-to-end GPU, differentiable or compiled batch backend.

## Validation (2026-09-29, Apple M5 Pro)

Isolated Python 3.11 environment: MuJoCo/MJX 3.5.0, MLX/MLX-Metal 0.32.2.
The main project Conda environment was not modified by this change.

- LEAP initial state: 1754 candidate pairs, 7016 allocated contact slots.
- Before optimization: collision did not finish within 30 seconds; a sampled
  144-by-144 mesh-edge case hit Metal's 499000-resource limit.
- After optimization: complete initial collision phase ~1.54 seconds with
  diagnostic instrumentation. This is not an end-to-end simulation speedup.
- Full LEAP step: still fails, now during elliptic contact constraint assembly
  (`support.jac`) with the same Metal resource limit, ~11.23 seconds into the
  probe. Zero complete hand steps; no training or batch-performance claim.
- Tilted/sliding cube, 100 steps: finite states, ~9.90 seconds, final position
  maximum error versus native MuJoCo ~1.62 mm. Mixed-unit qpos trajectory maximum
  difference ~0.0614; physical equivalence is not established. The pre-optimization
  baseline produces exactly the same final qpos and maximum trajectory error.
- Regression tests compare general convex SAT against the preserved scalar
  implementation and check active box contacts plus inactive slot preservation.
  Coverage is deliberately small, not exhaustive collision certification.

Run the regression from the repository root with the isolated environment:

```sh
python mjx/mujoco/test_collision_optimization.py
```

Fetch the `repair-baseline` tag as well if using a shallow clone. Next work:
avoid computing inactive contact Jacobians while preserving constraint and
solver indexing; then validate full hand trajectories against native MuJoCo.
