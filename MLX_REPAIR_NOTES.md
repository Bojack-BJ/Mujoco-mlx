# Experimental hand-contact repair

## Latest: E014 contact attribution and joint velocity repair

Fixed an MLX-only `com_vel` bug: a free/ball joint's angular derivatives must
use the same incoming velocity before accumulating the joint contribution.
Three native-oracle tests fail before and pass after the fix. Including frozen
official box-contact checks, 30/30 regression methods pass. The unchanged full
50-step LEAP gates now pass: maximum translation 0.0532 mm, orientation 0.0585°.
A 100-step extension also passes (0.0532 mm / 0.0763°), but only 0.2 s of simulated
time is covered and throughput remains 1.29 physical steps/s. No PPO replacement
or training speedup is established.

Fifteen asset-free contact cases also show that the shallow missed contact and
four-slot manifolds are shared with official MJX 3.5.0, not specific to MLX.
No collision threshold or native/JAX code was changed. See the
[E014 report](validation/contact-parity/REPORT.md) for before/after failures,
three-way reference data, fixtures, reproduction and explicit limitations.

## Previous: E013 systematic audit

**Physics runs on M5 must launch with `MLX_ENABLE_TF32=0`.** The default reduced
matrix precision corrupts mass matrices; a new native mass-matrix regression
detects this (25/25 pass in full float32, 24/25 under the default). Full-float32
paired collision improves 1.4115 → 0.1492 s (9.46×). Its 50-step trajectory still
fails: maximum position 2.286 mm and orientation 3.694°, at 1.37 physical steps/s.
Do not confuse fixing precision with solving all contact discrepancies. The
measurements in the next paragraph retain the earlier default-precision pass.

Supported plane/box collision groups now use `mx.vmap`: paired collision median
1.639 → 0.1545 s (10.61×), with eight active contacts exactly matching the old
implementation. The same historical ten-step full probe improves from 34.385 to
11.500 s (2.99×), with elementwise-identical final state. 24 repair regressions
and 23 original smoke functions pass after additional IO/equality/RK4/passive
force fixes. **A 50-step action trace still fails the preset orientation gate**
(0.02543 > 0.02 rad). This is not an accepted native/PPO replacement and has no
cross-environment batching. See the [system audit](validation/system-audit/REPORT.md)
for measurements, failures, reviewable commits and reproduction.

## Previous: E012 constraint/solver repair

Full LEAP now completes short runs, including a 10-step contact-state probe:
34.39 s total, final cube translation error 0.042 mm and orientation error
0.00254 rad (~0.146 degrees) versus native MuJoCo. Nine regression methods pass.
This is still far too slow for hand PPO and is not long-horizon validation.
The detailed [issue ledger](validation/hand-repair/ISSUES_AND_CHANGES.md), raw
attempts, environment, asset hashes and [PR draft](validation/hand-repair/PR_DRAFT.md)
are preserved for review. Entries below describe the earlier E011 checkpoint.

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
MLX_ENABLE_TF32=0 python mjx/mujoco/test_collision_optimization.py
```

Fetch the `repair-baseline` tag as well if using a shallow clone. Next work:
avoid computing inactive contact Jacobians while preserving constraint and
solver indexing; then validate full hand trajectories against native MuJoCo.
