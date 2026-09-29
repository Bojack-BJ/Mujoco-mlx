# E014: contact attribution and simultaneous joint velocity repair

Date: 2026-09-29. Starting revision: `26c814f791a4f2a7aff85ad77f479686bc6f7f72`.
This bounded follow-up isolates contact differences before changing physics.
It identifies one real **MLX-only free/ball joint velocity bug**. The previously
failing 50-step hand trace passes after that local fix. This is not an RL backend
acceptance: throughput remains approximately 1.33 physical steps/s, with no
cross-environment batching or PPO test.

Focused implementation and native-oracle tests: commit `219ec7c7`.
Diagnostics, frozen collision regressions and evidence are saved separately.

## Scope and environment

Apple M5 Pro; isolated Python 3.11.16, MuJoCo and official MJX 3.5.0,
MLX/MLX-Metal 0.32.2, JAX 0.10.2. MLX uses `Device(gpu, 0)` with
`MLX_ENABLE_TF32=0`; the independent official JAX reference uses CPU with
`jax_default_matmul_precision=highest`. Installed official MJX is imported before
exposing the local MLX tree, avoiding accidental use of the vendored JAX tree.
No main Conda environment, C/C++ MuJoCo, installed JAX source, model parameters,
contact threshold, or solver algorithm is modified.

The LEAP model/assets are the same as E013, recorded in
`../hand-repair/leap-asset-sha256.txt`. Diagnostic-only contact masking below is
not applied to the full hand trajectory or committed engine implementation.

## 1. Fifteen asset-free contact cases

`audit_contact_parity.py` verifies the native replay against E013's exact initial
qpos, control checksum and final native qpos, then extracts world-space boxes for
pairs 33:89 and 40:89 at native steps 0, 25 and 50. Nine axis-aligned cases sweep
separation and penetration through the micrometre range. The XML files in
`fixtures/` reproduce geometry without hand meshes; they do **not** preserve
original hand inertia or articulated dynamics.

`collision-three-way.json` compares native MuJoCo, installed official MJX/JAX,
MLX scalar, and two-replica MLX vmap at identical float32 JAX/MLX geometry. Native
retains its float64 XML geometry. Across all 15 cases:

- JAX versus MLX distance and contact position: exactly equal.
- Maximum normal-component difference: 5.96e-8.
- MLX scalar versus vmap: exactly equal for distance, position and normal.

Native active contacts versus the shared JAX/MLX result:

| Case | Native | JAX / MLX |
| --- | ---: | ---: |
| Step 0, pair 33:89 | 0 | 0 |
| Step 0, pair 40:89 | 2 | 4 |
| Step 25, pair 33:89 | 3 | 4 |
| Step 25, pair 40:89 | 0 | 0 |
| Step 50, pair 33:89 | 1 | 0 |
| Step 50, pair 40:89 | 2 | 4 |
| Axis penetration 0.25 / 0.5 / 0.9 micrometres | 4 | 0 |
| Axis penetration 1.1 / 2 / 10 / 1000 micrometres | 4 | 4 |

The missed LEAP contact has native distance -5.60498e-7 m. Both implementations'
box face-clipping predicate uses `>1e-6`. A worker-local ablation changes that
predicate to `>0` and recovers four active slots for the shallow cases, but does
not reproduce the native LEAP one-point manifold. The original predicate is
restored immediately; **no threshold change is landed**. This establishes shared
behavior for these cases, not a universal 1-micrometre contact cutoff or proof
that all native-versus-MJX differences come from this one predicate.

The first diagnostic attempt failed because the harness supplied unbatched
sizes to official `mesh.box`; `collision-adapter-failure.json` retains that
failure. The corrected adapter uses the official leading size batch, unbatching
only generated vertices/faces. This was a harness error, not an upstream bug.

Two new regression methods compare scalar and a mixed batch of all 15 cases
against the frozen, source-hashed official outputs. These protect port parity,
including the known native mismatch; they are not native-equivalence tests.

## 2. Preserve articulated dynamics for a three-way comparison

`audit_reduced_dynamics.py` retains all hand bodies, joints, actuators and inertia,
but limits candidate collision pairs to the two observed pairs. This reduces
the fixed contact buffer from 7016 to 8, making CPU JAX forward practical. At
each of the three native snapshots, it **first verifies** restricted versus
original native bias/passive/actuator/constraint forces and acceleration within
1e-7. This check passes; it is specific to those snapshots, not all future states.

The before/after JSON files record full arrays, error norms, solver iteration
counts, and source fingerprints. After the velocity fix, at native step 50:

| Maximum component difference | MLX vs JAX | JAX vs native |
| --- | ---: | ---: |
| Generalized constraint force | 1.2815e-5 | 0.231316 |
| Generalized acceleration | 0.0083275 | 105.175 |
| Bias force | 1.1921e-7 | 5.2643e-8 |

Generalized quantities contain different physical units across joints. These
are diagnostic array differences, not dimensionless accuracy scores. The large
late-state constraint discrepancy is shared with JAX; this does **not** prove
it caused the earlier accumulated trajectory error. In fact, the next local
fix makes that 50-step trajectory pass without changing collision behavior.

## 3. Actual local bug: free/ball angular DOFs were treated as serial joints

In `smooth.com_vel`, the old code computed one angular `cdof_dot`, added that
axis's velocity, then computed the next axis. Native MuJoCo and official MJX
instead compute all angular derivatives of one joint from the **same incoming
velocity**, then add the joint's complete velocity contribution. For a free
joint, the translation contribution is added first.

The port's sequential angular updates introduce spurious cross-axis bias forces.
An isotropic free cube spinning at [2, 3, 4] rad/s should have approximately zero
rotational bias torque; the old port gives [0.0800, -0.05333, 0.0400] N m. An
anisotropic box differs from native by up to 0.5200 N m, so simply zeroing all
rotational bias would also be wrong.

The patch computes derivatives using the common incoming velocity, then sums
the DOF contributions. Native-oracle tests cover an isotropic free body, an
anisotropic free body, and an offset ball joint below a moving hinge parent.
They compare `cvel`, `cdof_dot`, and `qfrc_bias` at nonidentity orientations.

- Before fix: the original 25 methods pass; all three new joint methods fail
  (seven assertion/subtest failures). Log: `regressions-before-velocity-fix.json`.
- After fix: 28/28 pass with unchanged tolerances.
- With two frozen official collision regressions: 30/30 pass in
  `regressions-final.json`.
- Upstream smoke: 23 passed, zero failed/skipped (`upstream-smoke.json`). Many
  of those checks are import-only, so this is not a complete physics test suite.
- At the same LEAP step-0 state, MLX-versus-JAX bias-force maximum error falls
  from 1.18789e-5 to 5.96046e-8. Remaining small-inertia acceleration differences
  are not asserted to be zero.

## 4. Full unmasked hand trajectory

The 50-step retest uses exactly the E013 initial qpos, two controls (checksum
`25eabccebcb6445cc68910e54d332b55dbcef9d56b7a368eab535196a9453e8d`), final native
reference, and unchanged error gates. At timestep 0.002 s, this is only 0.1 s of
simulated time after 100 native warmup steps.

| Maximum error over 50 physical steps | Before | After | Gate |
| --- | ---: | ---: | ---: |
| Cube translation | 2.28648 mm | 0.053209 mm | 1 mm |
| Cube orientation | 0.0644789 rad | 0.00102040 rad | 0.02 rad |
| Hand joint angle | 0.00495032 rad | 0.000244397 rad | 0.05 rad |
| Result | Fail | Pass | All gates |

The maximum orientation error is 0.058465 degrees. Full GPU-backed/hybrid
physics throughput is 1.328 steps/s; no speedup is claimed relative to E013's
1.369 steps/s because this is a correctness patch and a single, unpaired timing.
Full trajectories retain all original collisions, sensors and limits.

A bounded extension to **100 steps / four controls / 0.2 simulated seconds**
also passes: maximum translation 0.053209 mm, orientation 0.00133159 rad
(0.07629 degrees), joint error 0.000244397 rad. It completes in 79.10 s including
startup, at 1.290 physical steps/s. This is still a very short, single-seed
trajectory, not a grasping episode or learned-policy evaluation. Raw evidence:
`leap-action-50-after-velocity-fix.json` and
`leap-action-100-after-velocity-fix.json`.

## Reproduction

Use the isolated environment; substitute the same asset-hashed LEAP scene path.
The examples write to a new directory so the committed evidence is preserved.

```sh
export MLX_ENABLE_TF32=0
export JAX_PLATFORMS=cpu
python mjx/mujoco/audit_contact_parity.py --xml /path/to/cube_scene.xml \
  --trajectory-json validation/system-audit/leap-action-50-fp32.json \
  --timeout 150 --output /tmp/contact-parity-new/collision.json
python mjx/mujoco/audit_reduced_dynamics.py --xml /path/to/cube_scene.xml \
  --snapshots /tmp/contact-parity-new/collision.json \
  --timeout 180 --output /tmp/contact-parity-new/dynamics.json
python mjx/mujoco/run_hand_regressions.py --output /tmp/contact-parity-new/tests.json
python mjx/mujoco/benchmark_hand_system.py --xml /path/to/cube_scene.xml \
  --backend compare --native-warmup 100 --steps 50 --substeps 25 --action-seed 0 \
  --timeout 180 --output /tmp/contact-parity-new/trajectory.json
```

`revision` fields identify the starting commit; these measurements precede the
commits of this follow-up. Source hashes and the separately reviewable patch
identify tested changes. Personal paths in shared JSON are replaced with portable
placeholders; numerical arrays, test assertions and control hashes are unchanged.
The before dynamics file uses the clean starting engine; the after file includes
the new `smooth.py` hash. Full history and `repair-baseline` remain necessary for
older scalar-reference regression modules.

`source-sha256.txt` fingerprints tested sources. `pre-redaction-json-sha256.txt`
records the original local JSON bytes before path substitution, not the hashes
of the portable committed reports. They are retained for provenance only.

## Decision boundary

This round yields a small, independently reproducible PR-worthy correction.
It does not justify rewriting collision/solver algorithms or committing the
zero-threshold experiment. Passing a short trajectory does not establish
long-horizon fidelity, policy quality, multi-environment throughput, gradients,
or PPO speedup. The main training backend remains native MuJoCo; future NVIDIA
batch evaluation stays a separate task. The third-party fork is retained as an
experimental backend, not promoted into the main environment.
