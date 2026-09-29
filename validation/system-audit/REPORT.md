# E013: systematic correctness and performance audit

Date: 2026-09-29. Decision: **improved, but not accepted as a native MuJoCo / PPO replacement**.

Latest acceptance mode: launch a fresh process with **`MLX_ENABLE_TF32=0`**.
25/25 repair methods pass in that mode. Paired collision speedup is **9.46×**;
the 50-step action trace runs at **1.369 physical steps/s** but still fails both
translation (2.286 mm > 1 mm) and orientation (0.06448 rad > 0.02 rad) gates.
Earlier default-precision evidence below is retained and explicitly separated.

## Environment and scope

Apple M5 Pro, macOS 26.5.1; isolated Python 3.11.16, MuJoCo/MJX 3.5.0,
MLX/MLX-Metal 0.32.2. GPU runs explicitly use `Device(gpu, 0)`.
The main `mujoco-hand` environment, MuJoCo C/C++ and official JAX MJX were not
modified. These are changes to this third-party Python MLX port.

Starting revision: `10c4fa4a7653c4efc7891c8083809b6fae35eca3`.
Reviewable source commits:

- `6691e627`: supported convex collision batching and scalar readback removal.
- `f9dd2646`: empty elliptic constraints, equalities, RK4 and passive forces.
- `9ac02b50`: public state conversion and constraint row identity.
- `9cf56bf2`: bounded benchmark and same-state diagnostic scripts.
- `df8afecb`: native mass precision guard and precision-policy metadata.

The LEAP assets are unchanged; use the asset manifest in
`../hand-repair/leap-asset-sha256.txt`. Benchmarks record the top-level XML hash;
that field alone does not cover includes/meshes. `source-sha256.txt` identifies
the final tested local source, including diagnostic scripts. Early benchmark
JSON records `dirty: true`: those runs preceded the source commits above.
Local workspace/interpreter paths in shared JSON have been replaced by portable
placeholders. Numeric data and control checksums are unchanged. Diagnostic
`trajectory_json_sha256` values refer to the original pre-redaction input bytes,
not the sanitized JSON serialization; source and asset manifests remain verifiable.

## Bugs found and repairs

| Area | Failure and repair | Independent or regression evidence |
|---|---|---|
| Elliptic solver without contact slots | Scalar cone placeholders were indexed as batches. Preserve empty `(0, 6)` arrays and the ordinary non-cone cost path. | Newton/CG, joint limit and friction-loss dynamics against native MuJoCo. |
| Joint equalities | Arrays of joint addresses were cast to one integer; equality coefficients/indices mixed NumPy and MLX. Retain per-row addresses and normalize dynamic arguments. | One/two equalities, including missing second joint, against native constraint rows. |
| Connect/weld | Wrong array types and unnecessary site indexing of body identifiers. Use a static body/site branch and explicit conversions. | Connect/weld × body/site native constraint comparisons. |
| RK4 | Bare arrays were passed to a dataclass-only tree mapper; NumPy tableau weights could coerce state to host arrays. Use array arithmetic and Python scalar weights. | Ten-step actuated spring/damped pendulum against native. |
| Passive forces | Joint spring parameters could be treated as static scan metadata; tendon-only springs/damping were skipped by an early return. Normalize inputs and retain tendon calculation. | Heterogeneous joint springs and fixed tendon below/inside/above its deadband against native forces. |
| Solver diagnostics | Iteration counts were not written back. Preserve counts and reset them for unconstrained forward calls. | Nonzero count assertion in constrained tests. |
| Public IO | `_impl` was read from native Data or fields were placed at the wrong nesting level; Cholesky orientation disagreed with `solve_m`. Repair nested import/export and lower-triangular factorization. | Native snapshots, both make_data entry points, empty model and matrix-factor tests. |
| Native storage compatibility | MuJoCo 3.5 owns tendon sparse metadata on Data, not Model. Branch on the actual available layout. | Dense and sparse tendon round trips on pinned MuJoCo 3.5. |
| Constraint identity | Prefix-copying active rows into static storage could mislabel a tendon limit as a joint limit, or move an active equality into an inactive one's slot. Map by `(type, object ID, component)` and reconstruct export IDs. | Mixed joint/tendon activation combinations and inactive-first equality. |
| Contacts in IO | Padding/contact addresses and positive-distance margin/gap contacts were mishandled. Preserve valid contact entries, inactive address `-1`, and compacted export IDs. | Two cones × dense/sparse × penetrating/positive-gap cases. |

The six IO test methods validate the supported snapshot schema, **not every
native diagnostic field or all sparse dynamics**. Environment-batched exports
are explicitly rejected. Unsupported non-contact row layouts fail explicitly.

## Performance changes

1. Keep argmin/argmax indices on the device in box/SAT routines; replace plane
   manifold duplicate checking's nested scalar reads with array operations.
2. Batch `plane_convex` and `box_box` pairs with `mx.vmap` within a FunctionKey.
   Other convex/heightfield paths retain the scalar fallback.
3. Convert box topology to MLX before multiplication by vmapped dynamic sizes.
   The first integration of vmap otherwise caused a fatal process exit due to
   NumPy/MLX tracer mixing. Explicit conversion eliminated the reproduced crash;
   we have not diagnosed an MLX C++ runtime defect or made such a claim upstream.
4. Reuse one inactive contact block and gather back into the original slot order;
   use one `mx.repeat` for contact-parameter expansion.

No contact slots, physical limits, sensors or friction were disabled. Broadphase
inactive distances remain documented sentinels, not exact separation distances.
This is **collision-group batching, not batched environment stepping**.

## Late finding: GPU precision policy matters

The same-state diagnostic exposed a mass-matrix diagonal of 0.0799560546875
instead of 0.08 kg. A minimal 22×6 GPU GEMM reproduced that value, while an
explicit multiply/reduction and CPU matmul returned the expected float32 value
0.07999999821186066. This is not a mass-parameter or integration-unit error.

[MLX's official numerical precision documentation](https://ml-explore.github.io/mlx/build/html/usage/precision.html)
describes reduced-precision float32 matrix operations and the process-launch
setting `MLX_ENABLE_TF32=0`. We confirmed the setting on the installed 0.32.2,
without upgrading MLX or changing its C++ code. Do not change the flag after
initialization or silently mutate a caller's global runtime policy.

At the same three native reference states (0/25/50 steps), this setting reduces
the maximum mass-matrix error from 4.39453e-5 to **1.78814e-9**. At step 0 it
restores contact for geometry pair 40:89 (native 2 contacts; default MLX 0;
full-float32 MLX 4) and reduces maximum generalized constraint-force difference
from 0.9663 to 0.0006095. At step 50 it still misses pair 33:89 and the maximum
constraint-force difference remains 0.2313. This points to remaining contact/
constraint discrepancies, not solely accumulated state drift. Different contact
counts alone are not a proof of wrong forces. Native replay is verified exactly.

Evidence: `same-state-diagnostics.json` (default), `same-state-fp32.json` (full).
The new four-free-body native mass-matrix test fails with the default precision
on this M5, and passes at unchanged `atol=1e-7` with full float32. Thus the previous
24 passing tests had a genuine blind spot. `regression-default-precision.json`
records the expected 1/25 failure; `regression-final.json` records 25/25 passing.

Full-float32 measured ablation: **1.4115 → 0.14918 s**, **9.46×**, three paired
trials; eight active contact outputs are exactly equal to the old collision
implementation in the same precision mode (`collision-paired-fp32.json`).

However, `leap-action-50-fp32.json` still fails: 50 finite steps, 36.522 s of
synchronized stepping, 1.369 steps/s; maximum translation **2.28648 mm**,
orientation **0.0644789 rad / 3.694°**, joint difference 0.00495032 rad.
The same control-sequence checksum and unmodified gates are used. A more correct
mass matrix does not guarantee a closer nonlinear contact trajectory; restoring
precision did **not** solve all contact physics. Both failed traces are retained.
No global Conda/shell configuration was changed; the flag is passed per process
and is now logged by the runners. This remains a GPU run, not a CPU fallback.

## Earlier measured results (MLX default reduced precision)

| Measurement | Baseline | Current | Interpretation |
|---|---:|---:|---|
| Paired collision median | 1.639 s | 0.1545 s | **10.61×** faster; same process, frozen input, three alternating-order pairs after warmup. |
| Historical identical 10-step staged probe | 34.385 s | 11.500 s | **2.99×** end-to-end improvement; historical rather than contemporaneous full-engine baseline. |
| Mean collision in that probe | 2.141 s | 0.2389 s | Instrumented phase timing. |
| Mean constraint assembly in that probe | 0.1682 s | 0.01173 s | Lazy evaluation/stage boundaries affect attribution. |
| Mean solver in that probe | 0.7507 s | 0.6284 s | Remains a major cost; iteration/runtime variation matters. |

The paired ablation restores **only** `collision_convex.py` and
`collision_driver.py` from `10c4fa4a`; mesh/math/io dependencies are shared current
code. Input construction: 100 native steps, one current MLX step, then position
evaluation and freezing. Both measured implementations receive that same state.
All 7016 slots, geometry IDs, dimensions and addresses match. All eight active
slots' distances, positions and frames are exactly equal in the measured runs.
This does not certify collision accuracy relative to native MuJoCo.

The repeated 10-step trajectory's final MLX qpos is **elementwise identical** to
the saved E012 output. Native-relative final cube translation error remains
0.042017 mm, orientation 0.00254396 rad (0.1458°), maximum hand-joint error
7.38858e-5 rad. These short-run measurements are not long-horizon guarantees.

### Action trajectory: execution passes, accuracy acceptance fails

`leap-action-50.json`: 50 physical steps at 0.002 s, holding each seeded small
position-control action for 25 steps; two actions, 0.1 s of simulation.
Both backends start from the same state after 100 native preparation steps.

- All 50 steps finish with finite states; 48.574 s in synchronized MLX stepping,
  **1.029 physical steps/s**, or **0.04117 action-equivalents/s**.
- Maximum cube translation error: **0.7585 mm**, below the preset 1 mm gate.
- Maximum cube orientation error: **0.0254334 rad (1.457°)**, above the preset
  **0.02 rad (1.146°)** gate. The command returns nonzero and the JSON is **failed**.
- Maximum hand-joint error: 0.00524668 rad, below the preset 0.05 rad gate.
- Active MLX slots range 0–12, native active contacts 0–8; manifold counts are
  not generally one-to-one. Final counts are 8 versus 3.

Thresholds are diagnostic acceptance choices set before the run, not universal
physics tolerances; they were not loosened after failure. The run overlaps a
brief ~1.4 s regression process at startup, so use the isolated paired collision
ablation and the separately run 10-step probe for stronger performance evidence.
The CPU-only same 50-step/action trace achieves about 45,181 physics steps/s
(~16,014 including this harness's inspection). This is a short native sample,
not a steady-state batch benchmark. GPU execution still loses by orders of
magnitude for one hand environment. None of these timings includes PPO,
policy inference, rewards, rendering or environment resets.

## Test coverage and retained failed attempts

- `regression-final.json`: 25/25 repair regression methods in full float32, no skips. Contains
  mixed cones/dimensions, native Jacobians/forces/RK4, public IO and collision
  edge cases. Scalar-port reference tests alone cannot expose shared old bugs.
- `upstream-tests.json`: 23/23 original smoke functions. Many are import-only;
  do not equate this with full physics validation. The unrelated original math
  throughput benchmark is not part of that count.
- `upstream-runpy-harness-failure.json`: first runner incorrectly unregistered
  the dynamically loaded module, breaking dataclass annotation lookup; fixed
  by a registered import in `run_upstream_smoke.py`, not by changing physics.
- Initial edge tests found equality coefficient conversion; initial IO tests
  found the MuJoCo 3.5 tendon ABI; RK4 uncovered spring conversion. These were
  fixed and rerun, not removed from the test set.
- `collision-paired-no-active.json`: valid timing, but zero active contacts
  made active-value comparison vacuous. It is not the accepted ablation.
- `collision-warm110-rejected.json`: the added nonzero-contact gate rejected a
  second zero-active pose. Final ablation uses the explicit preparation above.
- `regression-preliminary.json` and `regression-collision.json` are intermediate
  nine-method checkpoints, superseded by the full final result.
- `regression-before-precision-guard.json` retains the earlier 24/24 result;
  the stricter native-mass test is what exposed the default precision issue.
- Both `leap-action-50.json` and `leap-action-50-fp32.json` remain
  **unresolved accuracy failures**, with their different precision modes.

## Remaining blockers and next engineering order

1. Diagnose contact manifold/constraint-force discrepancies at identical native
   states; then repeat action trajectories, including meaningful rotation,
   contact transitions and longer horizons. A finite trajectory is insufficient.
2. Reduce solver work on inactive rows while preserving row identity; profile
   host synchronizations and NumPy/SciPy dynamics. Cache topology only with an
   explicit invalidation strategy, not `id(model)` assumptions.
3. Only after physical acceptance, add a real environment batch dimension to
   Data, kinematics, contacts, constraints, solver and integration. `scan._vmap`
   is still a Python loop. Increasing `--num-envs` in the system harness means
   **serial replicas**, not GPU batching. Compiling the present full step is not
   a drop-in solution while host NumPy/SciPy paths remain.

Other unvalidated/unsupported areas: elliptic condim=1, spatial tendon wrapping,
arbitrary contact sensors and contact-force queries, arbitrary actuators, sparse
forward dynamics, differentiation and large-scale resets. IO currently compacts
rows using nonzero Jacobians, so degenerate zero-Jacobian constraints may not
round-trip; native `efc_b`/KBIP/full solver diagnostics are not retained.

## Reproduce

Run from repository root using the isolated environment, with LEAP assets at
`../../models/leap_hand/cube_scene.xml`. Fetch `repair-baseline` and branch history
for historical scalar oracles. No dependency reinstall is required.

```sh
export MLX_ENABLE_TF32=0
python mjx/mujoco/run_hand_regressions.py --output /tmp/hand-regressions.json
python mjx/mujoco/run_upstream_smoke.py --output /tmp/upstream-smoke.json
python mjx/mujoco/benchmark_collision_audit.py \
  --xml ../../models/leap_hand/cube_scene.xml --native-warmup 100 \
  --mlx-prepare-steps 1 --repeats 3 --timeout 120 --output /tmp/collision-paired.json
python mjx/mujoco/probe_hand_pipeline.py \
  --xml ../../models/leap_hand/cube_scene.xml --native-warmup 100 \
  --steps 10 --timeout 60 --output /tmp/contact-10.json
python mjx/mujoco/benchmark_hand_system.py \
  --xml ../../models/leap_hand/cube_scene.xml --backend compare \
  --native-warmup 100 --steps 50 --substeps 25 --num-envs 1 \
  --timeout 180 --output /tmp/hand-action-50.json
python mjx/mujoco/diagnose_hand_contacts.py \
  --xml ../../models/leap_hand/cube_scene.xml \
  --trajectory-json /tmp/hand-action-50.json --timeout 120 \
  --output /tmp/same-state-diagnostics.json
```

The action benchmark is expected to fail the trajectory gates with this checkpoint;
the diagnostic reports finite comparisons, not physical-equivalence acceptance.
Keep native MuJoCo for the project's training backend; no PPO backend switch or
training run was made during this audit. No upstream PR has been submitted.
