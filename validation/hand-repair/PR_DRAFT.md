# Draft: fix hand-contact execution and batch MLX constraint/solver work

Not submitted. Intended upstream: RobotFlow-Labs/Mujoco-mlx.

## E013 addendum (latest evidence)

Late precision finding: on M5, default float32 matrix operations produce a
mass-matrix error of 4.39e-5 for this model. The documented process setting
`MLX_ENABLE_TF32=0` reduces it to 1.79e-9. A new strict native-mass regression
makes the count 25/25 in full float32; the default now correctly fails 1/25.
Full-float32 paired collision gives 9.46× improvement, but its 50-step trajectory
still exceeds the unchanged position/orientation gates (2.286 mm / 3.694°).
The older default-precision numbers below are retained as separate evidence.

The [system audit](../system-audit/REPORT.md) adds empty-elliptic, equality, RK4,
passive-force and native IO repairs, including constraint object identity across
inactive rows. Supported plane/box collision groups use real MLX vmap; topology
conversion precedes traced sizes. Three alternating-order paired comparisons
show 10.61× collision-only improvement and exact equality of eight active
contacts. Same historical ten-step full probe: 34.385 → 11.500 s, identical final
qpos. There are now 24 repair regression methods and 23 upstream smoke functions
passing. These counts do not imply broad physical equivalence.

Crucially, a 50-step/two-action LEAP comparison runs but fails a predeclared
orientation tolerance: 0.02543 rad maximum versus 0.02 rad allowed. Keep that
failure visible. No cross-environment batch, PPO integration, or native-backend
replacement is claimed. Source changes are separately committed as collision
performance (`6691e627`), physics correctness (`f9dd2646`), and native IO
(`9ac02b50`). The sections below retain the earlier E012 checkpoint evidence.

## Problem

The MLX port handles simple examples but complex hand scenes fail before one
complete step. The initial LEAP model has 1754 candidate geometry pairs and
7016 allocated contacts. Scalar graph construction exhausts Metal resources
first in collision, then Jacobian assembly, then elliptic cone Hessians.
Further execution exposes array conversion, metadata and integration defects.

## Changes

- Conservative convex-pair rejection and vectorized Gauss-map edge operations.
- Preserve global contact dimension/address metadata during group concatenation.
- Batch point Jacobians, frictionless/pyramidal/elliptic constraint rows and K/B/I.
- Batch elliptic contact gathers, force and Hessian operations, and line-search
  quadratics. Contract small cone blocks directly into the generalized-coordinate
  Hessian instead of allocating a dense constraint-space square matrix.
- Initialize dynamic Data leaves as MLX arrays while retaining static metadata;
  retain equality/friction/limit/contact counts in nested DataMLX.
- Repair joint sensor indices, no-contact sensor lifecycle, no-tendon damping,
  dynamic actuator/integration arguments and implicitfast compatibility.
- Explicit CPU stream for the small implicit dense solve, required by MLX 0.32.2.

No C/C++ or official JAX code is changed. No contacts, limits, sensors or friction
are disabled to bypass failures. Contact slots and constraint row ordering remain
unchanged. See the issue ledger for each observed failure and its corresponding
fix; the branch history separates compatibility, assembly and solver work.

## Evidence

- 9 regression methods pass, including scalar-vs-batch comparisons, native
  Jacobians, seven supported dimension/cone combinations, CG/Newton, mixed
  dimensions plus friction/limit rows, implicit integration and sensors.
- Complete LEAP contact-state probe: 10/10 steps, finite states, 34.39 s total.
  Active allocated contacts range from 0 to 8. Native warmup: 100 steps; then
  the same state is used by both backends. These contact counts are not asserted
  equal to native contact manifold counts.
- Final cube position error 0.042 mm; quaternion-geodesic orientation error
  0.00254 rad; maximum final hand-joint error 7.39e-5 rad versus native MuJoCo.
  These are measured values, not general accuracy guarantees.
- Mean staged time: collision 2.14 s, constraints 0.168 s, solver 0.751 s.
  Roughly 0.29 physical steps/s including probe overhead: not an RL speedup.
- Earlier 50-step trace is retained but superseded: no hand-object contact and
  pre-count-fix source. Do not use it as correctness acceptance.

## Reproduction / limitations

Use `run_hand_regressions.py` and `probe_hand_pipeline.py` as documented in
`ISSUES_AND_CHANGES.md`. Environment, raw logs and asset checksums are included.
Scalar-reference tests need the branch history and `repair-baseline` tag.

The backend remains eager and hybrid CPU/GPU, with host synchronizations.
No batched-environment throughput, PPO training, long-horizon fidelity,
gradients, arbitrary sensor/actuator/tendon coverage, or complete upstream test
suite pass is claimed. Elliptic condim=1 remains explicitly unsupported.

Suggested review split if this is too large: (1) metadata and type correctness,
(2) collision/constraint batching, (3) solver contractions and integration.
