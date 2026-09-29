# Draft: fix hand-contact execution and batch MLX constraint/solver work

Not submitted. Intended upstream: RobotFlow-Labs/Mujoco-mlx.

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
