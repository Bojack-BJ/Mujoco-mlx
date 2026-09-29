# Hand-contact repair: issue and change ledger

Date: 2026-09-29. Upstream base: `5de472e576e5233d159c1a2c46bb4f70a75c2db4`.
Branch: `fix/hand-contact-mlx`. Scope: Python `mjx_mlx` only; no C/C++ or official
JAX MJX changes. This is an experimental repair branch, not a replacement verdict.

## Earlier work carried by this branch

| Problem | Change | Evidence / limit |
| --- | --- | --- |
| Missing dataclasses import, NumPy indices/scalars rejected by MLX | Import and explicit array conversions | Compatibility baseline `0e99a283` / `repair-baseline` |
| Hinge/ball address vectors cast to scalar | Retain per-joint address vectors | LEAP reaches collision |
| Quaternion scalar concatenation failure | Reshape scalar before concatenation | Free cube integrates |
| Flattened model/data bypass collision; mesh hulls discarded | Restore dispatch, retain `mesh_convex` | Cube ground contacts work |
| Batched constraint parameter shape produced NaNs | Correct row parameter tiling | Earlier 500-step upright cube finite |
| 1754 candidate pairs, 7016 contact slots produce excessive scalar work | Conservative box/mesh sphere rejection, preserve inactive slots | Collision previously >30 s, ~1.54 s after change |
| 144x144 mesh-edge pairs exceed 499000 Metal resources | Batched Gauss-map gather/cross/dot calculations | General convex collision regression passes |

The earlier `91a7da2d` stage still failed inside contact Jacobian assembly.
Broadphase distances for rejected pairs are sentinels, not geometric distances.
Broadphase uses a NumPy synchronization and is not differentiable/compiled batching.

## This iteration: chronological failure chain

| ID | Observation | Root cause and action |
| --- | --- | --- |
| C1 | Original LEAP step exhausts Metal resources in `support.jac` | Per-contact, per-DOF graph construction. Added `jac_batch` and vectorized all supported contact dimension groups; vectorized `_kbi` and final row assembly. |
| C2 | New batch path exposes `index 7016 out of bounds` | `_concat_contacts` concatenated global NumPy `dim`/`efc_address` once per group. Keep static metadata once, concatenate only MLX contact leaves. |
| C3 | Mixed friction/limit/contact row concatenation type error | `_row` returned NumPy scalars unchanged for unbatched rows; tree mapping treated them as metadata. Normalize every row leaf before branching. |
| C4 | JOINTPOS casts a vector to `int` | Use vector MLX indices for JOINTPOS, JOINTVEL, JOINTACTFRC. |
| C5 | Tendon spring transpose rejects NumPy array | Convert input. A subsequent Data conversion trial also exposed a process-level crash in empty NumPy/MLX tendon multiplication; skip nonexistent tendon work and convert damping inputs. |
| S1 | Step now reaches solver but exhausts resources building per-contact cone Hessians | Batch six-row gathers, bottom-zone row updates, force assembly and cone Hessians. |
| S2 | Newton path allocates a dense `nefc x nefc` matrix | Directly contract diagonal and six-row contact blocks into `nv x nv`. For LEAP, 21080 squared float32 entries alone would be ~1.78 GB. |
| S3 | Line search repeats whole-array updates for every tangent row | Gather/sum padded blocks and replace normal rows in one operation; retain existing row order and addresses. |
| I1 | `deriv_smooth_vel requires MLX MJX implementation` | Accept this port's flattened MLX Model plus nested Data representation. Convert damping diagonal arguments. Fluid drag remains unsupported. |
| I2 | `linalg::solve` unsupported on GPU | Explicit CPU stream for small dense implicit integration solve; no claim of pure-GPU execution. |
| I3 | Mixed hinge/free integration and heterogeneous actuator gains treated as static NumPy arguments | Convert dynamic integration/gain/bias arguments to MLX. Preserve enum grouping on host. |
| D1 | Two-joint test initially drifts ~0.0168 rad after 10 steps; state fields unexpectedly change across steps | Initial `_numpy_to_mlx` used a tree mapper that deliberately skips NumPy, leaving dynamic Data fields unconverted. Convert Data fields by declared type recursively, preserving only true static metadata. The underlying NumPy/MLX aliasing mechanism was not independently isolated; the initialization defect and corrected regression are verified. |
| D2 | Actuator-force sensors remain zero without constraints | Forward returned before acceleration-stage sensors. Always run those sensors; explicitly zero constraint force when there are no constraint rows. |
| D3 | Mixed-dimension test finds `nf=nl=0` despite friction and limits | `make_data` computed counts but placed them only into the top-level dictionary, then defaulted nested DataMLX counts to zero. Populate nested counts and metadata explicitly. This supersedes earlier LEAP numerical runs. |

Local unsuccessful LEAP attempts are preserved as `attempts/leap-*.json`.
Attempt 05 failed before report creation due to an indentation mistake in the
temporary tendon patch; it was corrected immediately. A later implicit-solve
patch indentation was caught by inspection and corrected before final testing.
These editing mistakes are not attributed to upstream.

The first regression harness used top-level `efc_J` instead of `_impl.efc_J`;
fixed in the harness. Its scalar frictionless oracle also needed MLX leaf
normalization; that normalization is explicit in the test, not hidden as an
upstream-correctness claim. Broad unittest discovery imported the unrelated JAX
tree and failed its imports; `run_hand_regressions.py` names only our test modules.

## Verification and acceptance boundaries

Final implementation snapshot: `85cfed38`. Contact/inline probes were run while
the last metadata fix was uncommitted (their JSON correctly records `dirty=true`);
the same source is now committed. `final-source-sha256.txt` identifies the full
validated Python implementation. The 50-step intermediate log is not this source.

- `test_collision_optimization.py`: active/inactive box slots and four scalar-vs-batch convex SAT cases.
- `test_constraint_batch.py`: Jacobian vs scalar and native MuJoCo; seven supported dimension/cone combinations (elliptic condim=1 explicitly unsupported); row coefficients; dynamic Data types and static metadata cardinality; K/B/I broadcasting.
- `test_solver_batch.py`: CG/Newton, condim 3/4/6, cost/force/Hessian/gradient/line search/solve vs scalar code; mixed dimensions plus limit/friction rows; ten-step damped, actuated two-joint implicitfast and sensors vs native MuJoCo.
- Native two-joint tolerances are explicit: qpos 2e-5; sensor 5e-4, allowing observed float32 small-inertia velocity error ~2.5e-4. These are regression tolerances, not universal simulator accuracy bounds.
- The 50-step intermediate LEAP trace had no active hand-object contacts and predates D3. It is not contact or correctness acceptance.
- Final results are in `regression.json` and `leap-contact-10.json`; the latter starts from 100 native warmup steps, copies state to both backends and resets native warmstart. It is a short contact smoke test, not policy validation.
- The asset-free inline reproducer completes 3/3 steps in ~1.80 s. Its boxes are
  still above the plane, so it validates the pipeline and multiple-group layout,
  not sustained ground contact; active contact coverage comes from the regression
  models and the LEAP contact-state probe.

Not established: long-horizon hand fidelity, arbitrary sensors/actuator dynamics,
tendons, mixed equality constraints, batched environments, gradients, PPO speedup,
or replacement of native MuJoCo. No automatic fallback disables contact, limits,
friction or sensors to make tests pass. Other NumPy/MLX interoperability gaps may remain.

## Reproduce

Use an isolated Apple Silicon environment with versions in `environment.json`.
Do not install into an existing training environment. Full history (including
`repair-baseline` and `91a7da2d`) is needed for scalar-reference tests.

```sh
python mjx/mujoco/run_hand_regressions.py --output validation/hand-repair/regression.json
python mjx/mujoco/probe_hand_pipeline.py --steps 3 --timeout 45 --output /tmp/mlx-inline.json
python mjx/mujoco/probe_hand_pipeline.py --xml /path/to/leap/cube_scene.xml --native-warmup 100 --steps 10 --timeout 60 --output /tmp/mlx-leap-contact.json
```

The inline model requires no external meshes. LEAP measurements require the
project's LEAP mesh assets and cube scene; the top-level XML hash is recorded,
but it is not a transitive hash of included XML/meshes. Asset checksums are also
saved separately. Probe failures return nonzero; the older local E011/E012 probe
returned zero after recording an exception, so inspect its JSON, not exit status.
# Subsequent E013 audit

The [systematic audit](../system-audit/REPORT.md) extends this earlier ledger with
public IO, equality/RK4/passive-force repairs, collision-group batching, strict
mass precision validation and bounded same-state/trajectory diagnostics.
Final full-float32 regressions: 25/25. Paired collision: 9.46× improvement.
Both default and full-float32 50-step action traces still fail physical acceptance.
Run physics in a fresh process with `MLX_ENABLE_TF32=0`; no native/PPO replacement
is claimed. The entries below preserve the E012 failure chain without rewriting it.
