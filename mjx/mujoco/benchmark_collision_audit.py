"""Paired collision-only ablation: 10c4fa4a versus current, in one GPU process.

Both variants receive the same frozen LEAP pose after native warmup (default 100
steps), transfer to current MLX, and current MLX preparation (default one step).
Current fwd_position is then evaluated before freezing the pose. At least one
active contact is required; otherwise accuracy checks would be vacuous and the
audit fails while preserving its evidence.
Only collision_driver and collision_convex are restored for the baseline;
mesh/math/io and other dependencies remain current. This is NOT an old-engine
versus new-engine benchmark, an integration benchmark, or a PPO benchmark.
One untimed warmup per variant precedes alternating-order timed pairs. Timings
include collision dispatch and mx.eval(dist, pos, frame), but exclude validation.
An independent supervisor enforces the wall-time deadline even for stalled Metal.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
import types


BASELINE = '10c4fa4a'
CONVEX_EXPORTS = ('box_box', 'capsule_convex', 'convex_convex', 'hfield_capsule',
                  'hfield_convex', 'hfield_sphere', 'plane_convex', 'sphere_convex')


def save(output, report):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.partial')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xml', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--native-warmup', type=int, default=100)
    parser.add_argument('--mlx-prepare-steps', type=int, default=1,
                        help='Current-backend physics steps before freezing the shared collision input.')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.repeats < 1 or args.native_warmup < 0 or args.mlx_prepare_steps < 0
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error('repeats and finite timeout must be positive; preparation steps must be nonnegative')
    return args


def supervise(args):
    start = time.perf_counter()
    initial = dict(status='starting', benchmark='paired_collision_only_ablation')
    save(args.output, initial)
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                              *sys.argv[1:], '--_worker'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = child.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = child.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            stdout, stderr = child.communicate(timeout=5)
    try:
        report = json.loads(args.output.read_text())
    except (OSError, ValueError):
        report = initial
    if timed_out:
        report.update(status='failed', timed_out=True, error='Hard wall-time deadline exceeded')
    elif child.returncode != 0 or report.get('status') != 'completed':
        report['status'] = 'failed'
        report.setdefault('error', 'Worker exited without a successful report')
    report.update(worker_returncode=child.returncode,
                  supervisor_wall_seconds=time.perf_counter() - start)
    if stderr:
        report['worker_stderr_tail'] = stderr[-8000:]
    if stdout:
        report['worker_stdout_tail'] = stdout[-2000:]
    save(args.output, report)
    print(json.dumps(dict(status=report['status'], output=str(args.output.resolve()),
                         median_speedup=report.get('median_speedup'))))
    return 0 if report['status'] == 'completed' else 1


def audit(args):
    started = time.perf_counter()
    report = dict(status='running', phase='imports',
                  benchmark='paired_collision_only_ablation',
                  MLX_ENABLE_TF32=os.environ.get('MLX_ENABLE_TF32', '1 (MLX default)'),
                  baseline_requested=BASELINE, native_warmup_steps=args.native_warmup,
                  mlx_prepare_steps=args.mlx_prepare_steps,
                  require_active_contacts=True,
                  input_construction='Native warmup at zero control, transfer qpos/qvel/act/ctrl/time '
                                     'to current MLX, run current forward.step preparation, then '
                                     'current fwd_position. Freeze this identical input for both variants.',
                  repeats=args.repeats, baseline_seconds=[], current_seconds=[],
                  warmup_seconds={}, ordered_trials=[], numerical_checks=[],
                  scope='Only baseline collision_convex and collision_driver are '
                        'historical. All other dependencies are shared current code. '
                        'Both variants run in one process on one frozen input state.',
                  tolerance=dict(absolute=2e-5, relative=0),
                  timeout_seconds=args.timeout)
    save(args.output, report)
    try:
        import mujoco
        import numpy as np
        import mlx.core as mx
        local = Path(__file__).resolve().parent
        mujoco.__path__.insert(0, str(local))
        from mujoco.mjx_mlx._src import collision_convex, collision_driver, forward, io
        root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'],
                                           cwd=local, text=True).strip())
        git = lambda *arguments: subprocess.check_output(['git', *arguments], cwd=root)
        full_baseline = git('rev-parse', BASELINE + '^{commit}').decode().strip()
        mx.set_default_device(mx.gpu)
        report.update(python=platform.python_version(), executable=sys.executable,
                      platform=platform.platform(), mujoco=mujoco.__version__,
                      mlx=importlib.metadata.version('mlx'), device=str(mx.default_device()),
                      baseline_revision=full_baseline,
                      current_revision=git('rev-parse', 'HEAD').decode().strip(),
                      dirty=bool(git('status', '--porcelain').decode().strip()),
                      xml=str(args.xml.resolve()),
                      xml_sha256=hashlib.sha256(args.xml.read_bytes()).hexdigest(),
                      hash_limitation='XML hash excludes referenced assets and includes',
                      source_sha256=dict(baseline={}, current={}))
        restored = {}
        for name in ('collision_convex', 'collision_driver'):
            filename = local / 'mjx_mlx' / '_src' / (name + '.py')
            source = git('show', f'{full_baseline}:{filename.relative_to(root)}')
            report['source_sha256']['baseline'][name] = hashlib.sha256(source).hexdigest()
            module = types.ModuleType('collision_audit_baseline_' + name)
            module.__file__ = f'{full_baseline}:{filename.relative_to(root)}'
            exec(compile(source, module.__file__, 'exec'), module.__dict__)
            restored[name] = module
        old_convex, old_driver = restored['collision_convex'], restored['collision_driver']
        replacements = {getattr(collision_convex, name): getattr(old_convex, name)
                        for name in CONVEX_EXPORTS}
        # Decorators name each wrapped function "collide", so match exported
        # function identity, not __name__. Primitive functions stay current.
        old_driver._COLLISION_FUNC = {
            key: replacements.get(function, function)
            for key, function in old_driver._COLLISION_FUNC.items()}
        for name in CONVEX_EXPORTS:
            setattr(old_driver, name, getattr(old_convex, name))
        assert all(getattr(old_convex, name) in old_driver._COLLISION_FUNC.values()
                   for name in CONVEX_EXPORTS), 'Incomplete baseline convex rebinding'
        report['rebound_convex_exports'] = list(CONVEX_EXPORTS)
        for name in ('collision_convex', 'collision_driver', 'mesh', 'math', 'io', 'forward'):
            filename = local / 'mjx_mlx' / '_src' / (name + '.py')
            report['source_sha256']['current'][name] = hashlib.sha256(filename.read_bytes()).hexdigest()
        report['phase'] = 'prepare_frozen_state'
        save(args.output, report)
        setup_started = time.perf_counter()
        m = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
        native = mujoco.MjData(m)
        for _ in range(args.native_warmup):
            mujoco.mj_step(m, native)
        x, d = io.put_model(m), io.make_data(m)
        d = d.replace(qpos=mx.array(native.qpos), qvel=mx.array(native.qvel),
                      act=mx.array(native.act), ctrl=mx.array(native.ctrl),
                      time=mx.array(native.time))
        for _ in range(args.mlx_prepare_steps):
            d = forward.step(x, d)
            mx.eval(d.qpos, d.qvel)
        d = forward.fwd_position(x, d)
        impl = d._impl if d._impl is not None else d
        mx.eval(d.qpos, d.qvel, d.geom_xpos, d.geom_xmat,
                impl.contact.dist, impl.contact.pos, impl.contact.frame, impl.efc_J)
        frozen_hash = hashlib.sha256(np.asarray(d.qpos).tobytes()
                                     + np.asarray(d.geom_xpos).tobytes()
                                     + np.asarray(d.geom_xmat).tobytes()).hexdigest()
        report.update(setup_seconds=time.perf_counter() - setup_started,
                      frozen_state_sha256=frozen_hash, qpos=np.asarray(d.qpos).tolist(),
                      initial_native_contact_count=int(native.ncon),
                      frozen_current_active_contacts=int(np.sum(
                          np.asarray(impl.contact.dist) < np.asarray(impl.contact.includemargin))),
                      model_dimensions=dict(nq=m.nq, nv=m.nv, ngeom=m.ngeom),
                      contact_slots=int(impl.contact.dist.size))
        functions = {'baseline': old_driver.collision, 'current': collision_driver.collision}

        def measured(name):
            start = time.perf_counter()
            result = functions[name](x, d)
            result_impl = result._impl if result._impl is not None else result
            contact = result_impl.contact
            mx.eval(contact.dist, contact.pos, contact.frame)
            elapsed = time.perf_counter() - start
            return elapsed, contact

        def compare_contacts(baseline, current, trial):
            check = dict(trial=trial, status='checking', max_absolute_error={})
            report['numerical_checks'].append(check)
            for field in ('dim', 'efc_address', 'geom', 'geom1', 'geom2', 'includemargin'):
                np.testing.assert_array_equal(np.asarray(getattr(current, field)),
                                              np.asarray(getattr(baseline, field)),
                                              err_msg=f'{trial}: {field} differs')
            baseline_dist, current_dist = np.asarray(baseline.dist), np.asarray(current.dist)
            self_same_shape = baseline_dist.shape == current_dist.shape
            assert self_same_shape, f'{trial}: contact slot shape differs'
            baseline_active = baseline_dist < np.asarray(baseline.includemargin)
            current_active = current_dist < np.asarray(current.includemargin)
            np.testing.assert_array_equal(current_active, baseline_active,
                                          err_msg=f'{trial}: active contact mask differs')
            check.update(slots=int(baseline_dist.size), active_contacts=int(baseline_active.sum()),
                         dim_and_address_equal=True, activation_equal=True)
            if not np.any(baseline_active):
                check.update(status='failed', error='No active contacts; contact accuracy validation is vacuous')
                raise AssertionError('No active contacts in the frozen state; choose different '
                                     '--native-warmup/--mlx-prepare-steps for a non-vacuous hand-contact audit')
            for field in ('dist', 'pos', 'frame'):
                expected, actual = np.asarray(getattr(baseline, field)), np.asarray(getattr(current, field))
                assert np.isfinite(expected).all() and np.isfinite(actual).all(), f'Nonfinite {field}'
                delta = np.abs(actual[baseline_active] - expected[baseline_active])
                check['max_absolute_error'][field] = float(np.max(delta, initial=0))
                np.testing.assert_allclose(actual[baseline_active], expected[baseline_active],
                                           atol=2e-5, rtol=0, err_msg=f'{trial}: active {field} differs')
            check['status'] = 'passed'

        warmed = {}
        for name in ('baseline', 'current'):
            report['phase'] = f'warmup_{name}'
            save(args.output, report)
            elapsed, warmed[name] = measured(name)
            report['warmup_seconds'][name] = elapsed
        compare_contacts(warmed['baseline'], warmed['current'], 'warmup')
        del warmed
        for repeat in range(args.repeats):
            order = ('baseline', 'current') if repeat % 2 == 0 else ('current', 'baseline')
            results = {}
            trial = dict(repeat=repeat, order=list(order), seconds={})
            report['ordered_trials'].append(trial)
            for name in order:
                report['phase'] = f'repeat_{repeat}_{name}'
                save(args.output, report)
                elapsed, results[name] = measured(name)
                report[name + '_seconds'].append(elapsed)
                trial['seconds'][name] = elapsed
            compare_contacts(results['baseline'], results['current'], repeat)
        after_hash = hashlib.sha256(np.asarray(d.qpos).tobytes()
                                    + np.asarray(d.geom_xpos).tobytes()
                                    + np.asarray(d.geom_xmat).tobytes()).hexdigest()
        assert after_hash == frozen_hash, 'Input state mutated during paired ablation'
        baseline_median = float(np.median(report['baseline_seconds']))
        current_median = float(np.median(report['current_seconds']))
        report.update(status='completed', phase='finished', input_state_unchanged=True,
                      baseline_median_seconds=baseline_median,
                      current_median_seconds=current_median,
                      median_speedup=baseline_median / current_median,
                      inactive_slot_policy='Preserve slots and activation; inactive sentinel '
                                           'distances, positions and frames need not match')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
    finally:
        report['worker_wall_seconds'] = time.perf_counter() - started
        save(args.output, report)
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    arguments = parse_args()
    sys.exit(audit(arguments) if arguments._worker else supervise(arguments))
