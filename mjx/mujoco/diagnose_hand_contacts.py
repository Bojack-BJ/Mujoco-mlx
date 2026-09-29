"""Compare forward dynamics at three identical native trajectory states.

No MLX integration is performed: differences here cannot be attributed to
accumulated MLX trajectory drift. This is diagnostic evidence, not acceptance.
The parent process enforces a hard deadline, including stalled GPU calls.
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

from benchmark_hand_system import save


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xml', type=Path, required=True)
    parser.add_argument('--trajectory-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be finite and positive')
    return args


def supervisor(args):
    start = time.perf_counter()
    initial = dict(status='starting', diagnostic_only=True, snapshots=[],
                   timeout_seconds=args.timeout)
    save(args.output, initial)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--_worker']
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = child.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(child.pid, signal.SIGTERM)
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
        report.update(status='failed')
        report.setdefault('error', 'Worker exited without a successful report')
    report.update(worker_returncode=child.returncode,
                  wall_seconds_including_startup=time.perf_counter() - start,
                  worker_stdout_tail=stdout[-4000:], worker_stderr_tail=stderr[-8000:])
    save(args.output, report)
    print(json.dumps(dict(status=report['status'], output=str(args.output.resolve()))))
    return 0 if report['status'] == 'completed' else 1


def worker(args):
    started = time.perf_counter()
    report = dict(status='running', phase='imports', diagnostic_only=True, snapshots=[],
                  methodology='Replay only the native reference; evaluate both forward '
                    'dynamics at native states after 0, 25, and 50 physical steps. '
                    'Reset both diagnostic warm starts to zero; do not integrate MLX.',
                  limitations=['MLX inputs round the native integration state to float32',
                               'No accuracy acceptance threshold is applied',
                               'Mass/force/acceleration norms can mix translational and rotational units',
                               'Geometry pair groups are compared, not a one-to-one contact correspondence',
                               'XML checksum excludes external included XML and meshes'])
    save(args.output, report)
    try:
        import mujoco
        import numpy as np
        import mlx.core as mx
        mujoco.__path__.insert(0, str(Path(__file__).resolve().parent))
        from mujoco.mjx_mlx._src import forward, io, support
        mx.set_default_device(mx.gpu)
        root = Path(__file__).resolve().parents[2]
        source_paths = sorted((Path(__file__).resolve().parent / 'mjx_mlx' / '_src').glob('*.py'))
        source_paths.extend([Path(__file__).resolve(), Path(__file__).with_name('benchmark_hand_system.py')])
        report.update(python=platform.python_version(), python_executable=sys.executable,
                      platform=platform.platform(), mujoco=mujoco.__version__,
                      mlx=importlib.metadata.version('mlx'), device=str(mx.default_device()),
                      revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                      dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()),
                      source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in source_paths},
                      xml=str(args.xml.resolve()), xml_sha256=hashlib.sha256(args.xml.read_bytes()).hexdigest(),
                      trajectory_json=str(args.trajectory_json.resolve()),
                      trajectory_json_sha256=hashlib.sha256(args.trajectory_json.read_bytes()).hexdigest())
        trajectory = json.loads(args.trajectory_json.read_text())
        if trajectory.get('xml_sha256') != report['xml_sha256']:
            raise ValueError('Trajectory and diagnostic XML checksums differ')
        if trajectory['steps_per_env'] < 50:
            raise ValueError('Need a trajectory with at least 50 requested physical steps')
        env = trajectory['environments'][0]
        controls = env['controls']
        m = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
        x = io.put_model(m)
        base = np.asarray(controls['base'], dtype=float)
        amplitudes = np.asarray(controls['maximum_offsets'], dtype=float)
        if base.shape != (m.nu,) or amplitudes.shape != (m.nu,):
            raise ValueError('Control metadata shape does not match model')
        lower, upper = np.full(m.nu, -np.inf), np.full(m.nu, np.inf)
        limited = np.asarray(m.actuator_ctrllimited, dtype=bool)
        lower[limited], upper[limited] = m.actuator_ctrlrange[limited].T
        for actuator in controls['position_actuator_ids']:
            joint = int(m.actuator_trnid[actuator, 0])
            if m.jnt_limited[joint]:
                bounds = np.sort(m.jnt_range[joint] * m.actuator_gear[actuator, 0])
                lower[actuator], upper[actuator] = max(lower[actuator], bounds[0]), min(upper[actuator], bounds[1])
        substeps = trajectory['substeps']
        count = (trajectory['steps_per_env'] + substeps - 1) // substeps
        rng = np.random.default_rng(trajectory['action_seed'])
        sequence = np.clip(base + rng.uniform(-1, 1, (count, m.nu)) * amplitudes, lower, upper)
        checksum = hashlib.sha256(sequence.tobytes()).hexdigest()
        if checksum != controls['sequence_sha256']:
            raise ValueError('Reconstructed control sequence checksum does not match benchmark')
        report.update(control_sequence_sha256=checksum, control_sequence=sequence.tolist(),
                      substeps=substeps, native_warmup_steps=trajectory['native_warmup_steps'],
                      action_seed=trajectory['action_seed'],
                      snapshot_control_semantics='At action boundaries use the next action; '
                        'after the final requested step retain the last action.')
        native = mujoco.MjData(m)
        native.ctrl[limited] = np.clip(native.ctrl[limited], m.actuator_ctrlrange[limited, 0],
                                      m.actuator_ctrlrange[limited, 1])
        for _ in range(trajectory['native_warmup_steps']):
            mujoco.mj_step(m, native)
        mujoco.mj_forward(m, native)
        native.qacc_warmstart[:] = 0
        np.testing.assert_allclose(native.qpos, env['initial_qpos'], rtol=0, atol=1e-12)
        fields = ('qpos', 'qvel', 'act', 'ctrl', 'qfrc_applied', 'xfrc_applied',
                  'mocap_pos', 'mocap_quat', 'eq_active', 'userdata', 'history', 'plugin_state')
        report['transferred_integration_fields'] = [*fields, 'time', 'qacc_warmstart=zeros']

        def difference(actual, expected):
            actual, expected = np.asarray(actual), np.asarray(expected)
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise AssertionError('Non-finite dynamics output')
            delta = actual - expected
            return dict(shape=list(delta.shape), max_abs=float(np.max(np.abs(delta), initial=0)),
                        mlx_norm=float(np.linalg.norm(actual)), native_norm=float(np.linalg.norm(expected)),
                        difference_norm=float(np.linalg.norm(delta)),
                        max_abs_index=[int(i) for i in np.unravel_index(np.argmax(np.abs(delta)), delta.shape)] if delta.size else [],
                        mlx_values=actual.tolist(), native_values=expected.tolist())

        def active_contacts(contact):
            distances = np.asarray(contact.dist)
            margins = np.asarray(contact.includemargin)
            positions = np.asarray(contact.pos)
            frames = np.asarray(contact.frame).reshape((-1, 3, 3))
            geoms = np.asarray(contact.geom)
            if not all(np.isfinite(value).all() for value in (distances, margins, positions, frames)):
                raise AssertionError('Non-finite contact output')
            groups = {}
            for index in np.flatnonzero(distances < margins):
                g1, g2 = map(int, geoms[index])
                pair = sorted((g1, g2))
                key = f'{pair[0]}:{pair[1]}'
                group = groups.setdefault(key, dict(geom_ids=pair,
                    geom_names=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geom) for geom in pair],
                    count=0, contacts=[]))
                group['count'] += 1
                group['contacts'].append(dict(slot=int(index), ordered_geom_ids=[g1, g2],
                    dist=float(distances[index]), includemargin=float(margins[index]),
                    pos=positions[index].tolist(), normal=frames[index, 0].tolist(),
                    normal_canonical_pair=(frames[index, 0] * (1 if g1 <= g2 else -1)).tolist()))
            return dict(total_slots=int(distances.size), active_count=int(np.sum(distances < margins)),
                        geometry_pairs=groups)

        for step in range(51):
            if step < 50:
                native.ctrl[:] = sequence[step // substeps]
            if step in (0, 25, 50):
                report['phase'] = f'snapshot_{step}'
                save(args.output, report)
                ref = mujoco.MjData(m)
                for field in fields:
                    getattr(ref, field)[:] = getattr(native, field)
                ref.time = native.time
                ref.qacc_warmstart[:] = 0
                if not all(np.isfinite(getattr(ref, field)).all() for field in fields):
                    raise AssertionError('Non-finite native snapshot integration state')
                d = io.make_data(m).replace(**{field: mx.array(getattr(ref, field)) for field in fields},
                    time=mx.array(ref.time), qacc_warmstart=mx.zeros((m.nv,)))
                mujoco.mj_forward(m, ref)
                snapshot_start = time.perf_counter()
                d = forward.forward(x, d)
                mx.eval(d.qacc, d.qfrc_constraint)
                forward_seconds = time.perf_counter() - snapshot_start
                native_mass = np.zeros((m.nv, m.nv))
                mujoco.mj_fullM(m, native_mass, ref.qM)
                comparisons = {'qM': difference(support.full_m(x, d), native_mass)}
                for field in ('qfrc_bias', 'qfrc_passive', 'qfrc_actuator', 'qfrc_constraint', 'qacc'):
                    comparisons[field] = difference(getattr(d, field), getattr(ref, field))
                native_contacts = active_contacts(ref.contact)
                mlx_contacts = active_contacts(d._impl.contact)
                pair_keys = sorted(set(native_contacts['geometry_pairs']) | set(mlx_contacts['geometry_pairs']))
                report['snapshots'].append(dict(native_physical_steps=step, simulation_time=float(ref.time),
                    mlx_forward_seconds=forward_seconds, finite=True,
                    input_qpos_max_rounding=float(np.max(np.abs(np.asarray(d.qpos) - ref.qpos), initial=0)),
                    input_qvel_max_rounding=float(np.max(np.abs(np.asarray(d.qvel) - ref.qvel), initial=0)),
                    dynamics=comparisons, native_contacts=native_contacts, mlx_contacts=mlx_contacts,
                    pair_count_comparison={key: dict(native=native_contacts['geometry_pairs'].get(key, {}).get('count', 0),
                                                     mlx=mlx_contacts['geometry_pairs'].get(key, {}).get('count', 0))
                                           for key in pair_keys}))
                save(args.output, report)
            if step < 50:
                mujoco.mj_step(m, native)
        if trajectory['steps_per_env'] == 50 and env.get('final_native_qpos') is not None:
            np.testing.assert_allclose(native.qpos, env['final_native_qpos'], rtol=0, atol=1e-12)
            report['native_final_replay_verified'] = True
        report.update(status='completed', phase='finished')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
    finally:
        report['elapsed_worker_seconds'] = time.perf_counter() - started
        save(args.output, report)
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    args = arguments()
    sys.exit(worker(args) if args._worker else supervisor(args))
