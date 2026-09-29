"""Bounded serial physics benchmark; not a batched GPU or PPO benchmark.

Example (from the repository root):
  python mjx/mujoco/benchmark_hand_system.py --xml /path/to/cube_scene.xml \
    --backend compare --native-warmup 100 --steps 25 --substeps 25 \
    --timeout 180 --output /tmp/hand-comparison.json

--steps counts physical steps PER environment. --substeps is the number of
physical steps holding each control action. Multiple environments run serially,
not concurrently. Timings include an explicit MLX state synchronization every
physical step; inspection, control transfer, and native-reference costs are
reported separately. No rendering, reward, policy inference, or learning runs.
The parent process enforces the deadline even if a device call stops responding.
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


def arguments():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--xml', type=Path, required=True)
  p.add_argument('--backend', choices=('native', 'mlx', 'compare'), default='compare')
  p.add_argument('--steps', type=int, default=25)
  p.add_argument('--num-envs', type=int, default=1)
  p.add_argument('--substeps', type=int, default=25)
  p.add_argument('--native-warmup', type=int, default=0)
  p.add_argument('--action-seed', type=int, default=0)
  p.add_argument('--action-amplitude', type=float, default=0.02,
                 help='Fraction of bounded actuator range; also capped by --max-control-offset.')
  p.add_argument('--max-control-offset', type=float, default=0.05,
                 help='Maximum offset in actuator-length units, not always joint radians.')
  p.add_argument('--max-joint-error-rad', type=float, default=0.05)
  p.add_argument('--max-slide-error-m', type=float, default=0.001)
  p.add_argument('--max-free-translation-error-m', type=float, default=0.001)
  p.add_argument('--max-quaternion-angle-error-rad', type=float, default=0.02)
  p.add_argument('--timeout', type=float, default=180)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
  a = p.parse_args()
  for name in ('steps', 'num_envs', 'substeps'):
    if getattr(a, name) < 1:
      p.error(f'--{name.replace("_", "-")} must be positive')
  if a.native_warmup < 0 or a.action_seed < 0 or a.timeout <= 0:
    p.error('warmup and seed must be nonnegative, timeout must be positive')
  for name in ('action_amplitude', 'max_control_offset', 'max_joint_error_rad',
               'max_slide_error_m', 'max_free_translation_error_m',
               'max_quaternion_angle_error_rad'):
    if not math.isfinite(getattr(a, name)) or getattr(a, name) < 0:
      p.error(f'--{name.replace("_", "-")} must be finite and nonnegative')
  if a.action_amplitude > 0.1 or not math.isfinite(a.timeout):
    p.error('action amplitude must be <= 0.1 and timeout must be finite')
  return a


def save(output, report):
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_name(output.name + '.partial')
  temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
  temporary.replace(output)


def supervisor(a):
  started = time.perf_counter()
  initial = dict(status='starting', execution='serial', backend=a.backend,
                 environments=[], timeout_seconds=a.timeout)
  save(a.output, initial)
  command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--_worker']
  child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, start_new_session=True)
  timed_out = False
  try:
    stdout, stderr = child.communicate(timeout=a.timeout)
  except subprocess.TimeoutExpired:
    timed_out = True
    os.killpg(child.pid, signal.SIGTERM)
    try:
      stdout, stderr = child.communicate(timeout=2)
    except subprocess.TimeoutExpired:
      os.killpg(child.pid, signal.SIGKILL)
      stdout, stderr = child.communicate(timeout=5)
  try:
    report = json.loads(a.output.read_text())
  except (OSError, ValueError):
    report = initial
  if timed_out:
    report.update(status='failed', error='Hard wall-time deadline exceeded', timed_out=True)
  elif child.returncode != 0 or report.get('status') != 'completed':
    report.update(status='failed')
    report.setdefault('error', 'Worker exited without a successful report')
  report.update(worker_returncode=child.returncode,
                wall_seconds_including_startup=time.perf_counter() - started)
  if stdout:
    report['worker_stdout_tail'] = stdout[-4000:]
  if stderr:
    report['worker_stderr_tail'] = stderr[-8000:]
  save(a.output, report)
  print(json.dumps(dict(status=report['status'], output=str(a.output.resolve()),
                        wall_seconds=report['wall_seconds_including_startup'])))
  return 0 if report['status'] == 'completed' else 1


def worker(a):
  started = time.perf_counter()
  report = dict(status='running', phase='imports', execution='serial',
                backend=a.backend, num_envs=a.num_envs, steps_per_env=a.steps,
                substeps=a.substeps, native_warmup_steps=a.native_warmup,
                action_seed=a.action_seed, environments=[], resident_envs=1,
                MLX_ENABLE_TF32=os.environ.get('MLX_ENABLE_TF32', '1 (MLX default)'),
                action_amplitude=a.action_amplitude,
                max_control_offset=a.max_control_offset,
                methodology='Instrumented serial simulation, no rendering or PPO. '
                  'Every physical step is synchronized and inspected. MLX mode '
                  'alone has no native accuracy reference; compare advances one '
                  'CPU reference immediately after every MLX physical step.',
                limitations=['No cross-environment batching or concurrency',
                             'XML hash does not hash included XML or mesh assets',
                             'Contacts describe the dynamics evaluation before integration',
                             'Native warmup is initial-state preparation, not device warmup',
                             'num_envs means serial replicas; only one live environment state is retained',
                             'First MLX step may include deferred model conversion or kernel initialization'])
  save(a.output, report)
  mx = None
  try:
    import mujoco
    import numpy as np
    repository = Path(__file__).resolve().parents[2]
    report.update(python=platform.python_version(), platform=platform.platform(),
                  python_executable=sys.executable, mujoco=mujoco.__version__,
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                    cwd=repository, text=True).strip(),
                  dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'],
                    cwd=repository, text=True).strip()),
                  model=str(a.xml.resolve()),
                  xml_sha256=hashlib.sha256(a.xml.read_bytes()).hexdigest())
    if a.backend != 'native':
      import mlx.core as mx
      mujoco.__path__.insert(0, str(Path(__file__).resolve().parent))
      from mujoco.mjx_mlx._src import forward, io
      mx.set_default_device(mx.gpu)
      report.update(mlx=importlib.metadata.version('mlx'), device=str(mx.default_device()))
    else:
      report['device'] = 'native MuJoCo CPU'
    report['phase'] = 'model_setup'
    setup_started = time.perf_counter()
    m = mujoco.MjModel.from_xml_path(str(a.xml.resolve()))
    x = io.put_model(m) if mx is not None else None
    report.update(model_setup_seconds=time.perf_counter() - setup_started,
                  model_dimensions=dict(nq=m.nq, nv=m.nv, nu=m.nu, ngeom=m.ngeom),
                  timestep_seconds=float(m.opt.timestep))
    thresholds = dict(joint_rad=a.max_joint_error_rad, slide_m=a.max_slide_error_m,
                      free_translation_m=a.max_free_translation_error_m,
                      quaternion_angle_rad=a.max_quaternion_angle_error_rad)
    report['accuracy_thresholds'] = thresholds

    def statistics(samples):
      values = np.asarray(samples, dtype=float)
      return dict(count=len(samples), total_seconds=float(values.sum()),
                  mean_seconds=float(values.mean()), min_seconds=float(values.min()),
                  median_seconds=float(np.median(values)),
                  p95_seconds=float(np.percentile(values, 95)),
                  max_seconds=float(values.max())) if len(values) else dict(count=0)

    def errors(qpos, reference):
      result = dict.fromkeys(thresholds, 0.0)
      for kind, adr in zip(m.jnt_type, m.jnt_qposadr):
        if kind == mujoco.mjtJoint.mjJNT_HINGE:
          result['joint_rad'] = max(result['joint_rad'], float(abs(qpos[adr] - reference[adr])))
        elif kind == mujoco.mjtJoint.mjJNT_SLIDE:
          result['slide_m'] = max(result['slide_m'], float(abs(qpos[adr] - reference[adr])))
        elif kind in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
          if kind == mujoco.mjtJoint.mjJNT_FREE:
            result['free_translation_m'] = max(result['free_translation_m'],
                float(np.linalg.norm(qpos[adr:adr+3] - reference[adr:adr+3])))
            adr += 3
          qa, qb = qpos[adr:adr+4], reference[adr:adr+4]
          denominator = np.linalg.norm(qa) * np.linalg.norm(qb)
          if denominator < 1e-12:
            raise AssertionError('Zero-length quaternion')
          angle = 2 * np.arccos(np.clip(abs(np.dot(qa, qb)) / denominator, 0, 1))
          result['quaternion_angle_rad'] = max(result['quaternion_angle_rad'], float(angle))
      return result

    def controls(d, seed):
      # Only recognized scalar-joint position servos are perturbed. Other
      # actuators retain their bounded initial controls; no torque assumptions.
      base = d.ctrl.copy()
      lower = np.full(m.nu, -np.inf)
      upper = np.full(m.nu, np.inf)
      amplitudes = np.zeros(m.nu)
      selected = []
      for i in range(m.nu):
        if m.actuator_ctrllimited[i]:
          lower[i], upper[i] = m.actuator_ctrlrange[i]
        joint = int(m.actuator_trnid[i, 0])
        kp = m.actuator_gainprm[i, 0]
        position = (m.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED
                    and m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE
                    and kp > 0 and np.isclose(m.actuator_biasprm[i, 1], -kp)
                    and m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
                    and m.jnt_type[joint] in (mujoco.mjtJoint.mjJNT_HINGE,
                                              mujoco.mjtJoint.mjJNT_SLIDE))
        if not position:
          continue
        if m.jnt_limited[joint]:
          joint_bounds = np.sort(m.jnt_range[joint] * m.actuator_gear[i, 0])
          lower[i], upper[i] = max(lower[i], joint_bounds[0]), min(upper[i], joint_bounds[1])
        if lower[i] > upper[i]:
          raise ValueError(f'Actuator {i} has incompatible control and joint bounds')
        base[i] = np.clip(d.actuator_length[i], lower[i], upper[i])
        width = upper[i] - lower[i]
        amplitudes[i] = min(a.max_control_offset, a.action_amplitude * width) if np.isfinite(width) else a.max_control_offset
        if a.action_amplitude == 0:
          amplitudes[i] = 0
        selected.append(i)
      base = np.clip(base, lower, upper)
      rng = np.random.default_rng(seed)
      sequence = np.clip(base + rng.uniform(-1, 1, ((a.steps + a.substeps - 1) // a.substeps, m.nu))
                         * amplitudes, lower, upper)
      return sequence, dict(position_actuator_ids=selected, base=base.tolist(),
                            maximum_offsets=amplitudes.tolist(),
                            sequence_sha256=hashlib.sha256(sequence.tobytes()).hexdigest())

    for env_index in range(a.num_envs):
      row = dict(environment=env_index, steps_completed=0, finite=True,
                 reference_finite=True if a.backend == 'compare' else None,
                 accuracy_max=dict.fromkeys(thresholds, 0.0) if a.backend == 'compare' else None)
      report['environments'].append(row)
      report['phase'] = f'environment_{env_index}_setup'
      save(a.output, report)
      setup_started = time.perf_counter()
      native = mujoco.MjData(m)
      limited = np.asarray(m.actuator_ctrllimited, dtype=bool)
      native.ctrl[limited] = np.clip(native.ctrl[limited], m.actuator_ctrlrange[limited, 0], m.actuator_ctrlrange[limited, 1])
      warmup_started = time.perf_counter()
      for _ in range(a.native_warmup):
        mujoco.mj_step(m, native)
      row['native_warmup_seconds'] = time.perf_counter() - warmup_started
      mujoco.mj_forward(m, native)
      native.qacc_warmstart[:] = 0
      sequence, row['controls'] = controls(native, a.action_seed + env_index)
      if mx is not None:
        d = io.make_data(m)
        # Transfer integration state, not backend-specific contact row storage.
        fields = ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart', 'qfrc_applied',
                  'xfrc_applied', 'mocap_pos', 'mocap_quat', 'eq_active', 'userdata')
        d = d.replace(**{name: mx.array(getattr(native, name)) for name in fields},
                      time=mx.array(native.time))
        mx.eval(d.qpos, d.qvel, d.act, d.ctrl)
      row['setup_seconds'] = time.perf_counter() - setup_started
      row['initial_time_seconds'] = float(native.time)
      row['initial_qpos'] = native.qpos.tolist()
      step_times, reference_times, control_times, inspection_times = [], [], [], []
      active_contacts, contact_slots, reference_contacts = [], [], []
      env_started, last_checkpoint = time.perf_counter(), time.perf_counter()
      report['phase'] = f'environment_{env_index}_steps'
      save(a.output, report)
      for step in range(a.steps):
        if step % a.substeps == 0:
          t = time.perf_counter()
          ctrl = sequence[step // a.substeps]
          native.ctrl[:] = ctrl
          if mx is not None:
            d = d.replace(ctrl=mx.array(ctrl))
            mx.eval(d.ctrl)
          control_times.append(time.perf_counter() - t)
        t = time.perf_counter()
        if mx is not None:
          d = forward.step(x, d)
          mx.eval(d.qpos, d.qvel, d.act, d.qacc)
        else:
          mujoco.mj_step(m, native)
        step_times.append(time.perf_counter() - t)
        if a.backend == 'compare':
          t = time.perf_counter()
          mujoco.mj_step(m, native)
          reference_times.append(time.perf_counter() - t)
        t = time.perf_counter()
        current = d if mx is not None else native
        row['finite'] = all(np.isfinite(np.asarray(getattr(current, f))).all()
                            for f in ('qpos', 'qvel', 'act', 'qacc', 'ctrl'))
        if not row['finite']:
          raise AssertionError('Non-finite simulated state')
        expected_time = row['initial_time_seconds'] + (step + 1) * m.opt.timestep
        if abs(float(current.time) - expected_time) > max(1e-5, expected_time * 1e-4):
          raise AssertionError('Simulation time reset or stopped advancing')
        if mx is not None:
          impl = current._impl if current._impl is not None else current
          contact = impl.contact
          distances = np.asarray(contact.dist)
          if not np.isfinite(distances).all():
            raise AssertionError('Non-finite contact distances')
          active_contacts.append(int(np.sum(distances < np.asarray(contact.includemargin))))
          contact_slots.append(int(distances.size))
        else:
          active_contacts.append(int(np.sum(native.contact.dist < native.contact.includemargin)))
          contact_slots.append(int(native.ncon))
        if a.backend == 'compare':
          row['reference_finite'] = all(np.isfinite(getattr(native, f)).all()
                                      for f in ('qpos', 'qvel', 'act', 'qacc', 'ctrl'))
          if not row['reference_finite']:
            raise AssertionError('Non-finite native reference')
          current_errors = errors(np.asarray(d.qpos, dtype=float), native.qpos)
          row['accuracy_max'] = {k: max(row['accuracy_max'][k], v) for k, v in current_errors.items()}
          reference_contacts.append(int(np.sum(native.contact.dist < native.contact.includemargin)))
        warnings = {str(w): int(native.warning[w].number) for w in
                    (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                     mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL)}
        if any(warnings.values()):
          raise AssertionError(f'Native instability warnings: {warnings}')
        inspection_times.append(time.perf_counter() - t)
        row['steps_completed'] = step + 1
        if time.perf_counter() - last_checkpoint > 1:
          row['step_timing'] = statistics(step_times)
          save(a.output, report)
          last_checkpoint = time.perf_counter()
      row.update(loop_wall_seconds=time.perf_counter() - env_started,
                 step_timing=statistics(step_times), control_timing=statistics(control_times),
                 first_step_seconds=step_times[0], steady_step_timing=statistics(step_times[1:]),
                 reference_step_timing=statistics(reference_times), inspection_timing=statistics(inspection_times),
                 physical_steps_per_second=a.steps / sum(step_times),
                 action_equivalents_per_second=a.steps / a.substeps / sum(step_times),
                 final_qpos=np.asarray(current.qpos).tolist())
      for name, values in (('active_contacts', active_contacts), ('contact_slots', contact_slots),
                           ('reference_active_contacts', reference_contacts)):
        row[name] = dict(min=min(values), max=max(values), final=values[-1]) if values else None
      if a.backend == 'compare':
        row['accuracy_passed'] = all(row['accuracy_max'][k] <= v for k, v in thresholds.items())
        row['final_native_qpos'] = native.qpos.tolist()
      save(a.output, report)
    simulated_steps = sum(r['steps_completed'] for r in report['environments'])
    physics_seconds = sum(r['step_timing']['total_seconds'] for r in report['environments'])
    loop_seconds = sum(r['loop_wall_seconds'] for r in report['environments'])
    report.update(total_environment_physical_steps=simulated_steps,
                  serial_physics_steps_per_second=simulated_steps / physics_seconds,
                  serial_action_equivalents_per_second=simulated_steps / a.substeps / physics_seconds,
                  instrumented_loop_steps_per_second=simulated_steps / loop_seconds,
                  instrumented_loop_action_equivalents_per_second=simulated_steps / a.substeps / loop_seconds,
                  status='completed', phase='finished')
    if a.backend == 'compare' and not all(r['accuracy_passed'] for r in report['environments']):
      raise AssertionError('Configured trajectory accuracy thresholds exceeded (see accuracy_max)')
  except BaseException:
    report.update(status='failed', error=traceback.format_exc())
  finally:
    report['elapsed_worker_seconds'] = time.perf_counter() - started
    try:
      import resource
      peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
      report['process_peak_rss_bytes'] = int(peak if sys.platform == 'darwin' else peak * 1024)
    except (ImportError, OSError):
      report['process_peak_rss_bytes'] = None
    report['mlx_peak_memory_bytes'] = None
    if mx is not None:
      for namespace in (mx, getattr(mx, 'metal', None)):
        method = getattr(namespace, 'get_peak_memory', None)
        if callable(method):
          try:
            report['mlx_peak_memory_bytes'] = int(method())
          except Exception as exc:
            report['mlx_memory_query_error'] = str(exc)
          break
    save(a.output, report)
  return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
  args = arguments()
  sys.exit(worker(args) if args._worker else supervisor(args))
