"""Bounded, staged MLX reproducer; exit nonzero on failure, write JSON evidence."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback

import mujoco
import numpy as np
mujoco.__path__.append(str(Path(__file__).resolve().parent))
import mlx.core as mx
from mujoco.mjx_mlx._src import io, forward, constraint, solver, collision_driver

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--xml', type=Path, help='Optional external MJCF (e.g. LEAP cube scene).')
p.add_argument('--steps', type=int, default=3)
p.add_argument('--timeout', type=int, default=45)
p.add_argument('--position-only', action='store_true')
p.add_argument('--native-warmup',type=int,default=0,help='Initialize both backends from a native contact state.')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
report = dict(steps_requested=a.steps, steps_completed=0, events=[],
    python=platform.python_version(), mujoco=mujoco.__version__,
    mlx=importlib.metadata.version('mlx'), device=str(mx.default_device()),
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()))

def wrap(module, name, fields):
    original = getattr(module,name)
    def measured(*args, **kwargs):
        t = time.perf_counter()
        row = dict(stage=name, status='running')
        report['events'].append(row)
        try:
            d = original(*args, **kwargs)
            impl = d._impl or d
            if name == 'collision':
                mx.eval(impl.contact.dist,impl.contact.pos,impl.contact.frame)
            mx.eval(*(getattr(impl,f) if hasattr(impl,f) else getattr(d,f) for f in fields))
            row.update(status='ok', seconds=time.perf_counter()-t)
            if name == 'make_constraint':
                row.update(rows=int(impl.efc_J.shape[0]), contacts=int(impl.contact.dist.size),
                    metadata_contacts=int(impl.contact.dim.size),
                    active_contacts=int(np.sum(np.asarray(impl.contact.dist)<np.asarray(impl.contact.includemargin))))
            return d
        except BaseException:
            row.update(status='failed',seconds=time.perf_counter()-t)
            raise
    setattr(module,name,measured)

wrap(constraint,'make_constraint',['efc_J','efc_D','efc_aref'])
wrap(solver,'solve',['qacc','qfrc_constraint'])
wrap(collision_driver,'collision',[])  # Constraint timing forces remaining contact evaluation.

def timeout(*_):
    raise TimeoutError('Configured wall-time budget exceeded')
signal.signal(signal.SIGALRM,timeout)
signal.alarm(a.timeout)
start = time.perf_counter()
try:
    if a.xml:
        report['model'] = a.xml.name
        report['xml_sha256'] = hashlib.sha256(a.xml.read_bytes()).hexdigest()
        m = mujoco.MjModel.from_xml_path(str(a.xml.resolve()))
    else:
        # Multiple collision groups reproduce the static metadata bug without assets.
        xml = '<mujoco><option cone="elliptic" integrator="implicitfast"/><worldbody><geom type="plane" size="1 1 .1"/>'
        for i in range(12):
            xml += f'<body pos="{i*.1} 0 .03"><freejoint/><geom type="box" size=".02 .02 .02"/></body>'
        xml += '</worldbody></mujoco>'
        m = mujoco.MjModel.from_xml_string(xml)
        report['model'] = '12 free boxes + plane (inline)'
    x,d,native = io.put_model(m),io.make_data(m),mujoco.MjData(m)
    for _ in range(a.native_warmup):
        mujoco.mj_step(m,native)
    if a.native_warmup:
        d = d.replace(qpos=mx.array(native.qpos),qvel=mx.array(native.qvel),
                      act=mx.array(native.act),ctrl=mx.array(native.ctrl),time=mx.array(native.time))
        native.qacc_warmstart[:] = 0
    report['native_warmup_steps'] = a.native_warmup
    max_error = 0.
    for i in range(a.steps):
        d = forward.fwd_position(x,d) if a.position_only else forward.step(x,d)
        mx.eval(d.qpos,d.qvel)
        if not np.isfinite(np.asarray(d.qpos)).all() or not np.isfinite(np.asarray(d.qvel)).all():
            raise AssertionError('Non-finite state')
        report['steps_completed'] = i+1
        if not a.position_only:
            mujoco.mj_step(m,native)
            max_error = max(max_error,float(np.max(np.abs(np.asarray(d.qpos)-native.qpos))))
    report.update(status='completed',max_qpos_error_mixed_units=max_error,
                  qpos_mlx=np.asarray(d.qpos).tolist(),qpos_native=native.qpos.tolist())
except BaseException:
    report.update(status='failed',error=traceback.format_exc().replace(str(root),'$REPO'))
finally:
    signal.alarm(0)
    report['elapsed_s'] = time.perf_counter()-start
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
sys.exit(0 if report['status']=='completed' else 1)
