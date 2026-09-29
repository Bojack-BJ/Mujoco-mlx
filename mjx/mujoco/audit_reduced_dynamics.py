"""Three-way same-state hand dynamics, with only the observed contact pairs.

Keep all hand bodies, joints, actuators and inertia. Recompile collision masks
for pairs 33:89 and 40:89 so CPU JAX can evaluate a small fixed contact buffer.
Verify native restricted dynamics against the unmodified native model first.
This is an attribution diagnostic, not a training-backend acceptance test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from audit_contact_parity import supervise
from benchmark_hand_system import save


def worker(a):
    report=dict(status='running',snapshots=[],diagnostic_only=True,
                MLX_ENABLE_TF32=os.environ.get('MLX_ENABLE_TF32'),
                scope=__doc__,phase='imports')
    save(a.output,report)
    try:
        import numpy as np
        import mujoco
        from mujoco import mjx
        from mujoco.mjx._src import support as js
        import jax
        import jax.numpy as jp
        mujoco.__path__.insert(0,str(Path(__file__).resolve().parent))
        from mujoco.mjx_mlx._src import io,forward,support
        import mlx.core as mx
        if os.environ.get('MLX_ENABLE_TF32') != '0':
            raise ValueError('Launch with MLX_ENABLE_TF32=0')
        mx.set_default_device(mx.gpu)
        jax.config.update('jax_default_matmul_precision','highest')
        root=Path(__file__).resolve().parents[2]
        original=mujoco.MjModel.from_xml_path(str(a.xml.resolve()))
        spec=mujoco.MjSpec.from_file(str(a.xml.resolve()))
        assert original.npair == 0, 'Explicit contact pairs need separate handling'
        selected={33,40,89}
        for i,geom in enumerate(spec.geoms):
            geom.contype=geom.conaffinity=0
            if i in (33,40):
                geom.contype,geom.conaffinity=1,2
            elif i == 89:
                geom.contype,geom.conaffinity=2,1
        m=spec.compile()
        assert m.nv==original.nv and m.nq==original.nq and m.nu==original.nu
        for i in selected:
            assert mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i)==mujoco.mj_id2name(original,mujoco.mjtObj.mjOBJ_GEOM,i)
        report.update(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                      mujoco=mujoco.__version__,jax=jax.__version__,official_mjx_path=mjx.__file__,
                      jax_devices=[str(d) for d in jax.devices()],mlx_device=str(mx.default_device()),
                      dimensions=dict(nq=m.nq,nv=m.nv,nu=m.nu),
                      source_hashes={f:hashlib.sha256((root/'mjx/mujoco/mjx_mlx/_src'/f).read_bytes()).hexdigest()
                                     for f in ('collision_convex.py','constraint.py','solver.py','forward.py','smooth.py')})
        jmodel= mjx.put_model(m)
        mmodel=io.put_model(m)
        jforward=jax.jit(mjx.forward)

        def diff(actual,expected):
            actual,expected=np.asarray(actual),np.asarray(expected)
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise AssertionError('Nonfinite dynamics')
            if actual.shape!=expected.shape:
                return dict(shape_mismatch=[list(actual.shape),list(expected.shape)])
            delta=actual-expected
            return dict(max_abs=float(np.max(np.abs(delta),initial=0)),
                        difference_norm=float(np.linalg.norm(delta)),reference_norm=float(np.linalg.norm(expected)),
                        actual_values=actual.tolist(),reference_values=expected.tolist())

        def contacts(c):
            ids=np.flatnonzero(np.asarray(c.dist)<np.asarray(c.includemargin))
            return dict(slots=int(c.dist.size),active_count=len(ids),
                        geom=np.asarray(c.geom)[ids].tolist(),dist=np.asarray(c.dist)[ids].tolist(),
                        pos=np.asarray(c.pos)[ids].tolist())

        states=json.loads(a.snapshots.read_text())['native_states']
        for state in states:
            row=dict(step=state['step'])
            report['snapshots'].append(row)
            refs=[]
            for model in (original,m):
                ref=mujoco.MjData(model)
                for f in ('qpos','qvel','ctrl'):
                    getattr(ref,f)[:]=state[f]
                ref.time=state['time']
                ref.qacc_warmstart[:]=0
                mujoco.mj_forward(model,ref)
                refs.append(ref)
            native=refs[1]
            fields=('qfrc_bias','qfrc_passive','qfrc_actuator','qfrc_constraint','qacc')
            row['native_restricted_vs_original']={f:diff(getattr(native,f),getattr(refs[0],f)) for f in fields}
            if any(v['max_abs']>1e-7 for v in row['native_restricted_vs_original'].values()):
                raise AssertionError('Restricted contacts change native dynamics at this snapshot')
            jd=mjx.make_data(m).replace(**{f:jp.array(state[f],dtype=jp.float32) for f in ('qpos','qvel','ctrl')},
                                       time=jp.array(state['time'],dtype=jp.float32))
            md=io.make_data(m).replace(**{f:mx.array(state[f]) for f in ('qpos','qvel','ctrl')},time=mx.array(state['time']))
            report['phase']=f'jax_forward_{state["step"]}'
            save(a.output,report)
            start=time.perf_counter();jd=jforward(jmodel,jd);jax.block_until_ready(jd)
            row['jax_seconds_including_first_compile']=time.perf_counter()-start
            report['phase']=f'mlx_forward_{state["step"]}'
            save(a.output,report)
            start=time.perf_counter();md=forward.forward(mmodel,md);mx.eval(md.qacc,md.qfrc_constraint)
            row['mlx_forward_seconds']=time.perf_counter()-start
            row['native_contacts']=contacts(native.contact)
            row['jax_contacts']=contacts(jd._impl.contact)
            row['mlx_contacts']=contacts(md._impl.contact)
            native_mass=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,native_mass,native.qM)
            comparisons={f:(getattr(jd,f),getattr(md,f),getattr(native,f)) for f in fields}
            comparisons['qM']=(js.full_m(jmodel,jd),support.full_m(mmodel,md),native_mass)
            row['dynamics']={f:dict(mlx_vs_jax=diff(v[1],v[0]),jax_vs_native=diff(v[0],v[2]),
                                    mlx_vs_native=diff(v[1],v[2])) for f,v in comparisons.items()}
            row['constraint_arrays_mlx_vs_jax']={f:diff(getattr(md._impl,f),getattr(jd._impl,f))
                                                for f in ('efc_J','efc_D','efc_aref','efc_force')}
            row['solver_niter']=dict(jax=int(jd._impl.solver_niter),mlx=int(md._impl.solver_niter))
            save(a.output,report)
        report.update(status='completed',phase='finished')
    except BaseException:
        report.update(status='failed',error=traceback.format_exc())
    save(a.output,report)
    return report['status']!='completed'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xml',type=Path,required=True)
    p.add_argument('--snapshots',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=float,default=180)
    p.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    sys.exit(worker(a) if a._worker else supervise(a,Path(__file__).resolve()))
