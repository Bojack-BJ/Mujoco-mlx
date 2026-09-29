"""Extract asset-free box cases and compare native, official JAX and MLX.

This is a bounded diagnostic, not an assertion that contact algorithms coincide.
Only this worker process temporarily changes a penetration predicate for an
ablation; no installed/source implementation is modified by that experiment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from benchmark_hand_system import save


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xml', type=Path, required=True)
    p.add_argument('--trajectory-json', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.timeout <= 0:
        p.error('timeout must be positive')
    return a


def supervise(a, script=None):
    start = time.perf_counter()
    save(a.output, dict(status='starting', diagnostic_only=True))
    proc = subprocess.Popen([sys.executable, str(script or Path(__file__).resolve()),
                             *sys.argv[1:], '--_worker'], start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    timeout = False
    try:
        out, err = proc.communicate(timeout=a.timeout)
    except subprocess.TimeoutExpired:
        timeout = True
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
    report = json.loads(a.output.read_text())
    if timeout or proc.returncode or report.get('status') != 'completed':
        report.update(status='failed', timed_out=timeout)
    report.update(worker_returncode=proc.returncode, stdout_tail=out[-2000:],
                  stderr_tail=err[-4000:], wall_seconds=time.perf_counter()-start)
    save(a.output, report)
    print(json.dumps(dict(status=report['status'], output=str(a.output))))
    return report['status'] != 'completed'


def worker(a):
    report = dict(status='running', cases=[], diagnostic_only=True,
                  MLX_ENABLE_TF32=os.environ.get('MLX_ENABLE_TF32'),
                  scope='Two-box collision only: no original hand dynamics or integration. '
                  'Identical float32 world geometry for JAX/MLX. Native retains float64 XML geometry.',
                  ablation='Temporary MLX strict penetration predicate >0 instead of >1e-6; '
                  'not a committed physics change and not evidence of general superiority.')
    save(a.output, report)
    try:
        import numpy as np
        import mujoco
        from mujoco import mjx
        from mujoco.mjx._src import collision_convex as jc, mesh as jm
        from mujoco.mjx._src.collision_types import GeomInfo as JGeomInfo
        import jax
        import jax.numpy as jp
        # Load installed official MJX first, before exposing the sibling MLX tree.
        mujoco.__path__.insert(0, str(Path(__file__).resolve().parent))
        from mujoco.mjx_mlx._src import collision_convex as mc, mesh as mm
        from mujoco.mjx_mlx._src.collision_types import GeomInfo as MGeomInfo
        import mlx.core as mx
        mx.set_default_device(mx.gpu)
        jax.config.update('jax_default_matmul_precision', 'highest')
        root = Path(__file__).resolve().parents[2]
        report.update(mujoco=mujoco.__version__, jax=jax.__version__,
                      official_mjx_path=mjx.__file__, official_collision_path=jc.__file__,
                      jax_devices=[str(d) for d in jax.devices()], mlx_device=str(mx.default_device()),
                      revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                      source_hashes={name:hashlib.sha256(Path(path).read_bytes()).hexdigest()
                          for name,path in [('official_collision',jc.__file__),('mlx_collision',mc.__file__)]})
        if os.environ.get('MLX_ENABLE_TF32') != '0':
            raise ValueError('Launch a fresh process with MLX_ENABLE_TF32=0')

        def jkernel(p1,r1,s1,p2,r2,s2):
            # Official mesh.box expects a leading size batch. Its topology is
            # shared; only the generated vertex/face arrays need unbatching.
            def box(p,r,s):
                c=jm.box(JGeomInfo(p,r,s[None]))
                return c.replace(size=s,vert=c.vert[0],face=c.face[0])
            return jc._box_box(box(p1,r1,s1),box(p2,r2,s2))
        jkernel = jax.jit(jkernel)

        def mkernel(p1,r1,s1,p2,r2,s2):
            return mc._box_box(mm.box(MGeomInfo(p1,r1,s1)), mm.box(MGeomInfo(p2,r2,s2)))

        def result(dist, pos, normal):
            dist,pos,normal = map(np.asarray,(dist,pos,normal))
            if not all(np.isfinite(v).all() for v in (dist,pos,normal)):
                raise AssertionError('Nonfinite collision result')
            active = dist < 0
            return dict(dist=dist.tolist(), pos=pos.tolist(), normal=normal.tolist(),
                        active_count=int(active.sum()),
                        minimum_distance_m=float(dist.min()) if dist.size else None)

        fixtures = a.output.parent / 'fixtures'
        fixtures.mkdir(parents=True, exist_ok=True)
        vector = lambda v:' '.join(format(float(x),'.17g') for x in np.asarray(v).flat)
        cases = []
        trajectory = json.loads(a.trajectory_json.read_text())
        controls = trajectory['environments'][0]['controls']
        original = mujoco.MjModel.from_xml_path(str(a.xml.resolve()))
        d = mujoco.MjData(original)
        limited = original.actuator_ctrllimited.astype(bool)
        d.ctrl[limited] = np.clip(d.ctrl[limited], original.actuator_ctrlrange[limited,0],
                                 original.actuator_ctrlrange[limited,1])
        for _ in range(trajectory['native_warmup_steps']):
            mujoco.mj_step(original,d)
        mujoco.mj_forward(original,d)
        d.qacc_warmstart[:] = 0
        np.testing.assert_allclose(d.qpos,trajectory['environments'][0]['initial_qpos'],atol=1e-12,rtol=0)
        # The stored LEAP controls stay strictly within their limits in this audit.
        base, amp = np.array(controls['base']),np.array(controls['maximum_offsets'])
        rng = np.random.default_rng(trajectory['action_seed'])
        sequence = base + rng.uniform(-1,1,(2,original.nu))*amp
        if hashlib.sha256(sequence.tobytes()).hexdigest() != controls['sequence_sha256']:
            raise ValueError('Control reconstruction mismatch; this extractor requires the recorded two-action trace')
        for step in range(51):
            if step in (0,25,50):
                copy = mujoco.MjData(original)
                copy.qpos[:] = d.qpos
                mujoco.mj_forward(original,copy)
                report.setdefault('native_states',[]).append(dict(step=step,
                    qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),
                    ctrl=sequence[min(step//25,1)].tolist(),time=float(d.time)))
                for pair in ((33,89),(40,89)):
                    values=[]
                    for geom in pair:
                        assert original.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
                        values.extend((copy.geom_xpos[geom].copy(),copy.geom_xmat[geom].reshape(3,3).copy(),original.geom_size[geom].copy()))
                    cases.append((f'leap-step{step}-pair{pair[0]}-{pair[1]}',values,
                                  dict(native_step=step, original_geom_ids=list(pair))))
            if step < 50:
                d.ctrl[:] = sequence[step//25]
                mujoco.mj_step(original,d)
        np.testing.assert_allclose(d.qpos,trajectory['environments'][0]['final_native_qpos'],atol=1e-12,rtol=0)
        report['native_replay_verified'] = True
        for depth in (-2e-6,0.,.25e-6,.5e-6,.9e-6,1.1e-6,2e-6,10e-6,1e-3):
            cases.append((f'axis-depth-{depth:.3g}',[np.zeros(3),np.eye(3),np.full(3,.025),
                          np.array([0.,0.,.05-depth]),np.eye(3),np.full(3,.025)],
                          dict(nominal_penetration_m=depth)))
        for name, values, metadata in cases:
            quats=[]
            for rotation in (values[1],values[4]):
                quat=np.zeros(4);mujoco.mju_mat2Quat(quat,rotation.reshape(-1));quats.append(quat)
            xml = ('<mujoco model="contact-parity"><option cone="elliptic"/>'
                   '<worldbody><geom name="fixed_box" type="box" '
                   f'pos="{vector(values[0])}" quat="{vector(quats[0])}" size="{vector(values[2])}"/>'
                   f'<body pos="{vector(values[3])}" quat="{vector(quats[1])}"><freejoint/>'
                   f'<geom name="free_box" type="box" mass=".08" size="{vector(values[5])}"/>'
                   '</body></worldbody></mujoco>\n')
            path = fixtures/(name+'.xml')
            path.write_text(xml)
            native_m = mujoco.MjModel.from_xml_string(xml)
            native_d = mujoco.MjData(native_m);mujoco.mj_forward(native_m,native_d)
            row=dict(name=name,fixture=str(path.relative_to(a.output.parent)),metadata=metadata,
                     geometry=[np.asarray(v).tolist() for v in values])
            row['native'] = result(native_d.contact.dist,native_d.contact.pos,
                                   native_d.contact.frame.reshape(-1,3,3)[:,0])
            # Use exactly the same float32 inputs for both array backends.
            inputs=[np.asarray(v,dtype=np.float32) for v in values]
            jo=jkernel(*(jp.array(v) for v in inputs));jax.block_until_ready(jo)
            mo=mkernel(*(mx.array(v) for v in inputs));mx.eval(*mo)
            bo=mx.vmap(mkernel)(*(mx.stack([mx.array(v)]*2) for v in inputs));mx.eval(*bo)
            row['official_jax']=result(*jo);row['mlx_scalar']=result(*mo)
            row['mlx_batch']=result(*(v[0] for v in bo))
            row['scalar_batch_max_abs']={key:float(np.max(np.abs(np.asarray(x)-np.asarray(y[0]))))
                                         for key,x,y in zip(('dist','pos','normal'),mo,bo)}
            row['jax_mlx_max_abs']={key:float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
                                    for key,x,y in zip(('dist','pos','normal'),jo,mo)}
            saved=mc._point_in_front_of_plane
            try:
                mc._point_in_front_of_plane=lambda plane,normal,point:mx.sum((point-plane)*normal)>0
                ablated=mkernel(*(mx.array(v) for v in inputs));mx.eval(*ablated)
                row['mlx_zero_threshold_ablation']=result(*ablated)
            finally:
                mc._point_in_front_of_plane=saved
            report['cases'].append(row)
            save(a.output,report)
        report.update(status='completed')
    except BaseException:
        report.update(status='failed',error=traceback.format_exc())
    save(a.output,report)
    return report['status'] != 'completed'


if __name__ == '__main__':
    args=arguments()
    sys.exit(worker(args) if args._worker else supervise(args))
