"""Small, reproducible scalar/batched solver comparisons on Apple MLX."""
from pathlib import Path
import subprocess
import types
import unittest
import mujoco
import numpy as np
import test_constraint_batch as fixtures
import mlx.core as mx
from mujoco.mjx_mlx._src import solver, forward, io
from mujoco.mjx_mlx._src.types import SolverType

ROOT = Path(__file__).resolve().parent
old = types.ModuleType('scalar_solver_baseline')
exec(subprocess.check_output(['git','show','91a7da2d:mjx/mujoco/mjx_mlx/_src/solver.py'],cwd=ROOT),old.__dict__)


class SolverRegression(unittest.TestCase):
    def test_cone_blocks_gradient_linesearch_and_solve(self):
        for dim in (3,4,6):
            for kind in (SolverType.CG, SolverType.NEWTON):
                with self.subTest(dim=dim, solver=kind):
                    _, x, d = fixtures.scene(dim)
                    x = x.replace(opt=x.opt.replace(solver=kind, iterations=3))
                    d = d.replace(qvel=mx.array(np.linspace(-.2,.3,x.nv)))
                    for fn in (forward.fwd_velocity,forward.fwd_actuation,forward.fwd_acceleration):
                        d = fn(x,d)
                    d = d.replace(qacc=d.qacc_smooth)
                    a, e = solver.Context.create(x,d), old.Context.create(x,d)
                    for field in ('cost','efc_force','active','h','grad','Mgrad'):
                        np.testing.assert_allclose(np.asarray(getattr(a,field)),np.asarray(getattr(e,field)),rtol=2e-4,atol=2e-4,err_msg=field)
                    a, e = solver._linesearch(x,d,a), old._linesearch(x,d,e)
                    np.testing.assert_allclose(np.asarray(a.qacc),np.asarray(e.qacc),rtol=3e-4,atol=3e-4)
                    a, e = solver.solve(x,d), old.solve(x,d)
                    np.testing.assert_allclose(np.asarray(a.qacc),np.asarray(e.qacc),rtol=5e-4,atol=5e-4)

    def test_implicit_damping_and_joint_sensors_against_native(self):
        m = mujoco.MjModel.from_xml_string('''<mujoco><option integrator="implicitfast"/>
          <worldbody><body><joint name="a" damping=".2"/><geom size=".1"/>
          <body pos="0 0 .3"><joint name="b" damping=".3"/><geom size=".1"/></body></body></worldbody>
          <actuator><position name="a" joint="a" kp="2" kv=".1"/><position name="b" joint="b" kp="3" kv=".2"/></actuator>
          <sensor><jointpos joint="a"/><jointpos joint="b"/><jointvel joint="a"/><jointvel joint="b"/>
          <jointactuatorfrc joint="a"/><jointactuatorfrc joint="b"/></sensor></mujoco>''')
        x,d = io.put_model(m),io.make_data(m)
        native = mujoco.MjData(m)
        native.qvel[:] = [.2,-.1]
        native.ctrl[:] = [.1,-.2]
        d = d.replace(qvel=mx.array(native.qvel),ctrl=mx.array(native.ctrl))
        for _ in range(10):
            d = forward.step(x,d)
            mujoco.mj_step(m,native)
        np.testing.assert_allclose(np.asarray(d.qpos),native.qpos,atol=2e-5)
        # Small-inertia float32 model: velocity error is ~2.5e-4 after 10 steps.
        np.testing.assert_allclose(np.asarray(d.sensordata),native.sensordata,atol=5e-4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
