"""Native-MuJoCo oracles for constraint paths missed by contact-only tests."""
from pathlib import Path
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
mujoco.__path__.append(str(ROOT))
import mlx.core as mx
from mujoco.mjx_mlx._src import constraint, forward, io, passive, solver, support


class AuditEdgeCases(unittest.TestCase):
    def test_native_mass_matrix_precision(self):
        """Float32 mass assembly must not silently use reduced input precision."""
        # Four bodies exercise the GPU matrix-multiply path used by the hand,
        # while excluding collision/solver error and accumulated integration.
        masses = (.08, .1, .06, .07)
        bodies = ''.join(
            f'<body pos="{i * .4} 0 .3"><freejoint/>'
            f'<geom type="box" size=".02 .03 .04" mass="{mass}" '
            f'contype="0" conaffinity="0"/></body>'
            for i, mass in enumerate(masses))
        m = mujoco.MjModel.from_xml_string(
            '<mujoco><option gravity="0 0 0"><flag contact="disable"/></option>'
            f'<worldbody>{bodies}</worldbody></mujoco>')
        self.assertEqual((m.nq, m.nv), (28, 24))
        x = io.put_model(m)
        d = forward.fwd_position(x, io.make_data(m))
        native = mujoco.MjData(m)
        mujoco.mj_forward(m, native)
        expected = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, expected, native.qM)
        np.testing.assert_allclose(
            np.asarray(support.full_m(x, d)), expected, rtol=0, atol=1e-7,
            err_msg='Mass-matrix precision regression: launch a fresh process with '
                    'MLX_ENABLE_TF32=0 before importing MLX; do not relax the tolerance.')

    def test_elliptic_without_contact_slots(self):
        """A cone setting must not prevent joint-only constraints being solved."""
        for kind in ("Newton", "CG"):
            with self.subTest(solver=kind):
                m = mujoco.MjModel.from_xml_string(f'''<mujoco>
                  <compiler angle="radian"/>
                  <option cone="elliptic" solver="{kind}" iterations="20"
                          tolerance="1e-9" gravity="0 0 0"/>
                  <worldbody><body>
                    <joint range="-.1 .1" frictionloss=".03"/>
                    <geom size=".1" contype="0" conaffinity="0"/>
                  </body></worldbody>
                </mujoco>''')
                x, d = io.put_model(m), io.make_data(m)
                native = mujoco.MjData(m)
                native.qpos[:] = .2
                native.qvel[:] = .1
                d = d.replace(qpos=mx.array(native.qpos), qvel=mx.array(native.qvel))
                for fn in (forward.fwd_position, forward.fwd_velocity,
                           forward.fwd_actuation, forward.fwd_acceleration):
                    d = fn(x, d)
                self.assertEqual((d._impl or d).contact.dim.size, 0)
                self.assertEqual((d._impl or d).nf, 1)
                self.assertEqual((d._impl or d).nl, 1)
                d = d.replace(qacc=d.qacc_smooth)
                context = solver.Context.create(x, d)
                self.assertEqual(context.fri.shape, (0, 6))
                self.assertEqual(context.u.shape, (0, 6))
                # Exercise line search directly even if a solver's warm start
                # happens to satisfy this small system in zero iterations.
                stepped = solver._linesearch(x, d, context)
                self.assertTrue(np.isfinite(np.asarray(stepped.qacc)).all())
                actual = solver.solve(x, d)
                self.assertGreater(int(actual._impl.solver_niter), 0)
                self.assertLessEqual(int(actual._impl.solver_niter), m.opt.iterations)
                mujoco.mj_forward(m, native)
                np.testing.assert_allclose(np.asarray(actual.qacc), native.qacc,
                                           rtol=3e-4, atol=3e-4)
                np.testing.assert_allclose(np.asarray(actual.qfrc_constraint),
                                           native.qfrc_constraint, rtol=3e-4, atol=3e-4)

    def test_single_and_multiple_joint_equalities_against_native(self):
        """Preserve per-equality addresses, including missing joint2 semantics."""
        equalities = (
            '<joint joint1="a" polycoef=".02 0 0 0 0"/>',
            '<joint joint1="b" joint2="c" polycoef=".01 .7 .2 0 0"/>',
        )
        for count in (1, 2):
            with self.subTest(equalities=count):
                m = mujoco.MjModel.from_xml_string(f'''<mujoco>
                  <option cone="pyramidal" gravity="0 0 0"/>
                  <worldbody>
                    <body><joint name="a"/><geom size=".1" contype="0" conaffinity="0"/></body>
                    <body pos="1 0 0"><joint name="b"/><geom size=".1" contype="0" conaffinity="0"/></body>
                    <body pos="2 0 0"><joint name="c"/><geom size=".1" contype="0" conaffinity="0"/></body>
                  </worldbody>
                  <equality>{''.join(equalities[:count])}</equality>
                </mujoco>''')
                x, d = io.put_model(m), io.make_data(m)
                native = mujoco.MjData(m)
                native.qpos[:] = [.12, -.07, .04]
                native.qvel[:] = [.2, -.1, .05]
                d = d.replace(qpos=mx.array(native.qpos), qvel=mx.array(native.qvel))
                actual = constraint.make_constraint(x, d)._impl
                mujoco.mj_forward(m, native)
                self.assertEqual(actual.ne, count)
                np.testing.assert_allclose(np.asarray(actual.efc_J),
                                           native.efc_J.reshape((-1, m.nv)), atol=2e-6)
                for field in ("efc_pos", "efc_D", "efc_aref"):
                    np.testing.assert_allclose(np.asarray(getattr(actual, field)),
                                               getattr(native, field), rtol=3e-5,
                                               atol=3e-5, err_msg=field)

    def test_connect_and_weld_equalities_against_native(self):
        """Body and site equalities share physical formulas, not index domains."""
        for kind in ("connect", "weld"):
            for target in ("body", "site"):
                with self.subTest(kind=kind, target=target):
                    if target == "body":
                        attributes = 'body1="a" body2="b"'
                        if kind == "connect":
                            attributes += ' anchor="0 0 .3"'
                    else:
                        attributes = 'site1="sa" site2="sb"'
                    m = mujoco.MjModel.from_xml_string(f'''<mujoco>
                      <option gravity="0 0 0"/>
                      <worldbody>
                        <site name="unrelated"/>
                        <body name="a" pos="-.2 0 .3"><freejoint/>
                          <geom size=".1" contype="0" conaffinity="0"/>
                          <site name="sa" pos=".2 0 0"/>
                        </body>
                        <body name="b" pos=".2 0 .3"><freejoint/>
                          <geom size=".1" contype="0" conaffinity="0"/>
                          <site name="sb" pos="-.2 0 0"/>
                        </body>
                      </worldbody>
                      <equality><{kind} {attributes}/></equality>
                    </mujoco>''')
                    x, d = io.put_model(m), io.make_data(m)
                    native = mujoco.MjData(m)
                    native.qpos[0] += .02
                    native.qpos[8] -= .01
                    native.qpos[3:7] = [np.cos(.03), 0, np.sin(.03), 0]
                    native.qvel[:] = np.linspace(-.1, .2, m.nv)
                    d = d.replace(qpos=mx.array(native.qpos), qvel=mx.array(native.qvel))
                    actual = forward.fwd_position(x, d)._impl
                    mujoco.mj_forward(m, native)
                    self.assertEqual(actual.ne, 3 if kind == "connect" else 6)
                    np.testing.assert_allclose(np.asarray(actual.efc_J),
                                               native.efc_J.reshape((-1, m.nv)),
                                               rtol=3e-5, atol=3e-6)
                    for field in ("efc_pos", "efc_D", "efc_aref"):
                        np.testing.assert_allclose(np.asarray(getattr(actual, field)),
                                                   getattr(native, field),
                                                   rtol=5e-5, atol=5e-5, err_msg=field)

    def test_rk4_against_native(self):
        """RK4 must keep each tableau stage and its dynamic arrays on MLX."""
        m = mujoco.MjModel.from_xml_string('''<mujoco>
          <option integrator="RK4" timestep=".002"/>
          <worldbody><body pos="0 0 1">
            <joint name="hinge" axis="0 1 0" damping=".1" stiffness=".3"/>
            <geom type="capsule" fromto="0 0 0 0 0 -.4" size=".04"/>
          </body></worldbody>
          <actuator><motor joint="hinge"/></actuator>
        </mujoco>''')
        x, d = io.put_model(m), io.make_data(m)
        native = mujoco.MjData(m)
        native.qpos[:] = .35
        native.qvel[:] = -.2
        native.ctrl[:] = .07
        d = d.replace(qpos=mx.array(native.qpos), qvel=mx.array(native.qvel),
                      ctrl=mx.array(native.ctrl))
        for _ in range(10):
            d = forward.step(x, d)
            mujoco.mj_step(m, native)
        for field in ("qpos", "qvel", "time"):
            value = getattr(d, field)
            self.assertIsInstance(value, mx.array)
            np.testing.assert_allclose(np.asarray(value), getattr(native, field),
                                       rtol=3e-5, atol=3e-5, err_msg=field)
        self.assertEqual(int(d._impl.solver_niter), 0)

    def test_heterogeneous_joint_springs_against_native(self):
        """Spring coefficients are dynamic scan arguments, not static enums."""
        m = mujoco.MjModel.from_xml_string('''<mujoco>
          <compiler angle="radian"/><option gravity="0 0 0"/>
          <worldbody>
            <body><joint stiffness=".3" damping=".1" springref=".05"/>
              <geom size=".1" contype="0" conaffinity="0"/></body>
            <body pos="1 0 0"><joint stiffness=".7" damping=".2" springref="-.03"/>
              <geom size=".1" contype="0" conaffinity="0"/></body>
          </worldbody>
        </mujoco>''')
        x, d = io.put_model(m), io.make_data(m)
        native = mujoco.MjData(m)
        native.qpos[:] = [.15, -.2]
        native.qvel[:] = [.12, -.3]
        d = d.replace(qpos=mx.array(native.qpos), qvel=mx.array(native.qvel))
        actual = passive._spring_damper(x, d)
        mujoco.mj_forward(m, native)
        np.testing.assert_allclose(np.asarray(actual), native.qfrc_passive,
                                   rtol=2e-6, atol=2e-6)

    def test_tendon_only_spring_deadband_and_damping_against_native(self):
        """Zero joint coefficients must not suppress fixed-tendon forces."""
        m = mujoco.MjModel.from_xml_string('''<mujoco>
          <option gravity="0 0 0"/>
          <worldbody>
            <body><joint name="a" type="slide"/>
              <geom size=".1" contype="0" conaffinity="0"/></body>
            <body pos="1 0 0"><joint name="b" type="slide"/>
              <geom size=".1" contype="0" conaffinity="0"/></body>
          </worldbody>
          <tendon><fixed stiffness="2.5" damping=".3" springlength=".1 .2">
            <joint joint="a" coef="1"/><joint joint="b" coef=".5"/>
          </fixed></tendon>
        </mujoco>''')
        self.assertFalse(np.any(m.jnt_stiffness))
        self.assertFalse(np.any(m.dof_damping))
        x = io.put_model(m)
        for length in (.03, .15, .25):
            with self.subTest(length=length):
                native = mujoco.MjData(m)
                native.qpos[:] = [length, 0.]
                native.qvel[:] = [.1, -.03]
                d = io.make_data(m).replace(qpos=mx.array(native.qpos),
                                             qvel=mx.array(native.qvel))
                d = forward.fwd_velocity(x, forward.fwd_position(x, d))
                mujoco.mj_forward(m, native)
                np.testing.assert_allclose(np.asarray(d.ten_length), native.ten_length,
                                           atol=2e-6)
                np.testing.assert_allclose(np.asarray(d._impl.ten_velocity),
                                           native.ten_velocity, atol=2e-6)
                self.assertGreater(np.linalg.norm(native.qfrc_passive), 0.)
                np.testing.assert_allclose(np.asarray(d.qfrc_passive),
                                           native.qfrc_passive, rtol=3e-6, atol=3e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
