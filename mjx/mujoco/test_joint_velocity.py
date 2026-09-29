"""Native oracles for simultaneous rotational DOFs in free and ball joints."""
from pathlib import Path
import unittest

import mujoco
import numpy as np

mujoco.__path__.append(str(Path(__file__).resolve().parent))
import mlx.core as mx
from mujoco.mjx_mlx._src import forward, io


class JointVelocityTests(unittest.TestCase):
    def check_velocity(self, m, qvel):
        native = mujoco.MjData(m)
        for j in range(m.njnt):
            adr = m.jnt_qposadr[j]
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                adr += 3
            elif m.jnt_type[j] != mujoco.mjtJoint.mjJNT_BALL:
                native.qpos[adr] = .3
                continue
            quat = np.array([.9, .1, .2, .3])
            native.qpos[adr:adr + 4] = quat / np.linalg.norm(quat)
        native.qvel[:] = qvel
        mujoco.mj_forward(m, native)
        x = io.put_model(m)
        d = io.make_data(m).replace(qpos=mx.array(native.qpos),
                                    qvel=mx.array(native.qvel))
        d = forward.fwd_velocity(x, forward.fwd_position(x, d))
        for field in ('cvel', 'cdof_dot', 'qfrc_bias'):
            value = getattr(d, field) if hasattr(d, field) else getattr(d._impl, field)
            with self.subTest(field=field):
                np.testing.assert_allclose(np.asarray(value), getattr(native, field),
                                           rtol=3e-5, atol=3e-6, err_msg=field)
        return native, d

    def test_isotropic_free_body(self):
        m = mujoco.MjModel.from_xml_string('''<mujoco>
          <option gravity="0 0 0"/>
          <worldbody><body pos=".3 .2 .5"><freejoint/>
            <geom type="box" size=".1 .1 .1" mass="1" contype="0" conaffinity="0"/>
          </body></worldbody></mujoco>''')
        native, d = self.check_velocity(m, [-.4, .6, .2, 2, 3, 4])
        np.testing.assert_allclose(native.qfrc_bias, 0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(d.qfrc_bias), 0, atol=3e-6)

    def test_anisotropic_free_body(self):
        m = mujoco.MjModel.from_xml_string('''<mujoco>
          <option gravity="0 0 0"/>
          <worldbody><body pos=".3 .2 .5"><freejoint/>
            <geom type="box" size=".1 .2 .3" mass="1" contype="0" conaffinity="0"/>
          </body></worldbody></mujoco>''')
        native, _ = self.check_velocity(m, [-.4, .6, .2, 2, 3, 4])
        self.assertGreater(np.linalg.norm(native.qfrc_bias[3:]), .01)

    def test_ball_with_moving_parent(self):
        m = mujoco.MjModel.from_xml_string('''<mujoco>
          <option gravity="0 0 0"/>
          <worldbody><body pos=".3 .2 .5">
            <joint axis="0 1 0"/>
            <geom type="box" size=".1 .2 .3" mass="1" contype="0" conaffinity="0"/>
            <body pos=".2 .1 .3"><joint type="ball"/>
              <geom type="box" pos=".1 .04 .02" size=".07 .09 .11" mass=".5"
                    contype="0" conaffinity="0"/>
            </body>
          </body></worldbody></mujoco>''')
        native, _ = self.check_velocity(m, [.7, 2, 3, 4])
        self.assertGreater(np.linalg.norm(native.cdof_dot), .01)


if __name__ == '__main__':
    unittest.main(verbosity=2)
