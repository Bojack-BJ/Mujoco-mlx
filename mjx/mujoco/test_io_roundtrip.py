"""Public single-environment MuJoCo <-> MLX IO regression coverage."""
import dataclasses
from pathlib import Path
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
mujoco.__path__.append(str(ROOT))
import mlx.core as mx
from mujoco.mjx_mlx._src import io, smooth, types


def joint_scene(jacobian='dense'):
    return mujoco.MjModel.from_xml_string(f'''<mujoco>
      <compiler angle="radian"/><size nuserdata="2"/>
      <option jacobian="{jacobian}"/>
      <worldbody><body name="mocap" mocap="true" pos=".2 .3 .4"/>
        <body><joint name="a" type="hinge" axis="0 1 0" limited="true"
          range="-.2 .2" frictionloss=".03"/>
          <geom type="capsule" fromto="0 0 0 0 0 .4" size=".02"/>
          <body pos="0 0 .4"><joint name="b" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0 0 .3" size=".02"/>
          </body>
        </body>
      </worldbody>
      <actuator><motor joint="a"/><motor joint="b"/></actuator>
      <tendon><fixed name="coupled"><joint joint="a" coef="1"/>
        <joint joint="b" coef="-.7"/></fixed></tendon>
      <sensor><jointpos joint="a"/><jointvel joint="b"/></sensor>
    </mujoco>''')


def dense_efc(m, d):
    if not mujoco.mj_isSparse(m):
        return d.efc_J.reshape((d.nefc, m.nv))
    result = np.zeros((d.nefc, m.nv))
    mujoco.mju_sparse2dense(result, d.efc_J, d.efc_J_rownnz,
                           d.efc_J_rowadr, d.efc_J_colind)
    return result


class IORoundtripRegression(unittest.TestCase):
    def assert_public_equal(self, native, restored):
        for field in types.Data.fields():
            if field.name != '_impl':
                np.testing.assert_allclose(getattr(restored, field.name),
                    getattr(native, field.name), atol=2e-6, rtol=3e-6,
                    err_msg=field.name)

    def assert_constraint_equal(self, m, native, restored):
        for name in ('ncon', 'ne', 'nf', 'nl', 'nefc'):
            self.assertEqual(getattr(native, name), getattr(restored, name), name)
        np.testing.assert_allclose(dense_efc(m, restored), dense_efc(m, native),
                                   atol=2e-6, rtol=3e-6)
        for name in ('efc_type', 'efc_id', 'efc_pos', 'efc_margin', 'efc_frictionloss',
                     'efc_D', 'efc_aref', 'efc_force'):
            np.testing.assert_allclose(getattr(restored, name), getattr(native, name),
                                       atol=2e-6, rtol=3e-6, err_msg=name)

    def test_make_data_native_and_mlx_model(self):
        m = joint_scene()
        for model in (m, io.put_model(m)):
            data = io.make_data(model)
            def check(node):
                for field in dataclasses.fields(node):
                    value = getattr(node, field.name)
                    if field.type is mx.array:
                        self.assertIsInstance(value, mx.array, field.name)
                    elif dataclasses.is_dataclass(value):
                        check(value)
            check(data)
            self.assertEqual(data._impl.nf, 1)
            self.assertEqual(data._impl.nl, 1)
            np.testing.assert_allclose(np.asarray(data.mocap_pos), [[.2, .3, .4]])
            # Freshly allocated contact slots must not export fake contacts.
            result = io.get_data(m, data)
            self.assertEqual(result.ncon, 0)
            self.assertEqual(result.nefc, 0)

    def test_joint_state_roundtrip_and_cholesky_orientation(self):
        m = joint_scene()
        native = mujoco.MjData(m)
        native.qpos[:] = [.25, -.3]
        native.qvel[:] = [.2, -.1]
        native.ctrl[:] = [.11, -.12]
        native.qfrc_applied[:] = [.01, -.02]
        native.userdata[:] = [1.2, -3.4]
        native.time = .123
        mujoco.mj_forward(m, native)
        data = io.put_data(m, native)
        self.assertIsInstance(data._impl, types.DataMLX)
        self.assertIsInstance(data.time, mx.array)
        self.assertIsInstance(data._impl.solver_niter, mx.array)
        restored = io.get_data(m, data)
        self.assert_public_equal(native, restored)
        self.assert_constraint_equal(m, native, restored)
        for name in ('actuator_moment', 'ten_J', 'qM', 'M'):
            np.testing.assert_allclose(getattr(restored, name), getattr(native, name),
                                       atol=2e-6, rtol=3e-6, err_msg=name)
        matrix = np.asarray(data._impl.qM)
        lower = np.asarray(data._impl.qLD)
        np.testing.assert_allclose(lower @ lower.T, matrix, atol=2e-7, rtol=3e-6)
        rhs = mx.array([.3, -.4])
        actual = smooth.solve_m(io.put_model(m), data, rhs)
        np.testing.assert_allclose(np.asarray(actual), np.linalg.solve(matrix, np.asarray(rhs)),
                                   atol=2e-4, rtol=3e-6)
        # Mutating native buffers must not mutate the imported snapshot.
        saved = np.asarray(data.qpos).copy()
        native.qpos[:] = 9
        np.testing.assert_array_equal(np.asarray(data.qpos), saved)

    def test_sparse_tendon_roundtrip(self):
        m = joint_scene('sparse')
        native = mujoco.MjData(m)
        native.qpos[:] = [.25, -.3]
        native.ctrl[:] = [.1, -.2]
        mujoco.mj_forward(m, native)
        data = io.put_data(m, native)
        result = io.get_data(m, data)
        self.assert_public_equal(native, result)
        self.assert_constraint_equal(m, native, result)
        # Verify logical tendon Jacobians even if compressed zero padding moves.
        reimported = io.put_data(m, result)
        np.testing.assert_allclose(np.asarray(reimported._impl.ten_J),
                                   np.asarray(data._impl.ten_J), atol=2e-7)

    def test_noncontact_rows_preserve_inactive_holes_type_and_object_id(self):
        m = mujoco.MjModel.from_xml_string('''<mujoco>
          <compiler angle="radian"/>
          <worldbody><body><joint name="a" limited="true" range="-.1 .1"/>
            <geom size=".1"/><body pos="0 0 .3">
              <joint name="b" limited="true" range="-.1 .1"/>
              <geom size=".1"/></body></body></worldbody>
          <tendon><fixed limited="true" range="-.05 .05">
            <joint joint="b" coef="1"/></fixed></tendon>
        </mujoco>''')
        for qpos in ((.3, 0.), (0., .08), (0., .3), (.3, .3)):
            with self.subTest(qpos=qpos):
                native = mujoco.MjData(m)
                native.qpos[:] = qpos
                mujoco.mj_forward(m, native)
                data = io.put_data(m, native)
                keys = io._noncontact_row_keys(m)
                active = np.flatnonzero(np.any(np.asarray(data._impl.efc_J) != 0, axis=1))
                expected_keys = [(int(kind), int(obj), 0)
                                 for kind, obj in zip(native.efc_type, native.efc_id)]
                self.assertEqual([keys[row] for row in active], expected_keys)
                result = io.get_data(m, data)
                self.assert_constraint_equal(m, native, result)

        # Equalities also contain inactive holes; the active equality's native
        # object id must survive rather than becoming the first equality id.
        eq_model = mujoco.MjModel.from_xml_string('''<mujoco>
          <worldbody><body><joint name="a"/><geom size=".1"/>
            <body pos="0 0 .3"><joint name="b"/><geom size=".1"/>
            </body></body></worldbody>
          <equality><joint joint1="a" active="false"/><joint joint1="b"/></equality>
        </mujoco>''')
        native = mujoco.MjData(eq_model)
        native.qpos[:] = [.02, .03]
        mujoco.mj_forward(eq_model, native)
        self.assertEqual(native.ne, 1)
        self.assertEqual(native.efc_id[0], 1)
        data = io.put_data(eq_model, native)
        np.testing.assert_array_equal(np.asarray(data._impl.efc_J)[0], 0)
        result = io.get_data(eq_model, data)
        self.assert_constraint_equal(eq_model, native, result)

    def test_contact_roundtrip_dense_sparse_and_positive_gap(self):
        for jacobian in ('dense', 'sparse'):
            for cone in ('pyramidal', 'elliptic'):
                for height, gap in ((.019, 0.), (.023, .007)):
                    with self.subTest(jacobian=jacobian, cone=cone, height=height):
                        m = mujoco.MjModel.from_xml_string(f'''<mujoco>
                          <option jacobian="{jacobian}" cone="{cone}"/>
                          <worldbody><geom type="plane" size="1 1 .1"
                            margin=".005" gap="{gap}"/>
                            <body pos="0 0 {height}"><freejoint/>
                              <geom type="box" size=".02 .03 .02"/>
                            </body>
                            <body pos=".5 0 1"><freejoint/>
                              <geom type="box" size=".02 .03 .02"/>
                            </body>
                          </worldbody></mujoco>''')
                        native = mujoco.MjData(m)
                        mujoco.mj_forward(m, native)
                        self.assertGreater(native.ncon, 0)
                        if gap:
                            self.assertTrue(np.all(native.contact.dist > 0))
                            self.assertTrue(np.all(native.contact.efc_address == -1))
                        data = io.put_data(m, native)
                        result = mujoco.MjData(m)
                        io.get_data_into(result, m, data)
                        self.assert_public_equal(native, result)
                        self.assert_constraint_equal(m, native, result)
                        for name in ('dist', 'pos', 'frame', 'geom', 'dim', 'efc_address'):
                            np.testing.assert_allclose(getattr(result.contact, name),
                                getattr(native.contact, name), atol=2e-6, rtol=3e-6,
                                err_msg=name)

    def test_empty_model_and_explicit_batch_rejection(self):
        m = mujoco.MjModel.from_xml_string('<mujoco/>')
        native = mujoco.MjData(m)
        mujoco.mj_forward(m, native)
        data = io.put_data(m, native)
        result = io.get_data(m, data)
        self.assert_public_equal(native, result)
        self.assert_constraint_equal(m, native, result)
        with self.assertRaisesRegex(NotImplementedError, 'Batched'):
            io.get_data_into([result], m, data)
        batched = data.replace(qpos=mx.zeros((2, 0)))
        with self.assertRaisesRegex(NotImplementedError, 'Batched'):
            io.get_data(m, batched)


if __name__ == '__main__':
    unittest.main(verbosity=2)
