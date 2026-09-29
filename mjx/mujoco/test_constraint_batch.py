"""Regression for batched constraint assembly against pre-change code and MuJoCo."""
from pathlib import Path
import subprocess
import types
import unittest
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
mujoco.__path__.append(str(ROOT))
import mlx.core as mx
from mujoco.mjx_mlx._src import constraint, support, io, forward

baseline = types.ModuleType('scalar_constraint_baseline')
exec(subprocess.check_output(['git', 'show', '91a7da2d:mjx/mujoco/mjx_mlx/_src/constraint.py'], cwd=ROOT), baseline.__dict__)
# The old frictionless path leaves NumPy invweight scalars as static tree
# metadata. Normalize leaf types in the oracle; retain its scalar arithmetic.
baseline_row = baseline._row
baseline._row = lambda j, *args: baseline_row(j, *(mx.array(a) for a in args))


def scene(dim=3, cone='elliptic'):
    # Two free bodies give nonzero Jacobians on both sides. One is separated.
    m = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option cone="{cone}"/><worldbody>
        <geom name="ground" type="plane" size="1 1 .1" condim="{dim}"/>
        <body pos="0 0 .019"><freejoint/><geom type="box" size=".02 .03 .02" condim="{dim}"/></body>
        <body pos=".3 .2 .3"><freejoint/><geom type="box" size=".02 .03 .02" condim="{dim}"/></body>
      </worldbody></mujoco>''')
    x, d = io.put_model(m), io.make_data(m)
    d = forward.fwd_position(x, d)
    return m, x, d


class ConstraintBatchRegression(unittest.TestCase):
    def test_data_arrays_and_global_contact_metadata(self):
        import dataclasses
        m, x, d = scene()
        def check(node):
            for f in dataclasses.fields(node):
                value = getattr(node,f.name)
                if f.type is mx.array:
                    self.assertIsInstance(value,mx.array,f.name)
                elif dataclasses.is_dataclass(value):
                    check(value)
        check(io.make_data(m))
        impl = d._impl or d
        self.assertEqual(impl.contact.dim.size,impl.contact.dist.size)
        self.assertEqual(impl.contact.efc_address.size,impl.contact.dist.size)
        self.assertIsInstance(impl.contact.dim,np.ndarray)

    def test_jacobian_against_native_and_scalar(self):
        m, x, d = scene()
        native = mujoco.MjData(m)
        mujoco.mj_forward(m, native)
        bodies = np.array([0, 1, 2, 1])
        points = np.array([[0.,0,0], [.01,.02,.03], [.2,.1,.3], [-.1,0,.1]])
        jp, jr = support.jac_batch(x, d, mx.array(points), bodies)
        for i, body in enumerate(bodies):
            expected_p, expected_r = support.jac(x, d, mx.array(points[i]), int(body))
            native_p, native_r = np.zeros((3,m.nv)), np.zeros((3,m.nv))
            mujoco.mj_jac(m, native, native_p, native_r, points[i], int(body))
            for actual, scalar, ref in ((jp[i],expected_p,native_p.T),(jr[i],expected_r,native_r.T)):
                np.testing.assert_allclose(np.asarray(actual), np.asarray(scalar), atol=2e-6)
                np.testing.assert_allclose(np.asarray(actual), ref, atol=2e-6)

    def test_all_dimensions_and_cones_preserve_rows(self):
        fields = ('J','pos_aref','pos_imp','invweight','solref','solimp','margin','frictionloss')
        for cone in ('pyramidal', 'elliptic'):
            for dim in (1,3,4,6):
                if cone == 'elliptic' and dim == 1:
                    continue  # Explicitly unsupported by io.make_data upstream.
                with self.subTest(cone=cone, dim=dim):
                    _, x, d = scene(dim,cone)
                    name = '_efc_contact_frictionless' if dim == 1 else '_efc_contact_' + cone
                    args = (x,d) if dim == 1 else (x,d,dim)
                    actual, expected = getattr(constraint,name)(*args), getattr(baseline,name)(*args)
                    for field in fields:
                        np.testing.assert_allclose(np.asarray(getattr(actual,field)),
                            np.asarray(getattr(expected,field)), atol=2e-6, rtol=2e-5, err_msg=field)
                    a, e = constraint.make_constraint(x,d), baseline.make_constraint(x,d)
                    a, e = a._impl or a, e._impl or e
                    for field in ('efc_J','efc_D','efc_aref','efc_pos','efc_margin','efc_frictionloss'):
                        np.testing.assert_allclose(np.asarray(getattr(a,field)), np.asarray(getattr(e,field)),
                            atol=2e-5, rtol=2e-5, err_msg=field)

    def test_batched_kbi(self):
        _, x, _ = scene()
        sr = mx.array([[.02,1.],[-100.,-10.],[.01,.5]])
        si = mx.array([[.9,.95,.001,.5,2.],[.8,.99,.02,.3,3.],[.9,.95,.01,.7,1.]])
        pos = mx.array([-.01,.003,0.])
        actual = constraint._kbi(x,sr,si,pos)
        scalar = [baseline._kbi(x,sr[i],si[i],pos[i]) for i in range(3)]
        for j in range(3):
            np.testing.assert_allclose(np.asarray(actual[j]),np.asarray(mx.stack([v[j] for v in scalar])),rtol=2e-5,atol=2e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
