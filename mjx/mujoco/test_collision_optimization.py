"""GPU regression against the preserved E010 source (run from this directory)."""
import dataclasses
from pathlib import Path
import subprocess
import types
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
mujoco.__path__.append(str(ROOT))
import mlx.core as mx
from mujoco.mjx_mlx._src import collision_convex as new, mesh, io, smooth
from mujoco.mjx_mlx._src.collision_types import GeomInfo, FunctionKey
from mujoco.mjx_mlx._src.types import GeomType

old = types.ModuleType('baseline_collision_convex')
git_root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT, text=True).strip())
source_path = (ROOT / 'mjx_mlx/_src/collision_convex.py').relative_to(git_root)
exec(subprocess.check_output(['git', 'show', f'repair-baseline:{source_path}'], cwd=ROOT), old.__dict__)


def box(pos, angle=0.):
    c, s = np.cos(angle), np.sin(angle)
    info = mesh.box(GeomInfo(mx.array(pos), mx.array([[c,-s,0],[s,c,0],[0,0,1]]), mx.array([.2,.3,.4])))
    return dataclasses.replace(info, **{f.name: mx.array(getattr(info, f.name))
        for f in dataclasses.fields(info) if isinstance(getattr(info, f.name), np.ndarray)})


class CollisionRegression(unittest.TestCase):
    def test_vectorized_gaussmap_matches_scalar(self):
        # Force the general mesh SAT route even though these are box hulls.
        for pos, angle in [([.1,.1,.1],.3), ([.45,0,0],0.), ([.35,.3,.2],.7), ([2.,0,0],.4)]:
            with self.subTest(pos=pos, angle=angle):
                a, b = box([0.,0,0]), box(pos, angle)
                expected, actual = old._convex_convex(a,b), new._convex_convex(a,b)
                mx.eval(*expected, *actual)
                for e, v in zip(expected, actual):
                    np.testing.assert_allclose(np.asarray(v), np.asarray(e), atol=2e-6, rtol=2e-5)

    def test_broadphase_preserves_active_contacts_and_slots(self):
        for x in (.1, 2.):
            with self.subTest(x=x):
                m = mujoco.MjModel.from_xml_string(f'<mujoco><worldbody><geom type="box" size=".2 .2 .2"/><body pos="{x} 0 0"><freejoint/><geom type="box" size=".2 .2 .2"/></body></worldbody></mujoco>')
                model, data = io.put_model(m), io.make_data(m)
                data = smooth.kinematics(model, data)
                key = FunctionKey(types=(GeomType.BOX,GeomType.BOX), data_ids=(-1,-1), condim=3)
                geom = mx.array([[0,1]])
                expected = old.box_box(model,data,key,geom)
                actual = new.box_box(model,data,key,geom)
                mx.eval(*expected,*actual)
                self.assertEqual(actual[0].shape, expected[0].shape)
                active = np.asarray(expected[0]) < 0
                np.testing.assert_array_equal(np.asarray(actual[0]) < 0, active)
                for e,v in zip(expected,actual):
                    np.testing.assert_allclose(np.asarray(v)[active], np.asarray(e)[active], atol=2e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
