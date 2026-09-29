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


def random_orientation(rng):
    """Seeded full-3D orientation, with CPU-only setup independent of MLX."""
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    return quat, matrix.reshape(3, 3)


def xml_vector(values):
    return ' '.join(f'{value:.17g}' for value in values)


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

    def test_grouped_plane_boxes_match_all_scalar_outputs(self):
        rng = np.random.default_rng(20260930)
        plane_quat, plane_mat = random_orientation(rng)
        plane_pos = np.array([.1, -.2, .3])
        bodies = []
        # Mixed penetrations, exact touching and separation. The plane itself
        # is tilted, so this also detects accidentally assuming world-Z up.
        for i, separation in enumerate((-.04, .0, .03, -.025, .2, -.015)):
            size = rng.uniform(.06, .22, size=3)
            quat, mat = random_orientation(rng)
            normal = plane_mat[:, 2]
            support = np.sum(np.abs(mat.T @ normal) * size)
            local = np.array([i * .6, (-1) ** i * .2, support + separation])
            pos = plane_pos + plane_mat @ local
            bodies.append(f'<body pos="{xml_vector(pos)}" quat="{xml_vector(quat)}">'
                          f'<freejoint/><geom name="box{i}" type="box" '
                          f'size="{xml_vector(size)}"/></body>')
        xml = ('<mujoco><worldbody><geom name="plane" type="plane" size="2 2 .1" '
               f'pos="{xml_vector(plane_pos)}" quat="{xml_vector(plane_quat)}"/>'
               + ''.join(bodies) + '</worldbody></mujoco>')
        m = mujoco.MjModel.from_xml_string(xml)
        model, data = io.put_model(m), io.make_data(m)
        data = smooth.kinematics(model, data)
        key = FunctionKey(types=(GeomType.PLANE, GeomType.BOX), data_ids=(-1, -1), condim=3)
        # Non-monotonic ordering detects mistakes when flattening vmap output.
        pairs = np.array([[m.geom('plane').id, m.geom(f'box{i}').id]
                          for i in (3, 0, 5, 1, 4, 2)], dtype=np.int32)
        for input_kind in ('numpy', 'mlx'):
            with self.subTest(input_kind=input_kind):
                geom = pairs if input_kind == 'numpy' else mx.array(pairs)
                expected = old.plane_convex(model, data, key, geom)
                actual = new.plane_convex(model, data, key, geom)
                mx.eval(*expected, *actual)
                self.assertGreater(np.count_nonzero(np.any(np.asarray(expected[0]).reshape(-1, 4) < 0, axis=1)), 1)
                for e, v in zip(expected, actual):
                    self.assertEqual(v.shape, e.shape)
                    np.testing.assert_allclose(np.asarray(v), np.asarray(e), atol=3e-6, rtol=2e-5)

    def test_grouped_boxes_preserve_activation_and_active_slots(self):
        rng = np.random.default_rng(70031)
        geoms, bodies = [], []
        for i in range(7):
            size_a, size_b = rng.uniform(.08, .23, size=(2, 3))
            quat_a, mat_a = random_orientation(rng)
            quat_b, _ = random_orientation(rng)
            anchor = np.array([i * 3., .0, .7])
            if i < 3:
                # Three guaranteed intersecting pairs exercise vmap with >1
                # live pair, even when the broadphase removes far pairs.
                offset = rng.uniform(-.04, .04, size=3)
            elif i in (3, 4):
                quat_b = quat_a.copy()
                offset = mat_a[:, 0] * (size_a[0] + size_b[0] + (.0 if i == 3 else .01))
            else:
                offset = np.array([1.8, .3, -.1])
            geoms.append(f'<geom name="anchor{i}" type="box" pos="{xml_vector(anchor)}" '
                         f'quat="{xml_vector(quat_a)}" size="{xml_vector(size_a)}"/>')
            bodies.append(f'<body pos="{xml_vector(anchor + offset)}" quat="{xml_vector(quat_b)}">'
                          f'<freejoint/><geom name="moving{i}" type="box" '
                          f'size="{xml_vector(size_b)}"/></body>')
        # Positive margins and negative gaps expand the active shell; culling
        # must not assume that only strictly penetrating contacts matter.
        xml = ('<mujoco><default><geom margin=".03" gap="-.01"/></default><worldbody>'
               + ''.join(geoms + bodies) + '</worldbody></mujoco>')
        m = mujoco.MjModel.from_xml_string(xml)
        model, data = io.put_model(m), io.make_data(m)
        data = smooth.kinematics(model, data)
        key = FunctionKey(types=(GeomType.BOX, GeomType.BOX), data_ids=(-1, -1), condim=3)
        pairs = np.array([[m.geom(f'anchor{i}').id, m.geom(f'moving{i}').id]
                          for i in (5, 2, 0, 6, 4, 1, 3)], dtype=np.int32)
        include = np.repeat(np.sum(m.geom_margin[pairs] - m.geom_gap[pairs], axis=1), 4)
        for input_kind in ('numpy', 'mlx'):
            with self.subTest(input_kind=input_kind):
                geom = pairs if input_kind == 'numpy' else mx.array(pairs)
                expected = old.box_box(model, data, key, geom)
                actual = new.box_box(model, data, key, geom)
                mx.eval(*expected, *actual)
                active = np.asarray(expected[0]) < include
                self.assertGreater(np.count_nonzero(np.any(active.reshape(-1, 4), axis=1)), 1)
                self.assertTrue(np.any(~active))
                np.testing.assert_array_equal(np.asarray(actual[0]) < include, active)
                for e, v in zip(expected, actual):
                    self.assertEqual(v.shape, e.shape)
                    np.testing.assert_allclose(np.asarray(v)[active], np.asarray(e)[active],
                                               atol=4e-6, rtol=2e-5)

    def test_grouped_zero_and_single_pair_shapes(self):
        xml = ('<mujoco><worldbody><geom name="plane" type="plane" size="1 1 .1"/>'
               '<geom name="anchor" type="box" size=".1 .15 .2" pos="0 0 .1"/>'
               '<body pos=".08 .01 .12" quat=".9238795 .2209424 .2209424 .2209424">'
               '<freejoint/><geom name="moving" type="box" size=".12 .09 .17"/>'
               '</body></worldbody></mujoco>')
        m = mujoco.MjModel.from_xml_string(xml)
        model, data = io.put_model(m), io.make_data(m)
        data = smooth.kinematics(model, data)
        for name, first_type, first_geom in (('plane_convex', GeomType.PLANE, 'plane'),
                                             ('box_box', GeomType.BOX, 'anchor')):
            key = FunctionKey(types=(first_type, GeomType.BOX), data_ids=(-1, -1), condim=3)
            for input_kind in ('numpy', 'mlx'):
                with self.subTest(collider=name, input_kind=input_kind):
                    pairs = np.array([[m.geom(first_geom).id, m.geom('moving').id]], dtype=np.int32)
                    geom = pairs if input_kind == 'numpy' else mx.array(pairs)
                    # The baseline wrapper mishandles NumPy singleton rows;
                    # keep the reference on its supported MLX pair input.
                    expected = getattr(old, name)(model, data, key, mx.array(geom))
                    actual = getattr(new, name)(model, data, key, geom)
                    mx.eval(*expected, *actual)
                    for e, v in zip(expected, actual):
                        np.testing.assert_allclose(np.asarray(v), np.asarray(e), atol=3e-6, rtol=2e-5)
                    empty = np.empty((0, 2), dtype=np.int32)
                    geom = empty if input_kind == 'numpy' else mx.array(empty)
                    actual = getattr(new, name)(model, data, key, geom)
                    mx.eval(*actual)
                    # The preserved old loop raises on an empty stack; the new
                    # grouped API must return a valid empty collision instead.
                    self.assertEqual([v.shape for v in actual], [(0,), (0, 3), (0, 3, 3)])


if __name__ == '__main__':
    unittest.main(verbosity=2)
