"""Frozen official MJX 3.5.0 box-contact oracle, not a native-equivalence test.

Regenerate the oracle with audit_contact_parity.py. The report records the
official source hash and deliberately retains its shallow-contact omissions.
"""
import json
from pathlib import Path
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
mujoco.__path__.append(str(Path(__file__).resolve().parent))
import mlx.core as mx
from mujoco.mjx_mlx._src import collision_convex, mesh
from mujoco.mjx_mlx._src.collision_types import GeomInfo


def kernel(p1, r1, s1, p2, r2, s2):
    return collision_convex._box_box(mesh.box(GeomInfo(p1, r1, s1)),
                                     mesh.box(GeomInfo(p2, r2, s2)))


class OfficialBoxContacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / 'validation/contact-parity/collision-three-way.json').read_text())
        assert cls.report['status'] == 'completed'
        assert cls.report['mujoco'] == '3.5.0'
        assert len(cls.report['cases']) == 15

    def check_outputs(self, values, expected):
        for key, actual in zip(('dist', 'pos', 'normal'), values):
            np.testing.assert_allclose(np.asarray(actual), expected[key],
                                       rtol=0, atol=2e-6, err_msg=key)
        # Numeric closeness alone can hide a contact activation change.
        np.testing.assert_array_equal(np.asarray(values[0]) < 0,
                                      np.asarray(expected['dist']) < 0)

    def test_scalar_against_official_mjx(self):
        for case in self.report['cases']:
            with self.subTest(case=case['name']):
                values = kernel(*(mx.array(v) for v in case['geometry']))
                self.check_outputs(values, case['official_jax'])

    def test_mixed_case_batch_against_official_mjx(self):
        inputs = [mx.stack([mx.array(c['geometry'][i]) for c in self.report['cases']])
                  for i in range(6)]
        values = mx.vmap(kernel)(*inputs)
        for i, case in enumerate(self.report['cases']):
            with self.subTest(case=case['name']):
                self.check_outputs(tuple(v[i] for v in values), case['official_jax'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
