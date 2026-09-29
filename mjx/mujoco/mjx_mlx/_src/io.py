# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions to initialize, load, or save data.

MLX port: removes JAX device placement, WARP/C/CPP backends, and CUDA-specific
code. All arrays are created via mlx.core (mx) instead of jax.numpy (jnp).
Only the MLX (pure-array) backend is supported.
"""

import copy
import dataclasses
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mujoco
from mujoco.mjx_mlx._src import collision_driver
from mujoco.mjx_mlx._src import constraint
from mujoco.mjx_mlx._src import mesh
from mujoco.mjx_mlx._src import support
from mujoco.mjx_mlx._src import types
import numpy as np
import scipy


def _numpy_to_mlx(tree):
  """Convert all numpy arrays in a dataclass tree to MLX arrays."""
  if isinstance(tree, types.Data):
    # tree_map intentionally skips NumPy metadata, so it cannot perform this
    # initial conversion. Follow declared field types for Data and its nested
    # structures, retaining contact.dim/efc_address and efc_type on the host.
    def convert_data(node):
      updates = {}
      for field in dataclasses.fields(node):
        value = getattr(node, field.name)
        if field.type is mx.array and isinstance(
            value, (np.ndarray, np.generic, int, float, bool)
        ):
          updates[field.name] = mx.array(value)
        elif dataclasses.is_dataclass(value):
          updates[field.name] = convert_data(value)
      return dataclasses.replace(node, **updates)
    return convert_data(tree)
  from mujoco.mjx_mlx._src import dataclasses as mjx_dataclasses  # pylint: disable=g-import-not-at-top

  def _convert(value):
    if isinstance(value, mx.array):
      return value
    if isinstance(value, np.ndarray):
      return mx.array(value)
    return value

  if hasattr(tree, '__dataclass_fields__'):
    return mjx_dataclasses.tree_map(_convert, tree)
  return tree


def _put_option(
    o: mujoco.MjOption,
) -> types.Option:
  """Returns mjx_mlx.Option given mujoco.MjOption."""
  if o.integrator not in set(types.IntegratorType):
    raise NotImplementedError(f'{mujoco.mjtIntegrator(o.integrator)}')

  if o.cone not in set(types.ConeType):
    raise NotImplementedError(f'{mujoco.mjtCone(o.cone)}')

  if o.jacobian not in set(types.JacobianType):
    raise NotImplementedError(f'{mujoco.mjtJacobian(o.jacobian)}')

  if o.solver not in set(types.SolverType):
    raise NotImplementedError(f'{mujoco.mjtSolver(o.solver)}')

  for i in range(mujoco.mjtEnableBit.mjNENABLE):
    if o.enableflags & 2**i and 2 ** i not in set(types.EnableBit):
      raise NotImplementedError(f'{mujoco.mjtEnableBit(2**i)}')

  fields = {
      f.name: getattr(o, f.name, None)
      for f in types.Option.fields()
      if f.name != 'has_fluid_params'
  }
  fields['integrator'] = types.IntegratorType(o.integrator)
  fields['cone'] = types.ConeType(o.cone)
  fields['solver'] = types.SolverType(o.solver)
  fields['disableflags'] = types.DisableBit(o.disableflags)
  fields['enableflags'] = types.EnableBit(o.enableflags)
  fields['jacobian'] = types.JacobianType(o.jacobian)

  has_fluid_params = o.density > 0 or o.viscosity > 0 or o.wind.any()
  implicitfast = o.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  if implicitfast and has_fluid_params:
    raise NotImplementedError('implicitfast not implemented for fluid drag.')

  # Only pass fields that exist in our Option dataclass
  option_field_names = {f.name for f in types.Option.fields()}
  filtered = {k: v for k, v in fields.items() if k in option_field_names}
  if 'has_fluid_params' in option_field_names:
    filtered['has_fluid_params'] = has_fluid_params

  return types.Option(**filtered)


def _put_statistic(s: mujoco.MjStatistic) -> types.Statistic:
  """Puts mujoco.MjStatistic into mjx_mlx.Statistic."""
  return types.Statistic(
      meaninertia=s.meaninertia,
      meanmass=s.meanmass,
      meansize=s.meansize,
      extent=s.extent,
      center=s.center,
  )


def put_model(
    m: mujoco.MjModel,
) -> types.Model:
  """Converts mujoco.MjModel into an mjx_mlx.Model with MLX arrays.

  Args:
    m: the model to convert

  Returns:
    an mjx_mlx.Model with MLX arrays
  """
  if m.nflex:
    raise NotImplementedError('Flex not implemented for MLX backend.')

  # contact sensor checks
  is_contact_sensor = m.sensor_type == types.SensorType.CONTACT
  if is_contact_sensor.any():
    objtype = m.sensor_objtype[is_contact_sensor]
    reftype = m.sensor_reftype[is_contact_sensor]
    contact_sensor_type = set(np.concatenate([objtype, reftype]))

    if types.ObjType.SITE in set(objtype):
      raise NotImplementedError(
          'Contact sensor with site matching semantics not implemented for MLX'
          ' backend.'
      )
    if types.ObjType.BODY in contact_sensor_type:
      raise NotImplementedError(
          'Contact sensor with body matching semantics not implemented for MLX'
          ' backend.'
      )
    if types.ObjType.XBODY in contact_sensor_type:
      raise NotImplementedError(
          'Contact sensor with subtree matching semantics not implemented for'
          ' MLX backend.'
      )
    if (m.sensor_intprm[is_contact_sensor, 1] == 3).any():
      raise NotImplementedError(
          'Contact sensor with netforce reduction not implemented for MLX'
          ' backend.'
      )

  mesh_geomid = set()
  for g1, g2, ip in collision_driver.geom_pairs(m):
    t1, t2 = m.geom_type[[g1, g2]]
    if not collision_driver.has_collision_fn(t1, t2):
      t1, t2 = mujoco.mjtGeom(t1), mujoco.mjtGeom(t2)
      raise NotImplementedError(f'({t1}, {t2}) collisions not implemented.')
    no_margin = {mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_HFIELD}
    if no_margin.intersection({t1, t2}):
      if ip != -1:
        margin = m.pair_margin[ip]
      else:
        margin = m.geom_margin[g1] + m.geom_margin[g2]
      if margin.any():
        t1, t2 = mujoco.mjtGeom(t1), mujoco.mjtGeom(t2)
        raise NotImplementedError(f'({t1}, {t2}) margin/gap not implemented.')
    for t, g in [(t1, g1), (t2, g2)]:
      if t == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_geomid.add(g)

  for enum_field, enum_type, mj_type in (
      (m.actuator_biastype, types.BiasType, mujoco.mjtBias),
      (m.actuator_dyntype, types.DynType, mujoco.mjtDyn),
      (m.actuator_gaintype, types.GainType, mujoco.mjtGain),
      (m.actuator_trntype, types.TrnType, mujoco.mjtTrn),
      (m.eq_type, types.EqType, mujoco.mjtEq),
      (m.sensor_type, types.SensorType, mujoco.mjtSensor),
      (m.wrap_type, types.WrapType, mujoco.mjtWrap),
  ):
    missing = set(enum_field) - set(enum_type)
    if missing:
      raise NotImplementedError(
          f'{[mj_type(m) for m in missing]} not supported'
      )

  # Fields that only exist in our MLX types, not on C MjModel
  _mlx_only_fields = {
      'dof_hasfrictionloss', 'tendon_hasfrictionloss', 'geom_rbound_hfield',
      'wrap_inside_maxiter', 'wrap_inside_tolerance', 'wrap_inside_z_init',
      'is_wrap_inside', 'mesh_convex', 'opt', 'stat', '_impl',
  }
  mj_field_names = {f.name for f in types.Model.fields() if f.name not in _mlx_only_fields}
  # Only copy fields that exist on the C MjModel
  fields = {}
  for f in mj_field_names:
    if hasattr(m, f):
      fields[f] = getattr(m, f)
    else:
      # Use default from the dataclass
      field_obj = next((fld for fld in types.Model.fields() if fld.name == f), None)
      if field_obj is not None and field_obj.default is not dataclasses.MISSING:
        fields[f] = field_obj.default
      elif field_obj is not None and field_obj.default_factory is not dataclasses.MISSING:
        fields[f] = field_obj.default_factory()
      else:
        fields[f] = 0  # safe fallback
  fields['cam_mat0'] = fields['cam_mat0'].reshape((-1, 3, 3))
  fields['opt'] = _put_option(m.opt)
  fields['stat'] = _put_statistic(m.stat)

  # MLX-specific fields (equivalent to JAX impl fields, but inlined)
  fields['dof_hasfrictionloss'] = fields['dof_frictionloss'] > 0
  fields['tendon_hasfrictionloss'] = fields['tendon_frictionloss'] > 0
  fields['geom_rbound_hfield'] = fields['geom_rbound']

  # spatial tendon wrap inside
  fields['wrap_inside_maxiter'] = 5
  fields['wrap_inside_tolerance'] = 1.0e-4
  fields['wrap_inside_z_init'] = 1.0 - 1.0e-5
  fields['is_wrap_inside'] = np.zeros(0, dtype=bool)
  if m.nsite:
    (wrap_id_geom,) = np.nonzero(
        (m.wrap_type == mujoco.mjtWrap.mjWRAP_SPHERE)
        | (m.wrap_type == mujoco.mjtWrap.mjWRAP_CYLINDER)
    )
    wrap_objid_geom = m.wrap_objid[wrap_id_geom]
    geom_pos = m.geom_pos[wrap_objid_geom]
    geom_size = m.geom_size[wrap_objid_geom, 0]

    side_id = np.round(m.wrap_prm[wrap_id_geom]).astype(int)
    side = m.site_pos[side_id]

    fields['is_wrap_inside'] = np.array(
        (np.linalg.norm(side - geom_pos, axis=1) < geom_size) & (side_id >= 0)
    )

  # Pre-compile meshes for MJX collisions.
  fields['mesh_convex'] = [None] * m.nmesh
  for i in mesh_geomid:
    dataid = m.geom_dataid[i]
    if fields['mesh_convex'][dataid] is None:
      fields['mesh_convex'][dataid] = mesh.convex(m, dataid)
  fields['mesh_convex'] = tuple(fields['mesh_convex'])

  # Only pass fields that exist on our Model dataclass
  model_field_names = {f.name for f in types.Model.fields()}
  model_fields = {k: copy.copy(v) for k, v in fields.items() if k in model_field_names}
  model = types.Model(**model_fields)

  # Convert numpy arrays to MLX arrays
  model = _numpy_to_mlx(model)
  return model


def _make_data_public_fields(m: types.Model) -> Dict[str, Any]:
  """Create public fields for the Data object."""
  float_ = np.float32
  zero_fields = {
      'time': (float_,),
      'qvel': (m.nv, float_),
      'act': (m.na, float_),
      'history': (m.nhistory, float_),
      'plugin_state': (m.npluginstate, float_),
      'qacc_warmstart': (m.nv, float_),
      'ctrl': (m.nu, float_),
      'qfrc_applied': (m.nv, float_),
      'xfrc_applied': (m.nbody, 6, float_),
      'mocap_pos': (m.nmocap, 3, float_),
      'mocap_quat': (m.nmocap, 4, float_),
      'qacc': (m.nv, float_),
      'act_dot': (m.na, float_),
      'userdata': (m.nuserdata, float_),
      'sensordata': (m.nsensordata, float_),
      'xpos': (m.nbody, 3, float_),
      'xquat': (m.nbody, 4, float_),
      'xmat': (m.nbody, 3, 3, float_),
      'xipos': (m.nbody, 3, float_),
      'ximat': (m.nbody, 3, 3, float_),
      'xanchor': (m.njnt, 3, float_),
      'xaxis': (m.njnt, 3, float_),
      'geom_xpos': (m.ngeom, 3, float_),
      'geom_xmat': (m.ngeom, 3, 3, float_),
      'site_xpos': (m.nsite, 3, float_),
      'site_xmat': (m.nsite, 3, 3, float_),
      'cam_xpos': (m.ncam, 3, float_),
      'cam_xmat': (m.ncam, 3, 3, float_),
      'subtree_com': (m.nbody, 3, float_),
      'actuator_force': (m.nu, float_),
      'actuator_length': (m.nu, float_),
      'qfrc_bias': (m.nv, float_),
      'qfrc_gravcomp': (m.nv, float_),
      'qfrc_fluid': (m.nv, float_),
      'qfrc_passive': (m.nv, float_),
      'qfrc_actuator': (m.nv, float_),
      'qfrc_smooth': (m.nv, float_),
      'qacc_smooth': (m.nv, float_),
      'qfrc_constraint': (m.nv, float_),
      'qfrc_inverse': (m.nv, float_),
      'cvel': (m.nbody, 6, float_),
      'cdof': (m.nv, 6, float_),
      'cdof_dot': (m.nv, 6, float_),
      'ten_length': (m.ntendon, float_),
  }
  zero_fields = {
      k: np.zeros(v[:-1], dtype=v[-1]) for k, v in zero_fields.items()
  }
  return zero_fields


def _make_data_contact(
    condim: np.ndarray, efc_address: np.ndarray
) -> types.Contact:
  """Create contact for the Data object."""
  ncon = condim.size
  float_ = np.float32
  int_ = np.int32
  contact = types.Contact(
      dist=np.zeros((ncon,), dtype=float_),
      pos=np.zeros((ncon, 3), dtype=float_),
      frame=np.zeros((ncon, 3, 3), dtype=float_),
      includemargin=np.zeros((ncon,), dtype=float_),
      friction=np.zeros((ncon, 5), dtype=float_),
      solref=np.zeros((ncon, mujoco.mjNREF), dtype=float_),
      solreffriction=np.zeros((ncon, mujoco.mjNREF), dtype=float_),
      solimp=np.zeros((ncon, mujoco.mjNIMP), dtype=float_),
      dim=condim,
      geom1=np.full((ncon,), -1, dtype=int_),
      geom2=np.full((ncon,), -1, dtype=int_),
      geom=np.full((ncon, 2), -1, dtype=int_),
      efc_address=efc_address,
  )
  return contact


def make_data(
    m: Union[types.Model, mujoco.MjModel],
) -> types.Data:
  """Allocate and initialize Data for the MLX backend.

  Args:
    m: the model to use (mjx_mlx.Model or mujoco.MjModel)

  Returns:
    an initialized mjx_mlx.Data with MLX arrays
  """
  dim = collision_driver.make_condim(m)
  efc_type = constraint.make_efc_type(m, dim)
  ne, nf, nl, nc = constraint.counts(efc_type)
  ncon, nefc = dim.size, ne + nf + nl + nc
  efc_address = constraint.make_efc_address(m, dim, efc_type)

  float_ = np.float32
  int_ = np.int32
  contact = _make_data_contact(dim, efc_address)

  if m.opt.cone == types.ConeType.ELLIPTIC and np.any(contact.dim == 1):
    raise NotImplementedError(
        'condim=1 with ConeType.ELLIPTIC not implemented.'
    )

  zero_impl_fields = {
      'solver_niter': (int_,),
      'cinert': (m.nbody, 10, float_),
      'ten_wrapadr': (m.ntendon, np.int32),
      'ten_wrapnum': (m.ntendon, np.int32),
      'ten_J': (m.ntendon, m.nv, float_),
      'wrap_obj': (m.nwrap, 2, np.int32),
      'wrap_xpos': (m.nwrap, 6, float_),
      'actuator_moment': (m.nu, m.nv, float_),
      'crb': (m.nbody, 10, float_),
      'qM': (m.nM, float_) if support.is_sparse(m) else (m.nv, m.nv, float_),
      'M': (m.nC, float_),
      'qLD': (m.nC, float_) if support.is_sparse(m) else (m.nv, m.nv, float_),
      'qLDiagInv': (m.nv, float_) if support.is_sparse(m) else (0, float_),
      'ten_velocity': (m.ntendon, float_),
      'actuator_velocity': (m.nu, float_),
      'cacc': (m.nbody, 6, float_),
      'cfrc_int': (m.nbody, 6, float_),
      'cfrc_ext': (m.nbody, 6, float_),
      'subtree_linvel': (m.nbody, 3, float_),
      'subtree_angmom': (m.nbody, 3, float_),
      'efc_J': (nefc, m.nv, float_),
      'efc_pos': (nefc, float_),
      'efc_margin': (nefc, float_),
      'efc_frictionloss': (nefc, float_),
      'efc_D': (nefc, float_),
      'efc_aref': (nefc, float_),
      'efc_force': (nefc, float_),
  }
  zero_impl_fields = {
      k: np.zeros(v[:-1], dtype=v[-1]) for k, v in zero_impl_fields.items()
  }

  all_data_fields = {
      'ne': ne, 'nf': nf, 'nl': nl, 'nefc': nefc, 'ncon': ncon,
      'contact': contact, 'efc_type': efc_type,
      'qpos': np.array(m.qpos0, dtype=float_),
      'eq_active': m.eq_active0,
      **_make_data_public_fields(m),
      **zero_impl_fields,
  }
  # Only pass fields that exist on our Data dataclass
  data_field_names = {f.name for f in types.Data.fields()}
  filtered_data = {k: v for k, v in all_data_fields.items() if k in data_field_names}

  # Create DataMLX _impl with internal state arrays
  impl_fields = dict(ne=ne, nf=nf, nl=nl, nefc=nefc, ncon=ncon,
                     contact=contact, efc_type=efc_type)
  for k, v in zero_impl_fields.items():
    if isinstance(v, np.ndarray):
      impl_fields[k] = v
    elif isinstance(v, tuple) and len(v) >= 2:
      impl_fields[k] = np.zeros(v[:-1], dtype=v[-1])
    else:
      impl_fields[k] = v
  impl_field_names = {f.name for f in types.DataMLX.fields()}
  impl_filtered = {k: v for k, v in impl_fields.items() if k in impl_field_names}
  # Add required fields with defaults
  for f in types.DataMLX.fields():
    if f.name not in impl_filtered:
      if f.type == int:
        impl_filtered[f.name] = 0
      elif f.name == 'contact':
        impl_filtered[f.name] = contact
      elif f.name == 'efc_type':
        impl_filtered[f.name] = efc_type
      elif f.default is not dataclasses.MISSING:
        impl_filtered[f.name] = f.default
      elif f.default_factory is not dataclasses.MISSING:
        impl_filtered[f.name] = f.default_factory()
  data_impl = types.DataMLX(**impl_filtered)
  filtered_data['_impl'] = data_impl

  d = types.Data(**filtered_data)

  if m.nmocap:
    body_mask = m.body_mocapid >= 0
    body_pos = m.body_pos[body_mask]
    body_quat = m.body_quat[body_mask]
    d = d.replace(
        mocap_pos=body_pos[m.body_mocapid[body_mask]],
        mocap_quat=body_quat[m.body_mocapid[body_mask]],
    )

  # Convert numpy arrays to MLX arrays
  d = _numpy_to_mlx(d)
  return d


def _put_contact(
    c: mujoco._structs._MjContactList,
    dim: np.ndarray,
    efc_address: np.ndarray,
) -> Tuple[types.Contact, np.ndarray]:
  """Converts mujoco.structs._MjContactList into mjx_mlx.Contact."""
  fields = {f.name: getattr(c, f.name) for f in types.Contact.fields()}
  fields['frame'] = fields['frame'].reshape((-1, 3, 3))

  contact_map = -np.ones_like(dim)
  for i, di in enumerate(fields['dim']):
    space = [j for j, dj in enumerate(dim) if di == dj and contact_map[j] == -1]
    if not space:
      raise ValueError(f'unable to place Contact[{i}], no space in condim {di}')
    contact_map[space[0]] = i

  if contact_map.size > 0:
    # reorganize contact with a zero contact at the end for -1
    zero = {
        k: np.zeros((1,) + v.shape[1:], dtype=v.dtype) for k, v in fields.items()
    }
    zero['dist'][:] = 1e10
    for name in ('geom', 'geom1', 'geom2'):
      zero[name][:] = -1
    fields = {k: np.concatenate([fields[k], zero[k]]) for k in fields}
    fields = {k: v[contact_map] for k, v in fields.items()}

  fields['dim'] = dim
  fields['efc_address'] = efc_address

  return types.Contact(**fields), contact_map


def _put_data_public_fields(d: mujoco.MjData) -> Dict[str, Any]:
  """Returns public fields from mujoco.MjData in a dictionary."""
  fields = {
      f.name: getattr(d, f.name)
      for f in types.Data.fields()
      if f.name != '_impl'
  }
  # MJX uses square matrices for these fields:
  for fname in ('xmat', 'ximat', 'geom_xmat', 'site_xmat', 'cam_xmat'):
    if fname in fields:
      fields[fname] = fields[fname].reshape((-1, 3, 3))

  return fields


def _noncontact_row_keys(m: mujoco.MjModel):
  """Static (constraint type, object id, component) order used by assembly.

  Native MuJoCo omits inactive equalities and limits.  Their remaining rows
  cannot be copied into the prefix of a fixed-size MLX constraint group.
  """
  disable = m.opt.disableflags
  if disable & types.DisableBit.CONSTRAINT:
    return []
  keys = []
  if not disable & types.DisableBit.EQUALITY:
    for eq_type, width in ((types.EqType.CONNECT, 3), (types.EqType.WELD, 6),
                           (types.EqType.JOINT, 1), (types.EqType.TENDON, 1)):
      for obj in np.flatnonzero(m.eq_type == eq_type):
        keys.extend((int(types.ConstraintType.EQUALITY), int(obj), component)
                    for component in range(width))
  if not disable & types.DisableBit.FRICTIONLOSS:
    for kind, ids in (
        (types.ConstraintType.FRICTION_DOF, np.flatnonzero(m.dof_frictionloss > 0)),
        (types.ConstraintType.FRICTION_TENDON, np.flatnonzero(m.tendon_frictionloss > 0)),
    ):
      keys.extend((int(kind), int(obj), 0) for obj in ids)
  if not disable & types.DisableBit.LIMIT:
    # make_constraint assembles ball joints before slide/hinge joints.
    ball = (m.jnt_type == types.JointType.BALL) & m.jnt_limited
    other = np.isin(m.jnt_type, (types.JointType.SLIDE, types.JointType.HINGE)) & m.jnt_limited
    for ids in (np.flatnonzero(ball), np.flatnonzero(other)):
      keys.extend((int(types.ConstraintType.LIMIT_JOINT), int(obj), 0) for obj in ids)
    keys.extend((int(types.ConstraintType.LIMIT_TENDON), int(obj), 0)
                for obj in np.flatnonzero(m.tendon_limited))
  return keys


def _noncontact_row_map(m: mujoco.MjModel, d: mujoco.MjData,
                        efc_type: np.ndarray, count: int) -> np.ndarray:
  """Return static destination rows for the materialized native rows."""
  keys = _noncontact_row_keys(m)
  if len(keys) != count or not np.array_equal(
      np.asarray([key[0] for key in keys]), efc_type[:count]):
    raise NotImplementedError('Unsupported static non-contact constraint layout.')
  slots = {key: row for row, key in enumerate(keys)}
  occurrences = {}
  result = []
  for row in range(d.ne + d.nf + d.nl):
    kind, obj = int(d.efc_type[row]), int(d.efc_id[row])
    component = occurrences.get((kind, obj), 0)
    occurrences[kind, obj] = component + 1
    key = kind, obj, component
    if key not in slots:
      raise NotImplementedError(f'Cannot map native constraint row {row}: {key}.')
    result.append(slots[key])
  return np.asarray(result, dtype=np.int32)


def put_data(
    m: mujoco.MjModel,
    d: mujoco.MjData,
) -> types.Data:
  """Puts mujoco.MjData into an mjx_mlx.Data with MLX arrays.

  Args:
    m: the model to use
    d: the data to convert

  Returns:
    an mjx_mlx.Data with MLX arrays
  """
  if not isinstance(m, mujoco.MjModel) or not isinstance(d, mujoco.MjData):
    raise TypeError('put_data requires one native MjModel and MjData.')
  dim = collision_driver.make_condim(m)
  efc_type = constraint.make_efc_type(m, dim)
  efc_address = constraint.make_efc_address(m, dim, efc_type)
  ne, nf, nl, nc = constraint.counts(efc_type)
  ncon, nefc = dim.size, ne + nf + nl + nc

  for d_val, val, name in (
      (d.ncon, ncon, 'ncon'),
      (d.ne, ne, 'ne'),
      (d.nf, nf, 'nf'),
      (d.nl, nl, 'nl'),
      (d.nefc, nefc, 'nefc'),
  ):
    if d_val > val:
      raise ValueError(f'd.{name} too high, d.{name} = {d_val}, model = {val}')

  fields = _put_data_public_fields(d)

  # Implementation specific fields.
  impl_field_names = (
      'solver_niter', 'cinert', 'ten_wrapadr', 'ten_wrapnum', 'ten_J',
      'wrap_obj', 'wrap_xpos', 'actuator_moment', 'crb', 'qM', 'M',
      'qLD', 'qLDiagInv', 'ten_velocity', 'actuator_velocity', 'cacc',
      'cfrc_int', 'cfrc_ext', 'subtree_linvel', 'subtree_angmom',
      'efc_J', 'efc_pos', 'efc_margin', 'efc_frictionloss', 'efc_D',
      'efc_aref', 'efc_force',
  )
  impl_fields = {
      name: getattr(d, name)
      for name in impl_field_names
      if hasattr(d, name)
  }
  # MJX does not support islanding, so only transfer the first solver_niter
  impl_fields['solver_niter'] = np.asarray(impl_fields['solver_niter'][0])

  # convert sparse actuator_moment to dense matrix
  moment = np.zeros((m.nu, m.nv))
  mujoco.mju_sparse2dense(
      moment,
      d.actuator_moment,
      d.moment_rownnz,
      d.moment_rowadr,
      d.moment_colind,
  )
  impl_fields['actuator_moment'] = moment

  # convert ten_J to dense matrix
  if m.ntendon:
    ten_J = np.zeros((m.ntendon, m.nv))
    if hasattr(m, 'ten_J_rownnz'):
      # Newer MuJoCo versions store fixed tendon sparsity on Model.
      mujoco.mju_sparse2dense(ten_J, d.ten_J, m.ten_J_rownnz,
                             m.ten_J_rowadr, m.ten_J_colind)
    elif mujoco.mj_isSparse(m):
      # MuJoCo 3.5 keeps a padded sparse buffer and its indices on Data.
      mujoco.mju_sparse2dense(ten_J, d.ten_J.reshape(-1), d.ten_J_rownnz,
                             d.ten_J_rowadr, d.ten_J_colind.reshape(-1))
    else:
      ten_J[:] = d.ten_J.reshape((m.ntendon, m.nv))
  else:
    ten_J = np.zeros((m.ntendon, m.nv))
  impl_fields['ten_J'] = ten_J

  contact, contact_map = _put_contact(d.contact, dim, efc_address)
  noncontact_map = _noncontact_row_map(m, d, efc_type, ne + nf + nl)

  # pad efc fields: MuJoCo efc arrays are sparse for inactive constraints.
  # efc_J is also optionally column-sparse (typically for large nv). MJX-MLX
  # is neither: it contains zeros for inactive constraints, and efc_J is always
  # (nefc, nv). This may change in the future.
  if mujoco.mj_isSparse(m):
    efc_j = np.zeros((d.efc_J_rownnz.shape[0], m.nv))
    mujoco.mju_sparse2dense(
        efc_j,
        impl_fields['efc_J'],
        d.efc_J_rownnz,
        d.efc_J_rowadr,
        d.efc_J_colind,
    )
    impl_fields['efc_J'] = efc_j
  else:
    impl_fields['efc_J'] = impl_fields['efc_J'].reshape(
        (-1 if m.nv else 0, m.nv)
    )

  # move efc rows to their correct offsets
  for fname in (
      'efc_J',
      'efc_pos',
      'efc_margin',
      'efc_frictionloss',
      'efc_D',
      'efc_aref',
      'efc_force',
  ):
    value = np.zeros((nefc, m.nv)) if fname == 'efc_J' else np.zeros(nefc)
    value[noncontact_map] = impl_fields[fname][:len(noncontact_map)]

    for id_to, id_from in enumerate(contact_map):
      if id_from == -1:
        continue
      num_rows = dim[id_to]
      if num_rows > 1 and m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL:
        num_rows = (num_rows - 1) * 2
      efc_i, efc_o = d.contact.efc_address[id_from], efc_address[id_to]
      if efc_i == -1:
        continue
      value[efc_o : efc_o + num_rows] = impl_fields[fname][
          efc_i : efc_i + num_rows
      ]

    impl_fields[fname] = value

  # convert qM and qLD if jacobian is dense
  if not support.is_sparse(m):
    impl_fields['qM'] = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, impl_fields['qM'], d.qM)
    try:
      # smooth.solve_m consumes a lower-triangular Cholesky factor.
      impl_fields['qLD'] = scipy.linalg.cholesky(impl_fields['qM'], lower=True)
    except scipy.linalg.LinAlgError:
      impl_fields['qLD'] = np.zeros((m.nv, m.nv))
    impl_fields['qLDiagInv'] = np.zeros(0)

  impl_fields.update(ne=ne, nf=nf, nl=nl, nefc=nefc, ncon=ncon,
                     contact=contact, efc_type=efc_type)
  # The native arrays are mutable views: import an independent snapshot.
  data = types.Data(
      _impl=types.DataMLX(**{k: copy.copy(v) for k, v in impl_fields.items()}),
      **{k: copy.copy(v) for k, v in fields.items()},
  )

  # Convert numpy arrays to MLX arrays
  data = _numpy_to_mlx(data)
  return data


def _contact_indices(m: mujoco.MjModel, cx: types.Contact) -> np.ndarray:
  """Select materialized contacts, excluding padding and separated slots.

  Unlike a dist <= 0 filter, this retains positive-distance contacts generated
  inside a contact margin, including contacts that are inactive due to gap.
  """
  geom = np.asarray(cx.geom)
  valid = np.all((geom >= 0) & (geom < m.ngeom), axis=1)
  ids = np.flatnonzero(valid)
  if not ids.size:
    return ids
  pairs = geom[ids]
  margin = np.sum(m.geom_margin[pairs], axis=1)
  for i in range(m.npair):
    pair = np.array([m.pair_geom1[i], m.pair_geom2[i]])
    match = np.all(pairs == pair, axis=1) | np.all(pairs == pair[::-1], axis=1)
    margin[match] = m.pair_margin[i]
  if m.opt.enableflags & mujoco.mjtEnableBit.mjENBL_OVERRIDE:
    margin[:] = m.opt.o_margin
  threshold = np.maximum(margin, np.asarray(cx.includemargin)[ids])
  return ids[np.asarray(cx.dist)[ids] <= threshold]


def _get_contact(c: mujoco._structs._MjContactList, cx: types.Contact,
                 con_id: np.ndarray):
  """Converts mjx_mlx.Contact to mujoco._structs._MjContactList."""
  # Convert MLX arrays to numpy for writing back to MuJoCo structs
  for field in types.Contact.fields():
    value = getattr(cx, field.name)
    if isinstance(value, mx.array):
      value = np.array(value)
    value = value[con_id]
    if field.name == 'frame':
      value = value.reshape((-1, 9))
    getattr(c, field.name)[:] = value


def get_data_into(
    result: Union[mujoco.MjData, List[mujoco.MjData]],
    m: mujoco.MjModel,
    d: types.Data,
):
  """Export one MLX state; batched-environment conversion is not supported."""
  if isinstance(result, list) or len(d.qpos.shape) != 1:
    raise NotImplementedError('Batched Data export is not supported by MLX.')
  if not isinstance(result, mujoco.MjData):
    raise TypeError('destination must be one native MjData.')
  if not isinstance(d._impl, types.DataMLX):
    raise TypeError('Data must contain its nested DataMLX implementation.')
  if d.qpos.shape != (m.nq,) or d.qvel.shape != (m.nv,):
    raise ValueError('Data state dimensions do not match the native model.')

  # Convert the full tree to numpy
  from mujoco.mjx_mlx._src import dataclasses as mjx_dataclasses  # pylint: disable=g-import-not-at-top
  d = mjx_dataclasses.tree_map(
      lambda x: np.array(x) if isinstance(x, mx.array) else x, d
  )

  dof_i, dof_j = [], []
  for i in range(m.nv):
    j = i
    while j > -1:
      dof_i.append(i)
      dof_j.append(j)
      j = int(m.dof_parentid[j])

  # Keep the export body scoped separately from the conversion above.
  for d_i, result_i in ((d, result),):
    impl = d_i._impl
    con_id = _contact_indices(m, impl.contact)
    ncon = int(con_id.size)
    efc_active = (impl.efc_J != 0).any(axis=1)
    nefc = int(efc_active.sum())
    sparse = bool(mujoco.mj_isSparse(m))
    nj = int((impl.efc_J != 0).sum()) if sparse else nefc * m.nv

    if ncon != result_i.ncon or nefc != result_i.nefc or nj != result_i.nJ:
      mujoco._functions._realloc_con_efc(result_i, ncon=ncon, nefc=nefc, nJ=nj)  # pylint: disable=protected-access

    keys = _noncontact_row_keys(m)
    if len(keys) != impl.ne + impl.nf + impl.nl:
      raise NotImplementedError('Unsupported static non-contact constraint layout.')
    # Native efc_id refers to the original object for non-contact rows and to
    # the *compacted* contact index for contact rows.
    row_ids = np.full(impl.nefc, -1, dtype=np.int32)
    row_ids[:len(keys)] = [key[1] for key in keys]
    for compact_id, slot in enumerate(con_id):
      address = int(impl.contact.efc_address[slot])
      width = int(impl.contact.dim[slot])
      if width > 1 and m.opt.cone == types.ConeType.PYRAMIDAL:
        width = 2 * (width - 1)
      if address >= 0:
        row_ids[address:address + width] = compact_id
    if np.any(row_ids[efc_active] < 0):
      raise ValueError('Active constraint rows have no exportable object/contact.')
    result_i.efc_id[:] = row_ids[efc_active]

    for field in (*types.Data.fields(), *types.DataMLX.fields()):
      if field.name not in mujoco.MjData.__dict__.keys():
        continue

      if field.name == 'contact':
        _get_contact(result_i.contact, impl.contact, con_id)
        efc_map = np.cumsum(efc_active) - 1
        address = np.asarray(impl.contact.efc_address)[con_id]
        mapped = np.full(ncon, -1, dtype=np.int32)
        valid = (address >= 0) & (address < efc_active.size)
        active = np.zeros(ncon, dtype=bool)
        active[valid] = efc_active[address[valid]]
        mapped[active] = efc_map[address[active]]
        result_i.contact.efc_address[:] = mapped
        continue

      # MuJoCo actuator_moment is sparse, MJX uses a dense representation.
      if field.name == 'actuator_moment':
        moment_rownnz = np.zeros(m.nu, dtype=np.int32)
        moment_rowadr = np.zeros(m.nu, dtype=np.int32)
        moment_colind = np.zeros(m.nJmom, dtype=np.int32)
        actuator_moment = np.zeros(m.nJmom)
        if m.nu:
          mujoco.mju_dense2sparse(
              actuator_moment,
              np.ascontiguousarray(impl.actuator_moment, dtype=np.float64),
              moment_rownnz,
              moment_rowadr,
              moment_colind,
          )
        result_i.moment_rownnz[:] = moment_rownnz
        result_i.moment_rowadr[:] = moment_rowadr
        result_i.moment_colind[:] = moment_colind
        result_i.actuator_moment[:] = actuator_moment
        continue

      # MuJoCo ten_J is sparse, MJX uses a dense representation.
      if field.name == 'ten_J':
        if hasattr(m, 'ten_J_rownnz'):
          ten_j = np.zeros(result_i.ten_J.shape)
          # Preserve Model-owned structural zeros and ordering.
          for row in range(m.ntendon):
            start = m.ten_J_rowadr[row]
            end = start + m.ten_J_rownnz[row]
            ten_j[start:end] = impl.ten_J[row, m.ten_J_colind[start:end]]
          result_i.ten_J[:] = ten_j
        elif sparse and m.ntendon:
          # MuJoCo 3.5 owns dynamic sparsity metadata on Data instead.
          ten_j = np.zeros(result_i.ten_J.size)
          rownnz = np.zeros(m.ntendon, dtype=np.int32)
          rowadr = np.zeros(m.ntendon, dtype=np.int32)
          colind = np.zeros(result_i.ten_J_colind.size, dtype=np.int32)
          mujoco.mju_dense2sparse(
              ten_j, np.ascontiguousarray(impl.ten_J, dtype=np.float64),
              rownnz, rowadr, colind)
          result_i.ten_J[:] = ten_j.reshape(result_i.ten_J.shape)
          result_i.ten_J_rownnz[:] = rownnz
          result_i.ten_J_rowadr[:] = rowadr
          result_i.ten_J_colind[:] = colind.reshape(result_i.ten_J_colind.shape)
        else:
          result_i.ten_J[:] = impl.ten_J
        continue

      value = getattr(impl, field.name) if hasattr(impl, field.name) else getattr(d_i, field.name)

      if field.name in ('ne', 'nf', 'nl'):
        starts = {'ne': 0, 'nf': impl.ne, 'nl': impl.ne + impl.nf}
        start = starts[field.name]
        value = int(efc_active[start:start + value].sum())
      elif field.name in ('nefc', 'ncon'):
        value = {'nefc': nefc, 'ncon': ncon}[field.name]
      elif field.name.endswith('xmat') or field.name == 'ximat':
        value = value.reshape((-1, 9))
      elif field.name == 'efc_J':
        value = value[efc_active]
        if sparse:
          efc_J_rownnz = np.zeros(nefc, dtype=np.int32)
          efc_J_rowadr = np.zeros(nefc, dtype=np.int32)
          efc_J_colind = np.zeros(nj, dtype=np.int32)
          efc_J = np.zeros(nj)
          mujoco.mju_dense2sparse(
              efc_J,
              np.ascontiguousarray(value, dtype=np.float64),
              efc_J_rownnz,
              efc_J_rowadr,
              efc_J_colind,
          )
          result_i.efc_J_rownnz[:] = efc_J_rownnz
          result_i.efc_J_rowadr[:] = efc_J_rowadr
          result_i.efc_J_colind[:] = efc_J_colind
          value = efc_J
        else:
          value = value.reshape(-1)
      elif field.name.startswith('efc_'):
        value = value[efc_active]
      elif field.name == 'solver_niter':
        result_i.solver_niter[:] = 0
        result_i.solver_niter[0] = int(value)
        continue
      if field.name == 'qM' and not support.is_sparse(m):
        value = value[dof_i, dof_j]
      elif field.name == 'qLD':
        value = np.zeros(m.nC)
      elif field.name == 'qLDiagInv' and not support.is_sparse(m):
        value = np.ones(m.nv)

      if isinstance(value, np.ndarray) and value.shape:
        result_field = getattr(result_i, field.name)
        if result_field.shape != value.shape:
          raise ValueError(
              f'Input field {field.name} has shape {value.shape}, but output'
              f' has shape {result_field.shape}'
          )
        result_field[:] = value
      else:
        setattr(result_i, field.name, value)

    # map inertia (sparse) to reduced inertia (compressed sparse)
    result_i.M[:] = result_i.qM[m.mapM2M]

    # recalculate qLD and qLDiagInv
    mujoco.mj_factorM(m, result_i)


def get_data(
    m: mujoco.MjModel,
    d: types.Data,
) -> mujoco.MjData:
  """Gets a single MLX state as native MjData (no batch inference)."""
  if len(d.qpos.shape) != 1:
    raise NotImplementedError('Batched Data export is not supported by MLX.')
  result = mujoco.MjData(m)

  get_data_into(result, m, d)
  return result


# ---------------------------------------------------------------------------
# State get/set utilities
# ---------------------------------------------------------------------------

_STATE_MAP = {
    mujoco.mjtState.mjSTATE_TIME: 'time',
    mujoco.mjtState.mjSTATE_QPOS: 'qpos',
    mujoco.mjtState.mjSTATE_QVEL: 'qvel',
    mujoco.mjtState.mjSTATE_ACT: 'act',
    mujoco.mjtState.mjSTATE_HISTORY: 'history',
    mujoco.mjtState.mjSTATE_WARMSTART: 'qacc_warmstart',
    mujoco.mjtState.mjSTATE_CTRL: 'ctrl',
    mujoco.mjtState.mjSTATE_QFRC_APPLIED: 'qfrc_applied',
    mujoco.mjtState.mjSTATE_XFRC_APPLIED: 'xfrc_applied',
    mujoco.mjtState.mjSTATE_EQ_ACTIVE: 'eq_active',
    mujoco.mjtState.mjSTATE_MOCAP_POS: 'mocap_pos',
    mujoco.mjtState.mjSTATE_MOCAP_QUAT: 'mocap_quat',
    mujoco.mjtState.mjSTATE_USERDATA: 'userdata',
    mujoco.mjtState.mjSTATE_PLUGIN: 'plugin_state',
}


def _state_elem_size(m: types.Model, state_enum: mujoco.mjtState) -> int:
  """Returns the size of a state component."""
  if state_enum not in _STATE_MAP:
    raise ValueError(f'Invalid state element {state_enum}')
  name = _STATE_MAP[state_enum]
  if name == 'time':
    return 1
  if name in (
      'qpos',
      'qvel',
      'act',
      'history',
      'qacc_warmstart',
      'ctrl',
      'qfrc_applied',
      'eq_active',
      'mocap_pos',
      'mocap_quat',
      'userdata',
      'plugin_state',
  ):
    val = getattr(
        m,
        {
            'qpos': 'nq',
            'qvel': 'nv',
            'act': 'na',
            'history': 'nhistory',
            'qacc_warmstart': 'nv',
            'ctrl': 'nu',
            'qfrc_applied': 'nv',
            'eq_active': 'neq',
            'mocap_pos': 'nmocap',
            'mocap_quat': 'nmocap',
            'userdata': 'nuserdata',
            'plugin_state': 'npluginstate',
        }[name],
    )
    if name == 'mocap_pos':
      val *= 3
    if name == 'mocap_quat':
      val *= 4
    return val
  if name == 'xfrc_applied':
    return 6 * m.nbody

  raise NotImplementedError(f'state component {name} not implemented')


def state_size(m: types.Model, spec: Union[int, mujoco.mjtState]) -> int:
  """Returns the size of a state vector for a given spec.

  Args:
    m: model describing the simulation
    spec: int bitmask or mjtState enum specifying which state components to
      include

  Returns:
    size of the state vector
  """
  size = 0
  spec_int = int(spec)
  for i in range(mujoco.mjtState.mjNSTATE.value):
    element = mujoco.mjtState(1 << i)
    if element & spec_int:
      size += _state_elem_size(m, element)
  return size


def get_state(
    m: types.Model, d: types.Data, spec: Union[int, mujoco.mjtState]
) -> mx.array:
  """Gets state from mjx_mlx.Data. Equivalent to `mujoco.mj_getState`.

  Args:
    m: model describing the simulation
    d: data for the simulation
    spec: int bitmask or mjtState enum specifying which state components to
      include

  Returns:
    a flat MLX array of state values
  """
  spec_int = int(spec)
  if spec_int >= (1 << mujoco.mjtState.mjNSTATE.value):
    raise ValueError(f'Invalid state spec {spec}')

  state = []
  for i in range(mujoco.mjtState.mjNSTATE.value):
    element = mujoco.mjtState(1 << i)
    if element & spec_int:
      if element not in _STATE_MAP:
        raise ValueError(f'Invalid state element {element}')
      name = _STATE_MAP[element]
      value = getattr(d, name)
      if not isinstance(value, mx.array):
        value = mx.array(value)
      if element == mujoco.mjtState.mjSTATE_EQ_ACTIVE:
        value = value.astype(mx.float32)
      state.append(mx.reshape(value, (-1,)))

  return mx.concatenate(state) if state else mx.array([])


def set_state(
    m: types.Model,
    d: types.Data,
    state: mx.array,
    spec: Union[int, mujoco.mjtState],
) -> types.Data:
  """Sets state in mjx_mlx.Data. Equivalent to `mujoco.mj_setState`.

  Args:
    m: model describing the simulation
    d: data for the simulation
    state: a flat MLX array of state values
    spec: int bitmask or mjtState enum specifying which state components to
      include

  Returns:
    data with state set to provided values
  """
  spec_int = int(spec)
  if spec_int >= (1 << mujoco.mjtState.mjNSTATE.value):
    raise ValueError(f'Invalid state spec {spec}')

  expected_size = state_size(m, spec)
  if state.size != expected_size:
    raise ValueError(
        f'state has size {state.size} but expected {expected_size}'
    )

  updates = {}
  offset = 0
  for i in range(mujoco.mjtState.mjNSTATE.value):
    element = mujoco.mjtState(1 << i)
    if element & spec_int:
      if element not in _STATE_MAP:
        raise ValueError(f'Invalid state element {element}')
      name = _STATE_MAP[element]
      size = _state_elem_size(m, element)
      value = state[offset : offset + size]
      if name == 'time':
        value = value[0]
      else:
        orig = getattr(d, name)
        orig_shape = orig.shape if isinstance(orig, mx.array) else np.array(orig).shape
        value = mx.reshape(value, orig_shape)
      if element == mujoco.mjtState.mjSTATE_EQ_ACTIVE:
        value = value.astype(mx.bool_)
      updates[name] = value
      offset += size

  return d.replace(**updates)
