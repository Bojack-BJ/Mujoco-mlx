"""Run all upstream smoke functions, without the unrelated math benchmark."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import mujoco

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parent
mujoco.__path__.append(str(root))
spec = importlib.util.spec_from_file_location('upstream_mlx_smoke', root / 'mjx_mlx/test_mlx_physics.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module  # Required by dataclasses' annotation lookup.
log, results = io.StringIO(), []
with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
    spec.loader.exec_module(module)
    for name, fn in vars(module).items():
        if name.startswith('test_') and callable(fn):
            try:
                result = fn()
                results.append(dict(name=name, status='skipped' if result is False else 'passed'))
            except Exception as exc:
                results.append(dict(name=name, status='failed', error=repr(exc)))
report = dict(tests=results, log=log.getvalue(),
              scope='All upstream test functions; many are import-only. Excludes math throughput benchmark.')
for status in ('passed', 'failed', 'skipped'):
    report[status] = sum(r['status'] == status for r in results)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ('passed', 'failed', 'skipped')}))
sys.exit(bool(report['failed']))
