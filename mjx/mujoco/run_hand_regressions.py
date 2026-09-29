"""Run only repair regressions; avoid importing the adjacent JAX test tree."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest

p = argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a = p.parse_args()
log = io.StringIO()
with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        'test_collision_optimization','test_constraint_batch','test_solver_batch'])
    result = unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
report = dict(tests_run=result.testsRun,success=result.wasSuccessful(),
              failures=len(result.failures),errors=len(result.errors),
              skipped=len(result.skipped),log=log.getvalue())
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(log.getvalue())
sys.exit(0 if result.wasSuccessful() else 1)
