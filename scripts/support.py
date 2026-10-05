"""Run evaluation commands with isolated outputs and CPU execution."""
from pathlib import Path
import os
import subprocess
import sys
import json
ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('MOSA_EVAL_OUTPUT', str(ROOT / 'build'))).resolve()
WORK.mkdir(parents=True, exist_ok=True)

def environment(source):
    env = os.environ.copy()
    for name in ['tmp', 'cache', 'mpl', 'numba']:
        (WORK / 'runtime' / name).mkdir(parents=True, exist_ok=True)
    env.update(MPLBACKEND='Agg', PYTHONFAULTHANDLER='1', PYTHONPATH=str(Path(source).resolve() / 'src') + os.pathsep + str(ROOT / 'scripts'), PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(WORK / 'runtime/tmp'), TMP=str(WORK / 'runtime/tmp'), TEMP=str(WORK / 'runtime/tmp'), XDG_CACHE_HOME=str(WORK / 'runtime/cache'), MPLCONFIGDIR=str(WORK / 'runtime/mpl'), NUMBA_CACHE_DIR=str(WORK / 'runtime/numba'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    return env

def command(args, source, log, extra_env=None, timeout=900):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = environment(source)
    env.update(extra_env or {})
    with log.open('w') as f:
        f.write('COMMAND ' + json.dumps([str(x) for x in args]) + '\n')
        f.flush()
        try:
            proc = subprocess.run([str(x) for x in args], cwd=source, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            f.write('\nTIMEOUT\n')
            code = 124
    return {'command': [str(x) for x in args], 'exit_code': code, 'log': str(log.relative_to(WORK))}

def pytest_run(source, name, selection=None):
    artifact = WORK / 'tests' / name
    artifact.mkdir(parents=True, exist_ok=True)
    report = artifact / 'tests.json'
    args = [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'measurement_plugin', '-p', 'no:cacheprovider', '--basetemp', str(artifact / 'tmp')] + (selection or [])
    result = command(args, source, artifact / 'pytest.log', {'MEASUREMENT_JSON': str(report)})
    if report.exists():
        result.update(json.loads(report.read_text()))
    return result
