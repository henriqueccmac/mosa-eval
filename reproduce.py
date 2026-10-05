"""Reproduce the software evaluation reported in thesis Section 4.5."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
ROOT = Path(__file__).resolve().parent
GROUPS = ('scenarios', 'alternatives', 'formats', 'mofa')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=GROUPS, help='Run one part of the evaluation.')
    parser.add_argument('--scenario', choices=['baseline2', 'baseline3', 'D1', 'D2', 'D3', 'D4', 'D5', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7'], help='Run one baseline or extension, e.g. baseline2 or A4.')
    parser.add_argument('--output', type=Path, default=ROOT / 'build', help='New output directory (default: build).')
    args = parser.parse_args()
    if args.scenario and args.group not in (None, 'scenarios'):
        parser.error('--scenario requires --group scenarios')
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory is not empty; choose a new --output directory.')
    if sys.flags.optimize:
        parser.error('Run without -O: evaluation assertions must be enabled.')
    provenance = json.loads((ROOT / 'source.json').read_text())
    for name, expected in provenance['files'].items():
        actual = hashlib.sha256((ROOT / 'source' / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f'Source snapshot differs: {name}')
    inputs = json.loads((ROOT / 'data/manifest.json').read_text())
    for name, expected in inputs['sha256'].items():
        if hashlib.sha256((ROOT / 'data' / name).read_bytes()).hexdigest() != expected:
            raise SystemExit(f'Synthetic input differs: {name}')
    os.environ['MOSA_EVAL_OUTPUT'] = str(out)
    sys.path.insert(0, str(ROOT / 'scripts'))
    from support import command, environment, pytest_run
    os.environ.update(environment(ROOT / 'source'))
    out.mkdir(parents=True, exist_ok=True)
    groups = ['scenarios'] if args.scenario else [args.group] if args.group else list(GROUPS)
    summary = {'source_revision': provenance['revision'], 'groups': {}}
    for group in groups:
        print(f'Running {group}...', flush=True)
        cmd = [sys.executable, '-B', str(ROOT / 'scripts' / f'{group}.py')]
        if group == 'scenarios':
            cmd.append(str(ROOT / 'source'))
            if args.scenario:
                cmd.extend(['--only', args.scenario])
        log = out / f'{group}.log'
        result = command(cmd, ROOT / 'source', log, timeout=7200)
        if group == 'mofa' and result['exit_code'] == 0:
            result['api_tests'] = pytest_run(ROOT / 'source', 'mofa', ['tests/test_api.py', '-k', 'mofa'])
            result['exit_code'] = result['api_tests']['exit_code']
        if group == 'scenarios' and result['exit_code'] == 0:
            expected = json.loads((ROOT / 'checks/expected_changes.json').read_text())['scenarios']
            rows = json.loads((out / 'scenarios/results.json').read_text())
            for row in rows:
                measured = [row['configuration']['added'], row['configuration']['removed'], len(row['conversion_flags_added']), len(row['source_files']), sum((f['added'] for f in row['source_files'])), sum((f['removed'] for f in row['source_files']))]
                if measured != expected[row['id']]:
                    raise RuntimeError(f"Unexpected change scope for {row['id']}: {measured}")
                if set(row['tests']['counts']) != {'passed'}:
                    raise RuntimeError(f"Incomplete test results for {row['id']}")
            result['change_measurements_match'] = True
        summary['groups'][group] = result
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f"  {('PASS' if result['exit_code'] == 0 else 'FAIL')}: {log}", flush=True)
        if result['exit_code']:
            return result['exit_code']
    print(f"Results: {out / 'summary.json'}", flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
