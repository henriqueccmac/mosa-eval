"""Run each data adaptation or source extension on an isolated MOSA copy."""
from pathlib import Path
import argparse
import copy
import difflib
import json
import os
import shutil
import sys
import traceback
import yaml
from support import ROOT, WORK, command, pytest_run, environment
from changes import apply
CLASS_MAP = json.loads((ROOT / 'checks/responsibilities.json').read_text())
CLI = [sys.executable, '-B', '-c', 'from mosa.cli import main; main()']
DIMS = {'gexp': 40, 'meth': 30, 'crispr': 20}
SOURCE_IDS = {'A1', 'A2', 'A4', 'A5', 'A6', 'A7'}

def config(views, store, out, overrides=None):
    return {'data': {'path': str(store), 'views': views}, 'model': {'type': 'mosa_vae', 'output_dir': str(out), 'joint_latent_dim': 8, 'num_epochs': 1, 'batch_size': 16, 'accelerator': 'cpu', 'devices': 1, 'random_seed': 42, **(overrides or {}), 'views': {v: {'hidden_layer_dims': [16]} for v in views}}, 'evaluation': {'test_size': 0.2, 'n_folds': 3}}

def canonical(cfg):
    cfg = copy.deepcopy(cfg)
    cfg['data'].pop('path')
    cfg['model'].pop('output_dir')
    return yaml.safe_dump(cfg, sort_keys=False).splitlines(keepends=True)

def diff_lines(before, after):
    diff = list(difflib.unified_diff(before, after, fromfile='a', tofile='b'))
    return {'added': sum((x.startswith('+') and (not x.startswith('+++')) for x in diff)), 'removed': sum((x.startswith('-') and (not x.startswith('---')) for x in diff)), 'diff': ''.join(diff)}

def source_diff(pristine, edited):
    files = []
    patches = []
    allpaths = sorted({p.relative_to(pristine) for p in (pristine / 'src').rglob('*') if p.is_file()} | {p.relative_to(edited) for p in (edited / 'src').rglob('*') if p.is_file()})
    for rel in allpaths:
        a = (pristine / rel).read_text().splitlines(keepends=True) if (pristine / rel).exists() else []
        b = (edited / rel).read_text().splitlines(keepends=True) if (edited / rel).exists() else []
        if a == b:
            continue
        d = diff_lines(a, b)
        files.append({'file': str(rel), 'added': d['added'], 'removed': d['removed']})
        patches.extend(difflib.unified_diff(a, b, fromfile='a/' + str(rel), tofile='b/' + str(rel)))
    return (files, ''.join(patches))

def verify_store(store, views, ids, types):
    import mudata
    import numpy as np
    with mudata.set_options(pull_on_update=False):
        data = mudata.read_h5mu(store)
    assert set(data.mod) == set(views), (list(data.mod), views)
    assert set(data.obs_names) == set(ids) and data.n_obs == len(ids)
    assert set(data.obs['model_type'].astype(str)) == set(types)
    observed = {}
    for v in views:
        assert data.mod[v].n_vars == DIMS[v]
        observed[v] = int(np.asarray(data.mod[v].layers['mask']).any(axis=1).sum())
    expected_observed = {'gexp': len(ids), 'meth': 58 if len(ids) == 68 else 50, 'crispr': 30}
    assert observed == {v: expected_observed[v] for v in views}, (observed, expected_observed)
    return {'samples': data.n_obs, 'views': list(data.mod), 'features': {v: data.mod[v].n_vars for v in views}, 'observed_samples': observed, 'types': sorted(types)}

def verify_outputs(out, views, ids):
    import pandas as pd
    import numpy as np
    sets = {}
    checks = {}
    for split in ['train', 'val', 'full']:
        latent = pd.read_parquet(out / split / 'latent.parquet')
        assert latent.index.is_unique
        assert np.isfinite(latent.to_numpy()).all()
        sets[split] = set(latent.index.astype(str))
        actual = {p.stem.removeprefix('recon_') for p in (out / split).glob('recon_*.parquet')}
        assert actual == set(views), (split, actual, views)
        for v in views:
            recon = pd.read_parquet(out / split / f'recon_{v}.parquet')
            assert recon.shape == (len(latent), DIMS[v])
            assert set(recon.index.astype(str)) == sets[split]
            assert list(recon.columns) == [f'{v}_f{i}' for i in range(DIMS[v])]
            assert np.isfinite(recon.to_numpy()).all()
        checks[split] = {'samples': len(latent), 'views': sorted(actual)}
    assert not sets['train'] & sets['val']
    assert sets['train'] | sets['val'] == set(ids)
    assert sets['full'] == set(ids)
    return checks

def scenario(sid, source, runroot):
    work = runroot / sid
    work.mkdir()
    export = work / 'source'
    shutil.copytree(source, export, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    import subprocess
    subprocess.run(['git', 'apply', str(ROOT / 'checks/capability-tests.patch')], cwd=export, check=True)
    apply(export, sid)
    if sid in SOURCE_IDS:
        shutil.copy2(ROOT / 'checks' / f'{sid}.py', export / 'tests/test_extension.py')
    files, patch = source_diff(source, export)
    for item in files:
        item['class'] = CLASS_MAP.get(item['file'], 'unclassified')
    test_files = []
    for rel in sorted(set((p.relative_to(source) for p in (source / 'tests').rglob('*.py'))) | set((p.relative_to(export) for p in (export / 'tests').rglob('*.py')))):
        a = (source / rel).read_text().splitlines(keepends=True) if (source / rel).exists() else []
        b = (export / rel).read_text().splitlines(keepends=True) if (export / rel).exists() else []
        if a != b:
            d = diff_lines(a, b)
            test_files.append({'file': str(rel), 'added': d['added'], 'removed': d['removed']})
    (work / 'source.diff').write_text(patch)
    views = ['gexp', 'meth'] if sid in ['baseline2', 'D3', 'D4', 'D5'] else ['gexp', 'meth', 'crispr']
    if sid == 'D2':
        views = ['gexp', 'crispr']
    storeviews = ['gexp', 'meth', 'crispr'] if sid == 'D2' else views
    data = ROOT / 'data'
    metadata = 'conditionals.csv'
    suffix = ''
    ids = [f'S{i:03}' for i in range(60)]
    types = ['Cell_Line', 'Organoid', 'Tumor']
    if sid in ['D3', 'D4']:
        suffix = '_pdx'
        metadata = 'conditionals_pdx.csv'
        ids += [f'X{i:03}' for i in range(8)]
        types += ['PDX']
    if sid == 'D5':
        metadata = 'conditionals_p2.csv'
        ids += [f'T{i:03}' for i in range(25)]
    store = work / 'store.h5mu'
    out = work / 'outputs'
    conversion = ['convert', '--conditionals', str(data / metadata)]
    for v in storeviews:
        conversion += ['--view', v + ':' + str(data / (v + suffix + '.csv'))]
    if sid == 'D5':
        conversion += ['--view', 'gexp:' + str(data / 'gexp_p2.csv'), '--id-map', str(data / 'id_map.csv')]
    if sid == 'A7':
        import pandas as pd
        feather_dir = work / 'feather_inputs'
        feather_dir.mkdir()
        new = []
        for v in storeviews:
            path = feather_dir / (v + '.feather')
            pd.read_csv(data / (v + '.csv'), index_col=0).reset_index().to_feather(path)
            new += ['--view', v + ':' + str(path)]
        conversion = ['convert', '--conditionals', str(data / metadata)] + new
    conversion += ['--output', str(store)]
    flags = {'D1': ['--view crispr:<table>'], 'D5': ['--view gexp:<provider2>', '--id-map <crosswalk>']}.get(sid, [])
    overrides = {'A1': {'fusion_method': 'mean'}, 'A2': {'latent_l2_weight': 0.01}, 'A3': {'use_kl_scheduler': True, 'kl_warmup_epochs': 1, 'kl_weight_final': 0.001}, 'A4': {'encoder_architecture': 'residual_mlp'}, 'A5': {'lr_scheduler': 'exponential'}, 'A6': {'preprocessing_mode': 'robust'}, 'D4': {'preprocessing_mode': 'center'}}.get(sid, {})
    cfg = config(views, store, out, overrides)
    baselineviews = ['gexp', 'meth'] if sid in ['D1', 'D3', 'D4', 'D5', 'baseline2'] else ['gexp', 'meth', 'crispr']
    baseline = config(baselineviews, store, out)
    cfgpath = work / 'config.yaml'
    cfgpath.write_text(yaml.safe_dump(cfg, sort_keys=False))
    cfgdiff = diff_lines(canonical(baseline), canonical(cfg))
    (work / 'config.diff').write_text(cfgdiff['diff'])
    result = {'id': sid, 'source_files': files, 'test_files': test_files, 'configuration': cfgdiff, 'conversion_flags_added': flags, 'intended': {'samples': len(ids), 'views': views, 'store_views': storeviews}, 'tests': None}

    def save():
        (work / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    save()
    try:
        result['conversion'] = command(CLI + conversion, export, work / 'conversion.log')
        assert result['conversion']['exit_code'] == 0, 'Conversion failed'
        result['store_verified'] = verify_store(store, storeviews, ids, types)
        if sid in SOURCE_IDS:
            result['tests'] = pytest_run(export, runroot.name + '_' + sid, ['tests'])
        else:
            result['tests'] = pytest_run(export, runroot.name + '_' + sid, ['tests'])
        result['training'] = command(CLI + ['train', '--config', str(cfgpath)], export, work / 'training.log')
        assert result['training']['exit_code'] == 0, 'Training failed'
        result['outputs_verified'] = verify_outputs(out, views, ids)
        result['status'] = 'passed' if result['tests']['exit_code'] == 0 else 'tests_failed'
    except Exception as error:
        result['status'] = 'failed'
        result['error'] = str(error)
        result['traceback'] = traceback.format_exc()
    save()
    print(sid, result['status'], result.get('error', ''), flush=True)
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('--only', nargs='+')
    args = p.parse_args()
    source = args.source.resolve()
    assert source.is_relative_to(ROOT) and (source / 'src/mosa').is_dir()
    os.environ.update(environment(source))
    runroot = WORK / 'scenarios'
    runroot.mkdir(parents=True, exist_ok=True)
    ids = args.only or ['baseline2', 'baseline3', 'D1', 'D2', 'D3', 'D4', 'D5', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']
    results = []
    for sid in ids:
        results.append(scenario(sid, source, runroot))
        (runroot / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    return int(any((r['status'] != 'passed' for r in results)))
if __name__ == '__main__':
    sys.exit(main())
