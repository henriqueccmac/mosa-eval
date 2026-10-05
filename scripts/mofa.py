"""Verify MOFA fitting, result reuse, plotting, cross-validation and search."""
from pathlib import Path
import json
import os
import subprocess
import sys
import hashlib
from support import ROOT, WORK, environment
SOURCE = ROOT / 'source'
TABLES = ROOT / 'data'
ENV = environment(SOURCE)
os.environ.update(ENV)
sys.path.insert(0, str(SOURCE / 'src'))
import numpy as np
import pandas as pd
import yaml
import h5py
from mosa.data.io import load_mudata
from mosa.models.registry import load_model
CLI = [sys.executable, '-B', '-c', 'from mosa.cli import main; main()']
RESULTS = {'source_commit': '12f4651b50a4da6937255afe830452a8706d8f23', 'cases': {}}

def save():
    (WORK / 'mofa/results.json').write_text(json.dumps(RESULTS, indent=2))

def command(work, name, args):
    log = work / (name + '.log')
    with log.open('w') as stream:
        stream.write('COMMAND ' + json.dumps(CLI + [str(a) for a in args]) + '\n')
        stream.flush()
        proc = subprocess.run(CLI + [str(a) for a in args], cwd=SOURCE, env=ENV, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
    value = {'exit_code': proc.returncode, 'log': str(log.relative_to(WORK))}
    print(work.name, name, value, flush=True)
    return value
for size in (60, 50):
    work = WORK / 'mofa' / ('samples_' + str(size))
    work.mkdir(parents=True, exist_ok=True)
    case = RESULTS['cases'][str(size)] = {}
    pd.read_csv(TABLES / 'conditionals.csv').iloc[:size].to_csv(work / 'conditionals.csv', index=False)
    for view in ('gexp', 'meth'):
        pd.read_csv(TABLES / (view + '.csv'), index_col=0).iloc[:, :size].to_csv(work / (view + '.csv'))
    store = work / 'data.h5mu'
    case['convert'] = command(work, 'convert', ['convert', '--conditionals', work / 'conditionals.csv', '--view', 'gexp:' + str(work / 'gexp.csv'), '--view', 'meth:' + str(work / 'meth.csv'), '--output', store])
    assert case['convert']['exit_code'] == 0
    cfg = {'data': {'path': str(store), 'views': ['gexp', 'meth']}, 'model': {'type': 'mofa', 'n_factors': 3, 'iterations': 100, 'drop_r2': None, 'random_seed': 42, 'output_dir': str(work / 'outputs')}, 'evaluation': {'test_size': 0, 'n_folds': 2}}
    config = work / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    case['train'] = command(work, 'train', ['train', '--config', config])
    assert case['train']['exit_code'] == 0
    checkpoint = work / 'outputs/mofa_model.hdf5'
    case['transform'] = command(work, 'transform', ['transform', '--checkpoint', checkpoint, '--input', store, '--output', work / 'transformed', '--reconstruct'])
    assert case['transform']['exit_code'] == 0
    data = load_mudata(store, ['gexp', 'meth'])
    model = load_model(checkpoint)
    latent = model.transform(data)
    exported_latent = pd.read_parquet(work / 'outputs/full/latent.parquet')
    restored_latent = pd.read_parquet(work / 'transformed/latent.parquet')
    pd.testing.assert_frame_equal(exported_latent, restored_latent)
    np.testing.assert_allclose(latent, exported_latent.to_numpy())
    assert list(exported_latent.index) == list(data.sample_names)
    assert np.isfinite(latent).all()
    case['latent_roundtrip'] = {'shape': list(latent.shape), 'finite': True, 'sample_order': True, 'equal': True}
    case['reconstructions'] = {}
    with h5py.File(checkpoint) as artifact:
        for view in data.view_names:
            trained = pd.read_parquet(work / 'outputs/full' / ('recon_' + view + '.parquet'))
            restored = pd.read_parquet(work / 'transformed' / ('recon_' + view + '.parquet'))
            pd.testing.assert_frame_equal(trained, restored)
            assert list(restored.columns) == list(data.feature_names[view])
            assert list(restored.index) == list(data.sample_names)
            bad = ~np.isfinite(restored.to_numpy())
            assert not (bad & data.masks[view]).any()
            case['reconstructions'][view] = {'shape': list(restored.shape), 'roundtrip_equal': True, 'names_and_order': True, 'nonfinite': int(bad.sum()), 'nonfinite_observed': int((bad & data.masks[view]).sum()), 'affected_samples': list(restored.index[bad.any(axis=1)]), 'nonfinite_intercepts': {group: int((~np.isfinite(artifact['intercepts'][view][group][:])).sum()) for group in artifact['intercepts'][view]}}
    case['plot'] = command(work, 'plot', ['plot', '--config', config])
    plots = sorted((str(p.relative_to(work)) for p in (work / 'outputs/plots').rglob('*.png')))
    case['plot']['files'] = plots
    case['plot']['expected_present'] = all((any((p.endswith('input_recon_' + level + '_' + view + '_recon.png') for p in plots)) for view in data.view_names for level in ('sample', 'feature'))) and any((p.endswith('umap_z.png') for p in plots))
    if size == 50:
        assert all((v['nonfinite'] == 0 for v in case['reconstructions'].values()))
        assert case['plot']['exit_code'] == 0 and case['plot']['expected_present']
    else:
        assert case['reconstructions']['meth']['nonfinite'] == 300
        assert case['plot']['exit_code'] != 0
    cfg['model']['output_dir'] = str(work / 'cv_outputs')
    cv_config = work / 'cv.yaml'
    cv_config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    case['cross_validation'] = command(work, 'cross_validation', ['cross-validate', '--config', cv_config])
    if size == 50:
        assert case['cross_validation']['exit_code'] == 0
        case['cross_validation']['artifacts'] = {}
        for view in data.view_names:
            recon = pd.read_parquet(work / 'cv_outputs' / ('recon_' + view + '.parquet'))
            assert recon.shape == data.views[view].shape
            assert list(recon.index) == list(data.sample_names)
            assert list(recon.columns) == list(data.feature_names[view])
            assert np.isfinite(recon.to_numpy()).all()
            case['cross_validation']['artifacts'][view] = {'shape': list(recon.shape), 'finite': True, 'names_and_order': True}
        assert not (work / 'cv_outputs/history.csv').exists()
        case['cross_validation']['history_written'] = False
        search = work / 'search.yaml'
        search.write_text(yaml.safe_dump({'n_factors': {'dist': 'categorical', 'choices': [2, 3]}}))
        cfg['model']['output_dir'] = str(work / 'search_outputs')
        search_config = work / 'search_config.yaml'
        search_config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        case['search'] = command(work, 'search', ['optimize', '--config', search_config, '--search-space', search, '--trials', '2'])
        assert case['search']['exit_code'] == 0
        assert 'completed: 2, pruned: 0' in (work / 'search.log').read_text()
        case['search']['persistent_outputs'] = sorted((str(p.relative_to(work)) for p in (work / 'search_outputs').rglob('*') if p.is_file()))
        assert not case['search']['persistent_outputs']
    else:
        assert case['cross_validation']['exit_code'] != 0
    save()
RESULTS['input_hashes'] = {name: hashlib.sha256((TABLES / name).read_bytes()).hexdigest() for name in ('conditionals.csv', 'gexp.csv', 'meth.csv')}
save()
print('MOFA workflow checks passed.', flush=True)
