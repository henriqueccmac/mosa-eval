"""Verify the 26 experiment configurations reported in Section 4.5.2."""
import copy, json, os, sys, traceback
from support import ROOT, WORK, command, environment
source = ROOT / 'source'
os.environ.update(environment(source))
sys.path.insert(0, str(source / 'src'))
import numpy as np
import pandas as pd
import torch
import yaml
from mosa.data.io import csv_to_mudata
from mosa.utils import load_config, validate_config_against_data
from mosa.models.registry import load_model
out = WORK / 'alternatives'
out.mkdir(parents=True, exist_ok=True)
tables = ROOT / 'data'
metadata = pd.read_csv(tables / 'conditionals.csv')
mut = pd.DataFrame([np.arange(len(metadata)) % 2], index=['TP53'], columns=metadata.model_id)
mut.to_csv(out / 'mutations.csv')
views = ['gexp', 'meth', 'crispr']
for view in views:
    df = pd.read_csv(tables / f'{view}.csv', index_col=0)
    df.where(df.isna(), (df > 0).astype(float)).to_csv(out / f'{view}_binary.csv')
stores = {}
for kind, discrete in [('continuous', []), ('mixed', ['crispr']), ('discrete', views)]:
    store = out / f'{kind}.h5mu'
    csv_to_mudata(str(tables / 'conditionals.csv'), [(v, str(out / f'{v}_binary.csv' if v in discrete else tables / f'{v}.csv')) for v in views], str(store), mutations_path=str(out / 'mutations.csv'))
    stores[kind] = str(store)
base = yaml.safe_load((ROOT / 'configs/baseline3.yaml').read_text())
base['data'].update(path=stores['continuous'], use_tissue=True, use_mutations=True)
cases = []

def add(name, data=None, model=None, per_view=None):
    cfg = copy.deepcopy(base)
    cfg['data'].update(data or {})
    cfg['model'].update(model or {})
    if per_view:
        for v in cfg['model']['views'].values():
            v.update(per_view)
    cases.append((name, cfg))
for kind, discrete in [('continuous', []), ('mixed', ['crispr']), ('discrete', views)]:
    add('view_type_' + kind, data={'path': stores[kind], 'discrete_views': discrete})
for field in ['use_tissue', 'use_mutations']:
    for state in [False, True]:
        add(f'{field}_{state}', data={field: state})
for field in ['adv_weight', 'contrastive_weight']:
    for value in [0.0, 0.01]:
        add(f'{field}_{value}', model={field: value})
for state in [False, True]:
    add(f'kl_warmup_{state}', model={'use_kl_scheduler': state, 'kl_warmup_epochs': 1 if state else 0, 'kl_weight_final': 0.001})
for value in ['mean', 'macro']:
    add('reduction_' + value, per_view={'loss_type': value})
for value in ['concat', 'poe']:
    add('fusion_' + value, model={'fusion_method': value})
for value in [0.0, 0.2]:
    add(f'feature_dropout_{value}', per_view={'dropout_p': value})
for value in [0.0, 0.1]:
    add(f'view_dropout_{value}', model={'view_dropout_prob': value})
add('both_auxiliary_losses', model={'adv_weight': 0.01, 'contrastive_weight': 0.01})
for value in ['standardize', 'center', 'none']:
    add('preprocessing_' + value, model={'preprocessing_mode': value})
mofa_meta = out / 'mofa_metadata.csv'
metadata.iloc[:50].to_csv(mofa_meta, index=False)
for v in views[:2]:
    pd.read_csv(tables / f'{v}.csv', index_col=0).iloc[:, :50].to_csv(out / f'mofa_{v}.csv')
mofa_store = out / 'mofa.h5mu'
csv_to_mudata(str(mofa_meta), [(v, str(out / f'mofa_{v}.csv')) for v in views[:2]], str(mofa_store))
mofa = copy.deepcopy(base)
mofa['data'].update(path=str(mofa_store), views=views[:2], use_tissue=False, use_mutations=False)
mofa['model'] = {'type': 'mofa', 'n_factors': 3, 'iterations': 100, 'drop_r2': None, 'random_seed': 42}
mofa['evaluation'] = {'test_size': 0}
cases.append(('integration_mofa', mofa))
selected = sys.argv[1] if len(sys.argv) > 1 else None
results = []
if selected:
    results = [r for r in json.loads((out / 'results.json').read_text())['cases'] if r['case'] != selected]
for name, cfg in cases:
    if selected and name != selected:
        continue
    folder = out / name
    folder.mkdir(exist_ok=True)
    cfg['model']['output_dir'] = str(folder / 'outputs')
    path = folder / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    row = {'case': name, 'passed': False}
    try:
        parsed = load_config(str(path))
        warnings = validate_config_against_data(parsed)
        assert not warnings, warnings
        row['validation_warnings'] = warnings
        result = command([sys.executable, '-B', '-c', 'from mosa.cli import main; main()', 'train', '--config', str(path)], source, folder / 'training.log')
        row['training_exit'] = result['exit_code']
        assert result['exit_code'] == 0, result
        outputs = folder / 'outputs'
        if name != 'integration_mofa':
            checkpoint = next(outputs.rglob('*.ckpt'))
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            hp = saved['hyper_parameters']
            for block, saved_block in [('data', 'data_cfg'), ('model', 'model_cfg')]:
                for key, value in cfg[block].items():
                    if key == 'type':
                        continue
                    actual = hp[saved_block][key]
                    if key == 'discrete_views':
                        assert set(actual) == set(value)
                    elif key == 'views' and block == 'model':
                        for v, settings in value.items():
                            for k, val in settings.items():
                                assert actual[v][k] == val
                    else:
                        assert actual == value, (block, key, actual, value)
            expected_cond = metadata.model_type.nunique() + (metadata.tissue.nunique() if cfg['data']['use_tissue'] else 0) + int(cfg['data']['use_mutations'])
            assert hp['conditional_dim'] == expected_cond
            disc = any((k.startswith('discriminator.') for k in saved['state_dict']))
            assert disc == (cfg['model'].get('adv_weight', 0) > 0)
            row.update(conditional_dim=expected_cond, discriminator_present=disc, saved_settings_verified=True)
        else:
            loaded = load_model(outputs / 'mofa_model.hdf5')
            assert type(loaded).__name__ == 'MOFAModel'
            assert loaded.data_cfg.views == views[:2]
            row['loaded_model'] = type(loaded).__name__
        expected_ids = set(metadata.model_id[:50] if name == 'integration_mofa' else metadata.model_id)
        full = pd.read_parquet(outputs / 'full/latent.parquet')
        assert full.index.is_unique and set(full.index) == expected_ids
        for split in ['full'] if name == 'integration_mofa' else ['train', 'full', 'val']:
            latent = pd.read_parquet(outputs / split / 'latent.parquet')
            assert latent.index.is_unique and np.isfinite(latent.to_numpy()).all()
            for v in cfg['data']['views']:
                recon = pd.read_parquet(outputs / split / f'recon_{v}.parquet')
                original = pd.read_csv(tables / f'{v}.csv', index_col=0)
                assert list(recon.columns) == list(original.index)
                assert list(recon.index) == list(latent.index)
                assert np.isfinite(recon.to_numpy()).all()
        if name != 'integration_mofa':
            train = set(pd.read_parquet(outputs / 'train/latent.parquet').index)
            val = set(pd.read_parquet(outputs / 'val/latent.parquet').index)
            assert not train & val and train | val == expected_ids
        row.update(passed=True, outputs_verified=True)
    except Exception:
        row['error'] = traceback.format_exc()
    results.append(row)
    (out / 'results.json').write_text(json.dumps({'source_commit': '12f4651b50a4da6937255afe830452a8706d8f23', 'cases': results}, indent=2) + '\n')
    print(name, row['passed'], row.get('error', ''), flush=True)
assert all((r['passed'] for r in results)), 'Failed checks retained in results.json'
