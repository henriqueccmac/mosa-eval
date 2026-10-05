"""Verify 48 input/storage combinations, two mixed-input cases and four rejections."""
from pathlib import Path
import os, sys, json, traceback, contextlib, io, zipfile
from support import ROOT, WORK, environment, command
source = ROOT / 'source'
os.environ.update(environment(source))
sys.path.insert(0, str(source / 'src'))
import numpy as np
import pandas as pd
from mosa.data.io import csv_to_mudata, load_mudata, _read_table
from mosa.errors import DataError
out = WORK / 'formats'
out.mkdir(exist_ok=True)
metadata = pd.DataFrame({'model_id': ['S3', 'S1', 'S4', 'S2'], 'model_type': ['Tumor', 'Cell_Line', 'Tumor', 'Cell_Line'], 'tissue': ['breast', 'lung', 'lung', 'breast']})
a = pd.DataFrame([[3.0, 1.0, 4.0, 2.0], [30.0, np.nan, 40.0, 20.0]], index=pd.Index(['gene_a', 'gene_b'], name='feature'), columns=['P3', 'P1', 'P4', 'P2'])
b = pd.DataFrame([[200.0, 100.0, 300.0], [2.0, 1.0, np.nan]], index=pd.Index(['site_a', 'site_b'], name='feature'), columns=['P2', 'P1', 'P3'])
mut = pd.DataFrame([[0, 1, 0, 1]], index=pd.Index(['TP53'], name='feature'), columns=['S1', 'S2', 'S3', 'S4'])
ids = pd.DataFrame({'source_id': ['P1', 'P2', 'P3', 'P4'], 'model_id': ['S1', 'S2', 'S3', 'S4']})
encodings = [(ext, comp, 'native') for ext in ['csv', 'tsv', 'txt', 'tab'] for comp in ['', '.gz', '.bz2', '.xz', '.zip']] + [(ext, '', layout) for ext in ['parquet', 'pq'] for layout in ['native', 'column']]
records = []

def persist():
    (out / 'results.json').write_text(json.dumps({'source_commit': '12f4651b50a4da6937255afe830452a8706d8f23', 'cases': records}, indent=2) + '\n')

def write(df, p, ext, index, layout):
    if ext in ['parquet', 'pq']:
        if index and layout == 'column':
            df.reset_index().to_parquet(p, index=False)
        else:
            df.to_parquet(p, index=index)
    else:
        df.to_csv(p, sep=',' if ext == 'csv' else '\t', index=index)

def verify(ds):
    assert set(ds.sample_names) == set(metadata.model_id)
    assert ds.view_names == ['expression', 'methylation']
    order = ds.sample_names
    for view, df in [('expression', a), ('methylation', b)]:
        expected = df.rename(columns=dict(zip(ids.source_id, ids.model_id))).T.reindex(order).astype('float32')
        assert ds.feature_names[view] == list(expected.columns)
        np.testing.assert_allclose(ds.views[view], expected.to_numpy(), rtol=0, atol=0, equal_nan=True)
        np.testing.assert_array_equal(ds.masks[view], expected.notna().to_numpy())
    for col in ['model_type', 'tissue']:
        assert ds.metadata[col].astype(str).tolist() == metadata.set_index('model_id').loc[order, col].tolist()
    assert ds.metadata['mutation_TP53'].tolist() == mut.loc['TP53', order].tolist()
    ds.validate()
for ext, comp, layout in encodings:
    tag = ext + comp + '_' + layout
    folder = out / tag
    folder.mkdir(exist_ok=True)
    paths = {}
    for name, df, index in [('metadata', metadata, False), ('expression', a, True), ('methylation', b, True), ('mutations', mut, True), ('id_map', ids, False)]:
        p = folder / (name + '.' + ext + comp)
        write(df, p, ext, index, layout)
        paths[name] = str(p)
    for storage in ['h5mu', 'zarr']:
        row = {'encoding': tag, 'storage': storage}
        try:
            store = folder / ('dataset.' + storage)
            with (folder / (storage + '.log')).open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                csv_to_mudata(paths['metadata'], [(v, paths[v]) for v in ['expression', 'methylation']], str(store), mutations_path=paths['mutations'], id_map_path=paths['id_map'], format=storage)
                ds = load_mudata(str(store), ['expression', 'methylation'])
                verify(ds)
                subset = load_mudata(str(store), ['methylation'])
                np.testing.assert_allclose(subset.views['methylation'], ds.views['methylation'], equal_nan=True)
                assert subset.sample_names == ds.sample_names
            row.update(status='passed', samples=ds.n_samples, checks=['values', 'missing_entries', 'absent_view', 'sample_identifiers', 'feature_names', 'metadata', 'mutations', 'identifier_mapping', 'selected_view_loading'])
        except Exception as e:
            row.update(status='failed', error=str(e), traceback=traceback.format_exc())
        records.append(row)
        persist()
        print(tag, storage, row['status'], row.get('error', ''), flush=True)
CLI = [sys.executable, '-B', '-c', 'from mosa.cli import main; main()']
for storage in ['h5mu', 'zarr']:
    store = out / ('cli_mixed.' + storage)
    args = ['convert', '--conditionals', str(out / 'csv.gz_native/metadata.csv.gz'), '--view', 'expression:' + str(out / 'parquet_column/expression.parquet'), '--view', 'methylation:' + str(out / 'tab.xz_native/methylation.tab.xz'), '--mutations', str(out / 'pq_native/mutations.pq'), '--id-map', str(out / 'tsv.zip_native/id_map.tsv.zip'), '--format', storage, '--output', str(store)]
    result = command(CLI + args, source, out / ('cli_' + storage + '.log'))
    row = {'encoding': 'mixed_cli', 'storage': storage, 'command_result': result}
    try:
        assert result['exit_code'] == 0
        verify(load_mudata(str(store), ['expression', 'methylation']))
        row['status'] = 'passed'
    except Exception as e:
        row.update(status='failed', error=str(e))
    records.append(row)
    persist()
    print('CLI', storage, row['status'], flush=True)
for ext in ['json', 'xlsx', 'feather']:
    p = out / ('unsupported.' + ext)
    p.write_text('not a supported table')
    try:
        _read_table(p)
        row = {'encoding': ext, 'status': 'failed', 'error': 'unsupported format accepted'}
    except DataError:
        row = {'encoding': ext, 'status': 'passed', 'check': 'unsupported format rejected'}
    records.append(row)
p = out / 'multiple.csv.zip'
with zipfile.ZipFile(p, 'w') as z:
    z.writestr('one.csv', 'x,y\na,1\n')
    z.writestr('two.csv', 'x,y\nb,2\n')
try:
    _read_table(p)
    row = {'encoding': 'multiple.csv.zip', 'status': 'failed', 'error': 'ambiguous ZIP accepted'}
except ValueError:
    row = {'encoding': 'multiple.csv.zip', 'status': 'passed', 'check': 'multiple-member ZIP rejected'}
records.append(row)
persist()
print('SUMMARY', len(records), 'checks;', sum((x['status'] == 'passed' for x in records)), 'passed', flush=True)
sys.exit(any((x['status'] != 'passed' for x in records)))
