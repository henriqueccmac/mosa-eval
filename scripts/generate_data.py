"""Generate the seeded synthetic inputs used in the thesis evaluation."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
SEED = 20260924

def generate(out: Path, seed: int=SEED):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    samples = [f'S{i:03}' for i in range(60)]
    tables = {}
    for name, features, ids in [('gexp', 40, samples), ('meth', 30, samples[:50]), ('crispr', 20, samples[:30])]:
        tables[name] = pd.DataFrame(rng.standard_normal((features, len(ids))), index=[f'{name}_f{i}' for i in range(features)], columns=ids)
    extra = [f'X{i:03}' for i in range(8)]
    for name in ['gexp', 'meth']:
        base = tables[name]
        tables[name + '_pdx'] = pd.concat([base, pd.DataFrame(rng.standard_normal((len(base), 8)), index=base.index, columns=extra)], axis=1)
    provider = [f'P2-{i:03}' for i in range(25)]
    tables['gexp_p2'] = pd.DataFrame(rng.standard_normal((40, 25)), index=tables['gexp'].index, columns=provider)
    for name, table in tables.items():
        table.to_csv(out / (name + '.csv'), float_format='%.17g')
    meta = pd.DataFrame({'model_id': samples, 'model_type': ['Cell_Line'] * 30 + ['Organoid'] * 20 + ['Tumor'] * 10, 'tissue': ['Lung'] * 20 + ['Skin'] * 20 + ['Colon'] * 20})
    meta.to_csv(out / 'conditionals.csv', index=False)
    pd.concat([meta, pd.DataFrame({'model_id': extra, 'model_type': 'PDX', 'tissue': 'Lung'})]).to_csv(out / 'conditionals_pdx.csv', index=False)
    canonical = [f'T{i:03}' for i in range(25)]
    pd.concat([meta, pd.DataFrame({'model_id': canonical, 'model_type': 'Tumor', 'tissue': 'Breast'})]).to_csv(out / 'conditionals_p2.csv', index=False)
    pd.DataFrame({'source_id': provider, 'model_id': canonical}).to_csv(out / 'id_map.csv', index=False)
    manifest = {'seed': seed, 'bit_generator': 'PCG64', 'distribution': 'independent standard normal N(0,1)', 'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.csv'))}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    p.add_argument('--seed', type=int, default=SEED)
    args = p.parse_args()
    generate(args.output, args.seed)
