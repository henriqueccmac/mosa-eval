# MOSA evaluation

Supplementary material for thesis Section 4.5: the MOSA source snapshot, synthetic inputs, configurations, source changes, and checks underlying the reported software evaluation.

## Reproduce

Requires Python 3.13 and Git. The synthetic experiments run on CPU.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

The command runs the data adaptations and source extensions, configuration alternatives, input-format checks, and MOFA workflows. It writes a summary to `build/summary.json`, with detailed results and logs under `build/`. A nonzero exit status indicates a failed check. Use a fresh `--output` directory to repeat a run.

For one extension: `python reproduce.py --scenario A4 --output build-a4`.

## Inspect

- `source/`: unmodified MOSA snapshot; revision and file hashes in `source.json`.
- `data/`: synthetic inputs; regenerate with `python scripts/generate_data.py`.
- `configs/`: evaluated experiment configurations.
- `patches/`: source changes for A1–A7; A3 changes configuration only.
- `scripts/`, `checks/`: execution and verification; each extension uses its own source copy.
- `results/`: reference results, including the changes measured for each scenario.

Two capability-guard tests explicitly disable projection through `checks/capability-tests.patch`; the MOSA production snapshot is unchanged.

The 60-sample MOFA case is expected to produce 300 non-finite reconstruction values and reject plotting and cross-validation. The 50-sample control completes those operations. These outcomes are checked explicitly; they are not reproduction failures.

Real-data MOSA training is documented separately in `results/real_data/` and is not repeated by the synthetic evaluation command. These experiments assess software behaviour and change scope, not comparative predictive performance.
