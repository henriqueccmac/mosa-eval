# Real-data MOSA run

Retained evidence for Section 4.5.3: experiment configuration, saved hyperparameters, ten epochs of training metrics, and checkpoint identity. Paths in YAML files are relative; numerical settings are unchanged.

The 14,132-sample input, 206 MB checkpoint, and exported reconstruction arrays are not distributed here. The checkpoint hash identifies the inspected artifact; it is not evidence of successful reloading. The synthetic reproduction command does not claim to repeat this run. With access to the original `data_newtest.h5mu` input, set `data.path` in `config.yaml` and run `mosa train --config results/real_data/config.yaml`.
