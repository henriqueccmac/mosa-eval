# Reference results

All synthetic checks use the MOSA revision identified in `source.json` and the dependency versions in `constraints.txt`. `summary.json` gives the outcomes; each group contains detailed assertions and execution logs. Paths in reference records are relative to the package or represented as `<environment>`.

- `scenarios/`: two controls and twelve change scenarios. Each record includes the configuration and source diffs, measured change scope, data/output assertions, and full test report. Baselines and configuration/data adaptations pass 416 test cases; source extensions pass 417, including their focused test.
- `alternatives/`: all 26 evaluated configurations pass validation, training and output checks.
- `formats/`: all 48 conversion/loading cases, two mixed-format command executions and four rejection checks pass.
- `mofa/`: fitted-sample outputs, checkpoint round trips, plotting and excluded-sample evaluation for the 60- and 50-sample inputs; 47 API cases pass. Expected failures are identified in the summary.
- `real_data/`: retained MOSA training metrics, configuration and checkpoint identity. This evidence is separate from the reproducible synthetic experiments.

Checks of unsupported model operations use the explicit capability fixture in `checks/capability-tests.patch`; it changes tests only. Production-source measurements exclude this fixture and the focused extension tests.

The scripts compare identifiers, dimensions, masks, finite values and modification scope. They do not require identical floating-point predictions or compare scientific effectiveness between methods.
