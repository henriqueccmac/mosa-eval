import numpy as np
import pandas as pd
import mudata
from mosa.data.io import csv_to_mudata

def test_feather_omic_table_converts_with_metadata_and_missing_values(tmp_path):
    metadata = pd.DataFrame({'model_id': ['s1', 's2'], 'model_type': ['A', 'B']})
    values = pd.DataFrame({'s1': [1.0, np.nan], 's2': [2.0, 4.0]}, index=['f1', 'f2'])
    metadata_path = tmp_path / 'metadata.csv'
    feather_path = tmp_path / 'view.feather'
    output_path = tmp_path / 'converted.h5mu'
    metadata.to_csv(metadata_path, index=False)
    values.reset_index().to_feather(feather_path)
    csv_to_mudata(str(metadata_path), [('v', str(feather_path))], str(output_path))
    with mudata.set_options(pull_on_update=False):
        result = mudata.read_h5mu(output_path)
    view = result.mod['v']
    assert list(view.var_names) == ['f1', 'f2']
    assert list(result.obs_names) == ['s1', 's2']
    np.testing.assert_allclose(np.asarray(view.X), values.to_numpy().T, equal_nan=True)
    np.testing.assert_array_equal(np.asarray(view.layers['mask']), ~np.isnan(values.to_numpy().T))
