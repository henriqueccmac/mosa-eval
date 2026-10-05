import numpy as np
from mosa.config import DataConfig
from mosa.models.mosa.config import MOSAConfig
from mosa.models.mosa.datamodule import MOSADataModule, _fit_robust_scaling

def test_robust_scaling_fits_observed_training_values_and_restores_state():
    values = np.array([[1.0, 99.0], [3.0, np.nan], [5.0, 1000.0], [9.0, np.nan]], dtype=np.float32)
    observed = np.array([[1, 1], [1, 0], [1, 0], [1, 0]], dtype=bool)
    stats = _fit_robust_scaling(values, observed)
    assert np.allclose(stats['mean'], [4.0, 99.0])
    assert np.allclose(stats['scale'], [3.5, 1.0])
    config = MOSAConfig(preprocessing_mode='robust')
    data_config = DataConfig(views=['a'])
    module = MOSADataModule(None, None, data_config, config)
    module.scalers['a'] = stats
    transformed = module._apply_fitted_preprocessing('a', values, observed, np.zeros(4, dtype=int))
    assert np.allclose(transformed[:, 0], [-3 / 3.5, -1 / 3.5, 1 / 3.5, 5 / 3.5])
    assert np.allclose(transformed[:, 1], 0.0)
    assert np.allclose(module.inverse_transform_view('a', transformed)[:, 0], values[:, 0])
    restored = MOSADataModule(None, None, data_config, config)
    restored.load_state_dict(module.state_dict())
    assert np.allclose(restored._apply_fitted_preprocessing('a', values, observed, np.zeros(4, dtype=int)), transformed)
