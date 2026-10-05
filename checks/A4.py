import pytest
import torch
from mosa.models.mosa.config import MOSAConfig, OmicViewConfig
from mosa.models.mosa.vae.encoder import OmicEncoder, ResidualMLPEncoder
from mosa.models.mosa.vae.vae_module import VAE

def test_residual_encoder_selected_by_config_and_trains():
    views = {'a': OmicViewConfig(name='a', hidden_layer_dims=[6])}
    config = MOSAConfig(views=views, joint_latent_dim=3, encoder_architecture='residual_mlp', view_dropout_prob=0)
    model = VAE(config=config, view_input_dims={'a': 4}, conditional_dim=0, n_batches=2)
    encoder = model.encoders['a'].eval()
    assert isinstance(encoder, ResidualMLPEncoder)
    values = torch.randn(5, 4)
    assert torch.allclose(encoder(values), encoder.net(values) + encoder.skip(values))
    encoder(values).sum().backward()
    assert encoder.skip.weight.grad is not None
    default = VAE(config=MOSAConfig(views=views), view_input_dims={'a': 4}, conditional_dim=0, n_batches=2)
    assert type(default.encoders['a']) is OmicEncoder
    with pytest.raises(ValueError, match='encoder_architecture'):
        MOSAConfig(encoder_architecture='missing')
