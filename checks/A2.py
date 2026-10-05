import pytest
import torch
from mosa.models.mosa.config import MOSAConfig
from mosa.models.mosa.vae.losses import latent_l2_loss

def test_latent_l2_loss_gradient_and_config_validation():
    values = torch.tensor([[1.0, -2.0]], requires_grad=True)
    loss = latent_l2_loss(values)
    assert loss.item() == 2.5
    loss.backward()
    assert torch.equal(values.grad, torch.tensor([[1.0, -2.0]]))
    with pytest.raises(ValueError, match='nonnegative'):
        MOSAConfig(latent_l2_weight=-1)
