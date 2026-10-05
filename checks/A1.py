import torch
from mosa.models.mosa.vae.latent import BaseLatentSpace

def test_mean_fusion_ignores_missing_view_and_backpropagates():
    model = BaseLatentSpace.create('mean', {'a': 3, 'b': 5}, 2).eval()
    a, b = (torch.randn(4, 3), torch.randn(4, 5))
    masks = {'a': torch.ones(4, dtype=torch.bool), 'b': torch.zeros(4, dtype=torch.bool)}
    fused = model({'a': a, 'b': b}, ['a', 'b'], masks)[0]
    assert torch.allclose(fused, model.fc_mu(model.projections['a'](a)))
    assert torch.allclose(fused, model({'a': a, 'b': b * 100}, ['a', 'b'], masks)[0])
    fused.sum().backward()
    assert model.projections['a'].weight.grad is not None
