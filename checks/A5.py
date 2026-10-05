from mosa.models.mosa.config import MOSAConfig, OmicViewConfig
from mosa.models.mosa.vae.vae_module import VAE

def test_exponential_scheduler_updates_learning_rate():
    config = MOSAConfig(views={'a': OmicViewConfig('a', hidden_layer_dims=[4])}, joint_latent_dim=2, lr_scheduler='exponential', lr_gamma=0.5)
    model = VAE(config=config, view_input_dims={'a': 3}, conditional_dim=0, n_batches=2)
    optimizers, schedulers = model.configure_optimizers()
    assert len(schedulers) == 1
    optimizer = optimizers[0]
    original = optimizer.param_groups[0]['lr']
    optimizer.step()
    schedulers[0].step()
    assert optimizer.param_groups[0]['lr'] == original * 0.5
