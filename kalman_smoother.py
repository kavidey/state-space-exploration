# %%
import time
from typing import Tuple
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
import numpy.random as npr
import numpy as np

import torch

import jax
import jax.numpy as jnp
from jax import nn
from jax import random as jnr

from flax import nnx

import optax
import orbax.checkpoint as ocp

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
# tfpk = tfp.math.psd_kernels

from lib.distributions import MultivariateNormalFullCovariance, kl_divergence
from lib.priors import KalmanFilter

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)
# %%
key = jnr.PRNGKey(42)

warmup_epochs = 125
warmup_kl_weight = 0.001

epochs = 500
batch_size = 32
latent_dim = 4

kl_weight = 0.5
kl_ramp = 30 # The epoch where the KL weight reaches its final value

A_init_epsilon = 0.01
Q_init_stdev = 0.02
model_name = f"svae_lds.ptr_{warmup_epochs}_{warmup_kl_weight}.klw_{kl_weight:.2f}.klr_{kl_ramp}.ep_{epochs}"

checkpoint_path = (Path("vae_checkpoints") / f"{model_name}_{time.strftime('%Y%m%d-%H%M%S')}").absolute()
checkpoint_path.mkdir(parents=True)

ocp_options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2)
# %% Create data set
triangle = lambda t: sawtooth(np.pi * t, width=0.5)
make_dot_trajectory = lambda x0, v: lambda t: triangle(v * (t + (1 + x0) / 2.0))
make_renderer = lambda grid, sigma: lambda x: np.exp(
    -1.0 / 2 * (x - grid) ** 2 / sigma**2
)


def make_dot_data(
    image_width, T, num_steps, x0=0.0, v=0.5, render_sigma=0.2, noise_sigma=0.1
):
    grid = np.linspace(-1, 1, image_width, endpoint=True)
    render = make_renderer(grid, render_sigma)
    x = make_dot_trajectory(x0, v)
    images = np.vstack([render(x(t)) for t in np.linspace(0, T, num_steps)])
    return images + noise_sigma * npr.randn(*images.shape)


x = make_dot_data(10, 10, 50, x0=1.0, v=0.75, render_sigma=0.15, noise_sigma=0.05)

np.random.seed(47)
N = 512
B = 16

inputs = []
for i in range(N):
    inputs.append(
        make_dot_data(10, 10, 50, x0=i / 8, v=0.75, render_sigma=0.15, noise_sigma=0.06)
    )

train_dset = np.stack(inputs)

inputs = []
for i in range(N):
    inputs.append(
        make_dot_data(10, 10, 50, x0=i / 8, v=0.75, render_sigma=0.15, noise_sigma=0.06)
    )
# %% Dataset Visualization
test_dset = np.stack(inputs)
plt.imshow(train_dset[0].T)
plt.show()
# %% Dataset Preparation
torch.manual_seed(47)

train_dataloader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size)

process_batch = jnp.array
setup_batch = process_batch(next(iter(train_dataloader)))
# %% Network Definitions
def initializer_diag_with_noise(epsilon: float):
    '''
    Generates an initializer for the A matrix of a kalman filter.
    The generated matrix is I + epsilon * normal

    Args:
        epsilon: noise factor
    
    Returns:
        (jnr.PRNGKey, tuple) -> jnp.array
    '''
    def initializer(rng: jnr.PRNGKey, shape):
        # make sure its square
        assert shape[-1] == shape[-2]
        x = jnp.eye(shape[-1])
        return x + jnr.normal(rng, x.shape) * epsilon
    
    return initializer

# Converts a vector to a covariance matrix using inverse cholesky decomposition
vec_to_cov_cholesky = tfb.Chain([
    tfb.CholeskyOuterProduct(),
    tfb.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=None)
])
vmap_diag = jax.vmap(jnp.diag)
class Encoder(nnx.Module):
    def __init__(self, data_dim: int, latent_dim: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(data_dim, 6, rngs=rngs)
        self.linear_mu = nnx.Linear(6, latent_dim, rngs=rngs)
        self.linear_logvar = nnx.Linear(6, latent_dim, rngs=rngs)

    def __call__(self, x: jnp.array):
        x = self.linear1(x)
        x = nn.relu(x)
        xhat = self.linear_mu(x)

        # learning logvar = log(sigma^2) ensures that sigma is positive and helps with learning small numbers
        logvar = self.linear_logvar(x)
        sigma = jnp.exp(logvar * 0.5)

        return MultivariateNormalFullCovariance(xhat, vmap_diag(sigma))

class Decoder(nnx.Module):
    def __init__(self, data_dim: int, latent_dim: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(latent_dim, 6, rngs=rngs)
        self.linear2 = nnx.Linear(6, data_dim, rngs=rngs)

    def __call__(self, z):
        z = self.linear1(z)
        z = nn.relu(z)
        z = self.linear2(z)
        return z

class VAE(nnx.Module):
    def __init__(self, data_dim: int, latent_dim: int, rngs: nnx.Rngs):
        self.rngs = rngs
        self.encoder = Encoder(data_dim, latent_dim, rngs=rngs)
        self.decoder = Decoder(data_dim, latent_dim, rngs=rngs)

    def __call__(self, x):
        z = self.encoder(x)
        z_sample = z.sample(self.rngs.noise())
        x_recon = self.decoder(z_sample)
        return x_recon, z

class SVAE_LDS(nnx.Module):
    def __init__(self, data_dim: int, latent_dim: int, rngs: nnx.Rngs):
        self.latent_dim = latent_dim
        self.rngs = rngs

        # Initialize VAE components
        self.encoder = Encoder(data_dim, latent_dim, rngs=rngs)
        self.decoder = Decoder(data_dim, latent_dim, rngs=rngs)

        # Initialize Kalman Filter matrices
        self.A = nnx.Param(initializer_diag_with_noise(epsilon=A_init_epsilon)(rngs.noise(), (self.latent_dim, self.latent_dim)))
        self.b = nnx.Param(jnp.zeros((self.latent_dim,)))
        self.Q_param = nnx.Param(nn.initializers.normal(Q_init_stdev)(rngs.noise(), (int(self.latent_dim*(self.latent_dim+1)/2),)))

    def __call__(self, x):
        z_hat = self.encoder(x)

        z_t_sub_1 = MultivariateNormalFullCovariance(
            jnp.zeros((self.latent_dim),), jnp.eye(self.latent_dim)
        )

        f_dist, p_dist, marginal_loglik = KalmanFilter.run_forward(z_hat, z_t_sub_1, self.A, self.b, self.Q(), jnp.eye(self.latent_dim))
        q_dist = KalmanFilter.run_backward(f_dist, self.A, self.b, self.Q(), jnp.eye(self.latent_dim))
        z_recon = q_dist.sample(self.rngs.noise())
        x_recon = self.decoder(z_recon)

        return x_recon, z_recon, z_hat, f_dist, q_dist, p_dist, marginal_loglik

    def Q(self):
        return vec_to_cov_cholesky.forward(self.Q_param)
# %% VAE Train Code
def loss_fn(model, x, kl_weight):
    bs = x.shape[0]

    recon, q_dist = nnx.vmap(model)(x)

    def unbatched_loss(x, recon, q_dist):
        mse_loss = optax.l2_loss(recon, x)
        k = q_dist.mean().shape[1]
        kl_loss = jax.vmap(lambda q_dist: kl_divergence(q_dist, MultivariateNormalFullCovariance(jnp.zeros((k)), jnp.eye(k))))(q_dist)

        return mse_loss, kl_loss
    
    losses = jax.vmap(unbatched_loss)(x, recon, q_dist)
    mse_loss = jnp.sum(losses[0]) / (bs * x.shape[1])
    kl_loss = jnp.sum(losses[1]) / (bs * x.shape[1])

    loss = mse_loss + kl_loss * kl_weight
    return loss, (mse_loss, kl_loss)

@nnx.jit
def train_step(model: VAE, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch: jnp.array, kl_weight: float):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (mse_loss, kl_loss)), grads = grad_fn(model, batch, kl_weight)
    metrics.update(loss=loss.mean(), mse_loss=mse_loss.mean(), kl_loss=kl_loss.mean())
    optimizer.update(grads)
# %% Create VAE Model
warmup_model = VAE(setup_batch.shape[-1], latent_dim, nnx.Rngs(0))
optimizer = nnx.Optimizer(warmup_model, optax.adamw(learning_rate=1e-3))
metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average('loss'),
    mse_loss=nnx.metrics.Average('mse_loss'),
    kl_loss=nnx.metrics.Average('kl_loss'),
)
# %% VAE Training
# mngr = ocp.CheckpointManager(checkpoint_path/"vae_warmup", options=ocp_options)

metrics_history = {
  'train_loss': [],
  'train_mse_loss': [],
  'train_kl_loss': [],
}

pbar = tqdm(range(warmup_epochs))
for epoch in pbar:
    for i, batch in enumerate(train_dataloader):
        train_step(warmup_model, optimizer, metrics, process_batch(batch), warmup_kl_weight)

    # Log the training metrics.
    for metric, value in metrics.compute().items():  # Compute the metrics.
      metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
    metrics.reset()  # Reset the metrics for the test set.
    pbar.set_postfix_str(f"Loss: {metrics_history['train_loss'][-1]:.2f}")
    
#     mngr.save(epoch, args=ocp.args.StandardSave(params))
# mngr.wait_until_finished()
# %% VAE Reconstruction and Evaluation
# restored_warmup_params = mngr.restore(mngr.latest_step(), warmup_params)

@nnx.jit
def pred_step(model, x):
    return model(x)

i = 0
sample_batch = process_batch(next(iter(test_dataloader)))[i]

recon, q_dist = pred_step(warmup_model, sample_batch)
f, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

ax[0].imshow(sample_batch.T, aspect='auto', vmin=-0.3, vmax=1.3)
ax[0].set_title('Sequence')

ax[1].imshow(recon.T, aspect='auto', vmin=-0.3, vmax=1.3)
ax[1].set_title('Reconstruction')

ax[2].plot(q_dist.mean())
ax[2].set_title('Each Dimension of the Latent Variable')

plt.show()
# %% LDS Train Code
def loss_fn(model, x, kl_weight):
    bs = x.shape[0]

    recon, z_recon, z_hat, f_dist, q_dist, p_dist, marginal_loglik = nnx.vmap(model)(x)

    def unbatched_loss(x, recon, z_hat, q_dist, f_dist, p_dist, marginal_loglik):
        mse_loss = optax.l2_loss(recon, x)
        
        def observation_likelihood(z_hat: MultivariateNormalFullCovariance, q_z: MultivariateNormalFullCovariance, p_z: MultivariateNormalFullCovariance):
            k = z_hat.mean().shape[-1]
            # -1/2 ( k*log(2pi) + log(det(Sigma_i)) + (x_i - mu_i)^T @ Sigma_i^-1 @ (x_i - mu_i) + tr(P_i Sigma_i^-1) )
            mean_diff = z_hat.mean() - q_z.mean()
            inv_cov = jnp.linalg.inv(z_hat.covariance())
            return -(1/2) * (k * jnp.log(2*jnp.pi) + jnp.log(jnp.linalg.det(z_hat.covariance())) + mean_diff.T @ inv_cov @ mean_diff + jnp.linalg.trace(q_z.covariance() @ inv_cov))
        
        kl_loss = jax.vmap(observation_likelihood)(z_hat, q_dist, p_dist) - marginal_loglik
        
        return mse_loss, kl_loss
    
    losses = jax.vmap(unbatched_loss)(x, recon, z_hat, q_dist, f_dist, p_dist, marginal_loglik)
    mse_loss = jnp.sum(losses[0]) / (bs * x.shape[1])
    kl_loss = jnp.sum(losses[1]) / (bs * x.shape[1])

    loss = mse_loss + kl_loss * kl_weight
    return loss, (mse_loss, kl_loss)

@nnx.jit
def train_step(model: VAE, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch: jnp.array, kl_weight: float):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (mse_loss, kl_loss)), grads = grad_fn(model, batch, kl_weight)
    metrics.update(loss=loss.mean(), mse_loss=mse_loss.mean(), kl_loss=kl_loss.mean())
    optimizer.update(grads)
# %% Create LDS Model
model = SVAE_LDS(setup_batch.shape[-1], latent_dim, nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-4))
metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average('loss'),
    mse_loss=nnx.metrics.Average('mse_loss'),
    kl_loss=nnx.metrics.Average('kl_loss'),
)

model.encoder = warmup_model.encoder
model.decoder = warmup_model.decoder
# %% Print LDS Parameters
print("A", model.A.value)
print("b", model.b.value)
print("Q", vec_to_cov_cholesky.forward(model.Q_param.value))
# %% LDS Training
# mngr = ocp.CheckpointManager(checkpoint_path/"lds", options=ocp_options)

metrics_history = {
  'train_loss': [],
  'train_mse_loss': [],
  'train_kl_loss': [],
}

pbar = tqdm(range(epochs))
for epoch in pbar:
    for i, batch in enumerate(train_dataloader):
        train_step(model, optimizer, metrics, process_batch(batch), kl_weight)

    # Log the training metrics.
    for metric, value in metrics.compute().items():  # Compute the metrics.
      metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
    pbar.set_postfix_str(f"Loss: {metrics_history['train_loss'][-1]:.2f}")
    metrics.reset()  # Reset the metrics for the test set.
    
#     mngr.save(epoch, args=ocp.args.StandardSave(params))
# mngr.wait_until_finished()
# %% LDS Loss Curves
plt.plot(metrics_history['train_loss'], label="loss")
plt.plot(metrics_history['train_kl_loss'], label="kl loss")
plt.plot(metrics_history['train_mse_loss'], label="mse loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
# %% LDS Evaluation
# restored_params = mngr.restore(mngr.latest_step(), params)

@nnx.jit
def pred_step(model, x):
    return model(x)

sample_batch = process_batch(next(iter(test_dataloader)))
# %% Print LDS Parameters
print("A", model.A.value)
print("b", model.b.value)
print("Q", vec_to_cov_cholesky.forward(model.Q_param.value))
# %% LDS Reconstruction
i = 0
recon, z_recon, z_hat, f_dist, q_dist, p_dist, marginal_loglik = pred_step(model, sample_batch[i])
f, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
f.tight_layout()

ax[0].imshow(sample_batch[i].T, aspect='auto', vmin=-0.3, vmax=1.3)
ax[0].set_title('Sequence')

ax[1].imshow(recon.T, aspect='auto', vmin=-0.3, vmax=1.3)
ax[1].set_title('Reconstruction')

ax[2].plot(q_dist.mean())
ax[2].set_title('Latent Posterior Mean')

ax[3].plot(jax.vmap(jnp.diag)(q_dist.covariance()))
ax[3].set_title('Latent Posterior Covariance (diagonal elements)')

plt.show()
# %% LDS Imputation
i = 20
masked_batch = sample_batch.at[:,10:40].set(0)
recon, z_recon, z_hat, f_dist, q_dist, p_dist, marginal_loglik = pred_step(model, masked_batch[i])

f, ax = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
f.tight_layout()

ax[0].imshow(masked_batch[i].T, aspect='auto', vmin=-0.3, vmax=1.3)
ax[0].set_title('Masked Sequence')

ax[1].imshow(recon.T, aspect='auto', vmin=-0.3, vmax=1.3)
ax[1].set_title('Reconstruction')

ax[2].plot(q_dist.mean())
ax[2].set_title('Latent Posterior Mean')

ax[3].plot(jax.vmap(jnp.diag)(q_dist.covariance()))
ax[3].set_title('Latent Posterior Covariance (diagonal elements)')

ax[4].plot(z_recon)
ax[4].set_title('Latent Posterior Samples')

plt.show()
# %%
