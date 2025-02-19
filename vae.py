# %%
# Imports
from pathlib import Path
from typing import Any, Sequence, Callable, NamedTuple, Optional, Tuple

PyTree = Any  # Type definition for PyTree, for readability
import matplotlib.pyplot as plt
from numpy import einsum
import numpy as np

from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from jax import nn
from jax import random as jnr

from flax import nnx

import optax

import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import orbax.checkpoint as ocp
# %%
model_name = "vae"

key = jnr.PRNGKey(42)

batch_size = 64
validation_split = 0.2
epochs = 100

kl_weight = 1
latent_dims = 20

checkpoint_path = (Path("vae_checkpoints") / model_name).absolute()
checkpoint_path.mkdir(exist_ok=True, parents=True)

dataset_path = Path("dataset/mnist")
dataset_path.mkdir(exist_ok=True)

options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2)
mngr = ocp.CheckpointManager(checkpoint_path, options=options)
# %%
train_dataset = MNIST(dataset_path, train=True, transform=T.ToTensor(), download=True)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)

test_dataset = MNIST(dataset_path, train=False, transform=T.ToTensor(), download=True)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)
# %%
class Encoder(nnx.Module):
    def __init__(self, data_dim: int, latent_dim: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(data_dim, 400, rngs=rngs)
        self.linear_mu = nnx.Linear(400, latent_dim, rngs=rngs)
        self.linear_logvar = nnx.Linear(400, latent_dim, rngs=rngs)

    def __call__(self, x):
        x = self.linear1(x)
        x = nn.relu(x)
        mu = self.linear_mu(x)
        # learning logvar = log(sigma^2) ensures that sigma is positive and helps with learning small numbers
        logvar = self.linear_logvar(x)
        return mu, logvar


class Decoder(nnx.Module):
    def __init__(self, data_dim: int, latent_dim: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(latent_dim, 400, rngs=rngs)
        self.linear2 = nnx.Linear(400, data_dim, rngs=rngs)

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
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def reparameterize(self, mu, logvar):
        eps = jnr.normal(self.rngs.noise(), mu.shape)
        # convert logvar back to sigma and sample from learned distribution
        return eps * jnp.exp(logvar * 0.5) + mu

    def decode(self, z):
        return self.decoder(z)
# %%
def loss_fn(model, x):
    reduce_dims = list(range(1, len(x.shape)))
    recon, mean, logvar = model(x)
    mse_loss = optax.l2_loss(recon, x).sum(axis=reduce_dims).mean()
    kl_loss = jnp.mean(
        -0.5 * jnp.sum(1 + logvar - mean**2 - jnp.exp(logvar), axis=reduce_dims)
    )  # KL loss term to keep encoder output close to standard normal distribution.

    loss = mse_loss + kl_weight * kl_loss
    return loss, (mse_loss, kl_loss)

@nnx.jit
def train_step(model: VAE, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, (mse_loss, kl_loss)), grads = grad_fn(model, batch)
  metrics.update(loss=loss, mse_loss=mse_loss, kl_loss=kl_loss)
  optimizer.update(grads)

@nnx.jit
def eval_step(model: VAE, metrics: nnx.MultiMetric, batch):
  loss, (mse_loss, kl_loss) = loss_fn(model, batch)
  metrics.update(loss=loss, mse_loss=mse_loss, kl_loss=kl_loss)
# %%
model = VAE(784, latent_dims, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-4))
metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average('loss'),
    mse_loss=nnx.metrics.Average('mse_loss'),
    kl_loss=nnx.metrics.Average('kl_loss'),
)
# %% Train
metrics_history = {
  'train_loss': [],
  'train_mse_loss': [],
  'train_kl_loss': [],
  'test_loss': [],
  'test_mse_loss': [],
  'test_kl_loss': [],
}

pbar = tqdm(range(epochs))
for epoch in pbar:
    for i, (batch, c) in enumerate(train_loader):
        train_step(model, optimizer, metrics, batch.numpy().reshape(batch_size, 784))

    # Log the training metrics.
    for metric, value in metrics.compute().items():  # Compute the metrics.
      metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
    metrics.reset()  # Reset the metrics for the test set.

    # Compute the metrics on the test set after each training epoch.
    for test_batch, c in test_loader:
      eval_step(model, metrics, test_batch.numpy().reshape(batch_size, 784))

    # Log the test metrics.
    for metric, value in metrics.compute().items():
      metrics_history[f'test_{metric}'].append(value)
    metrics.reset()  # Reset the metrics for the next training epoch.
    pbar.set_postfix_str(f"Loss: {metrics_history['test_loss'][-1]:.2f}")
    
#     mngr.save(epoch, args=ocp.args.StandardSave(params))
# mngr.wait_until_finished()
# %% Training Metrics
plt.plot(metrics_history['test_loss'], label="loss")
plt.plot(metrics_history['test_kl_loss'], label="kl loss")
plt.plot(metrics_history['test_mse_loss'], label="mse loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
# %% Sample
# restored_params = mngr.restore(mngr.latest_step(), params)
model.eval()

@nnx.jit
def sample_fn(z: jnp.array):
    return model.decode(z)

num_samples = 100
h = w = 10

key, z_key = jax.random.split(key)
z = jax.random.normal(z_key, (num_samples, latent_dims))
sample = sample_fn(z)

sample = einsum("ikjl", np.asarray(sample).reshape(h, w, 28, 28)).reshape(
    28 * h, 28 * w
)
plt.imshow(sample, cmap="gray")
plt.show()
# %%
