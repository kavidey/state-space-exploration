# %% [markdown]
# Note: data loading code is modified from the following sources
# - https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial4/Optimization_and_Initialization.html
# - https://github.com/wandb/examples/blob/master/colabs/jax/Simple_Training_Loop_in_JAX_and_Flax.ipynb
# - https://github.com/pytorch/examples/blob/main/vae/main.py
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
from jax import random

import flax
from flax import linen as nn
from flax.training import train_state, checkpoints

import optax

import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
import orbax.checkpoint as ocp
# %%
model_name = "vae"

key = random.PRNGKey(42)

batch_size = 64
validation_split = 0.2
epochs = 100

kl_weight = 1
latent_dims = 20

checkpoint_path = (Path("vae_checkpoints") / model_name).absolute()
checkpoint_path.mkdir(exist_ok=True, parents=True)

dataset_path = Path("minst_dataset")
dataset_path.mkdir(exist_ok=True)

options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2)
mngr = ocp.CheckpointManager(checkpoint_path, options=options)
# %%
train_dataset = MNIST(dataset_path, train=True, transform=T.ToTensor(), download=True)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)


# %%
class Encoder(nn.Module):
    latent_dims: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(400, name="enc_fc1")(x)
        x = nn.relu(x)
        mu = nn.Dense(latent_dims, name="enc_fc2_mu")(x)
        # learning logvar = log(sigma^2) ensures that sigma is positive and helps with learning small numbers
        logvar = nn.Dense(latent_dims, name="enc_fc2_logvar")(x)
        return mu, logvar


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z):
        z = nn.Dense(400, name="dec_fc1")(z)
        z = nn.relu(z)
        z = nn.Dense(784, name="dec_fc2")(z)
        return z


class VAE(nn.Module):
    latent_dims: int

    def setup(self):
        self.encoder = Encoder(latent_dims=latent_dims, name="encoder")
        self.decoder = Decoder(name="decoder")

    @nn.compact
    def __call__(self, x, z_rng):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar, z_rng)
        return self.decode(z), mu, logvar

    def reparameterize(self, mu, logvar, rng):
        eps = random.normal(rng, mu.shape)
        # convert logvar back to sigma and sample from learned distribution
        return eps * jnp.exp(logvar * 0.5) + mu

    def decode(self, z):
        return self.decoder(z)


# %%
def create_train_step(key, model, optimiser):
    params = model.init(
        key, jnp.zeros((batch_size, 784)), jax.random.PRNGKey(0)
    )  # dummy key just as example input
    opt_state = optimiser.init(params)

    def loss_fn(params, x, key):
        reduce_dims = list(range(1, len(x.shape)))
        recon, mean, logvar = model.apply(params, x, key)
        mse_loss = optax.l2_loss(recon, x).sum(axis=reduce_dims).mean()
        kl_loss = jnp.mean(
            -0.5 * jnp.sum(1 + logvar - mean**2 - jnp.exp(logvar), axis=reduce_dims)
        )  # KL loss term to keep encoder output close to standard normal distribution.

        loss = mse_loss + kl_weight * kl_loss
        return loss, (mse_loss, kl_loss)

    @jax.jit
    def train_step(params, opt_state, x, key):
        losses, grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x, key)
        loss, (mse_loss, kl_loss) = losses

        updates, opt_state = optimiser.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss, mse_loss, kl_loss

    return train_step, params, opt_state


# %% Initialize the Model
key, model_key = jax.random.split(key)

model = VAE(latent_dims=latent_dims)
optimiser = optax.adamw(learning_rate=1e-4)

train_step, params, opt_state = create_train_step(model_key, model, optimiser)
# %% Train

running_loss = []

freq = 500
pbar = tqdm(range(100))
for epoch in pbar:
    total_loss, total_mse, total_kl = 0.0, 0.0, 0.0
    for i, (batch, c) in enumerate(train_loader):
        key, subkey = jax.random.split(key)

        batch = batch.numpy().reshape(batch_size, 784)
        params, opt_state, loss, mse_loss, kl_loss = train_step(
            params, opt_state, batch, subkey
        )

        total_loss += loss
        total_mse += mse_loss
        total_kl += kl_loss

        pbar.set_postfix_str(f"Loss: {total_loss/i:.2f}")
    running_loss.append(total_loss/len(train_loader))
    mngr.save(epoch, args=ocp.args.StandardSave(params))
mngr.wait_until_finished()
# %% Training Metrics
plt.plot(running_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
# %% Sample
restored_params = mngr.restore(mngr.latest_step(), args=ocp.args.StandardSave(params))
def build_sample_fn(model, params):
    def sample_fn(z: jnp.array) -> jnp.array:
        return model.apply(params, z, method=model.decode)

    return sample_fn


sample_fn = build_sample_fn(model, restored_params)

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
