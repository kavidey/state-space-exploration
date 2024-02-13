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
model_name = "svae_vae.normal_prior.no_kl"

key = random.PRNGKey(42)

batch_size = 64
validation_split = 0.2
epochs = 50

kl_weight = 10
latent_dims = 2

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
        xhat = nn.Dense(latent_dims, name="enc_fc2_xhat")(x)
        # learning logvar = log(sigma^2) ensures that sigma is positive and helps with learning small numbers
        logvar = nn.Dense(latent_dims, name="enc_fc2_logvar")(x)
        return xhat, logvar


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
        xhat, logvar = self.encoder(x)
        # Convert logvar back to sigma and sample from learned distribution
        sigma_0 = jnp.exp(logvar * 0.5)

        # Calculate the posterior using bayes rule
        mu_star, sigma_star = self.posterior(xhat, sigma_0)

        z = self.reparameterize(mu_star, sigma_star, z_rng)
        return self.decode(z), mu_star, sigma_star
    
    def posterior(self, xhat, sigma_0):
        mu_0 = xhat
        
        sigma_prior = 1
        mu_prior = 1

        sigma_star = 1/(1/sigma_0 + 1/sigma_prior)
        mu_star = sigma_star * (mu_0 / sigma_0 + mu_prior/sigma_prior)

        return mu_star, sigma_star

    def reparameterize(self, mu, sigma, rng):
        eps = random.normal(rng, mu.shape)
        return eps * sigma + mu

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
        recon, mean, sigma = model.apply(params, x, key)
        logvar = 2 * jnp.log(sigma)
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

pbar = tqdm(range(epochs))
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
plt.show()
# %% Sample
restored_params = mngr.restore(mngr.latest_step(), args=ocp.args.StandardSave(params))
def build_sample_fn(model, params):
    def sample_fn(z: jnp.array) -> jnp.array:
        return model.apply(params, z, method=model.decode)

    return sample_fn


sample_fn = build_sample_fn(model, restored_params)

num_samples = 100
h = w = 15

# key, z_key = jax.random.split(key)
# z = jax.random.normal(z_key, (num_samples, latent_dims))
latent_size = 2
ypts = jnp.linspace(-latent_size, latent_size, h)
xpts = jnp.linspace(-latent_size, latent_size, w)
z_ = jnp.meshgrid(xpts, ypts)
z = jnp.vstack((z_[0].flatten(), z_[1].flatten())).T

sample = sample_fn(z)

sample = einsum("ikjl", np.asarray(sample).reshape(h, w, 28, 28)).reshape(
    28 * h, 28 * w
)
plt.imshow(sample, cmap="gray")
plt.show()
# %%
