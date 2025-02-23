# %%
import time
from typing import Tuple
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
import numpy.random as npr

import torch

import jax
import jax.numpy as jnp

import jax.random as jnr
import numpy as np

from flax import linen as nn

import optax
import orbax.checkpoint as ocp

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
# tfpk = tfp.math.psd_kernels

from lib.distributions import MultivariateNormalFullCovariance, MVN_kl_divergence
from lib.priors import KalmanFilter

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)
# %%
key = jnr.PRNGKey(42)

warmup_epochs = 125
warmup_kl_weight = 0.1

epochs = 500
batch_size = 32
latent_dims = 4

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
class Encoder(nn.Module):
    latent_dims: int

    @nn.compact
    def __call__(self, x: jnp.array):
        x = nn.Dense(6, name="enc_fc1")(x)
        x = nn.relu(x)
        xhat = nn.Dense(latent_dims, name="enc_fc2_xhat")(x)

        # learning logvar = log(sigma^2) ensures that sigma is positive and helps with learning small numbers
        logvar = nn.Dense(latent_dims, name="enc_fc2_logvar")(x)
        sigma = jnp.exp(logvar * 0.5)

        return MultivariateNormalFullCovariance(xhat, jax.vmap(jnp.diag)(sigma))


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z):
        z = nn.Dense(6, name="dec_fc1")(z)
        z = nn.relu(z)
        z = nn.Dense(10, name="dec_fc2")(z)
        return z

class VAE(nn.Module):
    latent_dims: int

    @nn.compact
    def __call__(self, x, z_rng):
        z = Encoder(latent_dims=self.latent_dims, name="encoder")(x)
        z_sample = z.sample(z_rng)
        x_recon = Decoder(name="decoder")(z_sample)

        return x_recon, z
    
Batched_VAE = nn.vmap(VAE, 
    in_axes=0, out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False}
)

class SVAE_LDS(nn.Module):
    latent_dims: int

    def setup(self):
        # Initialize VAE components
        self.encoder = Encoder(latent_dims=self.latent_dims, name="encoder")
        self.decoder = Decoder(name="decoder")

        # Initialize Kalman Filter matrices
        self.A = self.param(
            "kf_A", initializer_diag_with_noise(epsilon=A_init_epsilon), (self.latent_dims, self.latent_dims)
        )
        self.b = self.param("kf_b", nn.initializers.zeros, (self.latent_dims,))
        self.Q_param = self.param(
            "kf_Q", nn.initializers.normal(Q_init_stdev), (int(self.latent_dims*(self.latent_dims+1)/2),)
        )

    def __call__(self, x, z_rng):
        z_hat = self.encoder(x)

        z_t_sub_1 = MultivariateNormalFullCovariance(
            jnp.zeros((self.latent_dims),), jnp.eye(self.latent_dims)
        )

        f_dist, p_dist, marginal_loglik = KalmanFilter.run_forward(z_hat, z_t_sub_1, self.A, self.b, self.Q(), jnp.eye(self.latent_dims))
        q_dist = KalmanFilter.run_backward(f_dist, self.A, self.b, self.Q(), jnp.eye(self.latent_dims))
        z_recon = q_dist.sample(z_rng)
        x_recon = self.decoder(z_recon)

        return x_recon, z_recon, z_hat, f_dist, q_dist, p_dist, marginal_loglik

    def Q(self):
        return vec_to_cov_cholesky.forward(self.Q_param)
    
Batched_SVAE_LDS = nn.vmap(SVAE_LDS, 
    in_axes=0, out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False}
)
# %% VAE Train Code

def create_train_step_warmup(
            key: jnr.PRNGKey, model: nn.Module, optimizer: optax.GradientTransformation
):
    params = model.init(key, setup_batch, jnr.split(jnr.PRNGKey(0), setup_batch.shape[0]))
    opt_state = optimizer.init(params)

    def loss_fn(params, x, key, kl_weight):
        bs = x.shape[0]

        recon, q_dist = model.apply(params, x, jnr.split(key, x.shape[0]))

        def unbatched_loss(x, recon, q_dist):
            mse_loss = optax.l2_loss(recon, x)
            k = q_dist.mean().shape[1]
            kl_loss = jax.vmap(lambda q_dist: MVN_kl_divergence(q_dist.mean(), q_dist.covariance(), jnp.zeros((k)), jnp.eye(k)))(q_dist)

            return mse_loss, kl_loss
        
        losses = jax.vmap(unbatched_loss)(x, recon, q_dist)
        mse_loss = jnp.sum(losses[0]) / (bs * x.shape[1])
        kl_loss = jnp.sum(losses[1]) / (bs * x.shape[1])

        loss = mse_loss + kl_loss * kl_weight
        return loss, (mse_loss, kl_loss)

    @jax.jit
    def train_step(params, opt_state, x, key, kl_weight):
        losses, grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x, key, kl_weight)

        loss, (mse_loss, kl_loss) = losses
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss, mse_loss, kl_loss

    return train_step, params, opt_state
# %% Create VAE Model
key, model_key = jnr.split(key)

warmup_model = Batched_VAE(latent_dims=latent_dims)
optimizer = optax.adam(learning_rate=1e-3)

train_step, warmup_params, opt_state = create_train_step_warmup(model_key, warmup_model, optimizer)
# %% VAE Training
mngr = ocp.CheckpointManager(checkpoint_path/"vae_warmup", options=ocp_options)
pbar = tqdm(range(warmup_epochs))
for epoch in pbar:
    total_loss = 0.0
    for i, batch in enumerate(train_dataloader):
        key, subkey = jnr.split(key)

        batch = process_batch(batch)
        warmup_params, opt_state, loss, mse_loss, kl_loss = train_step(
            warmup_params, opt_state, batch, subkey, warmup_kl_weight
        )

        total_loss += loss

    pbar.set_postfix_str(f"Loss: {total_loss/len(train_dataloader):.2f}")
    mngr.save(epoch, args=ocp.args.StandardSave(warmup_params))
mngr.wait_until_finished()
# %% VAE Reconstruction and Evaluation
restored_warmup_params = mngr.restore(mngr.latest_step(), warmup_params)

def create_pred_step(model: nn.Module, params):
    @jax.jit
    def pred_step(x, key):
        recon, q_dist = model.apply(params, x, jnr.split(key, x.shape[0]))
        return recon, q_dist
    
    return pred_step

sample_batch = process_batch(next(iter(test_dataloader)))
pred_step = create_pred_step(warmup_model, restored_warmup_params)

recon, q_dist = pred_step(sample_batch, key)
f, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
i = 0

ax[0].imshow(sample_batch[i].T, aspect='auto', vmin=-0.3, vmax=1.3)
ax[0].set_title('Sequence')

ax[1].imshow(recon[i].T, aspect='auto', vmin=-0.3, vmax=1.3)
ax[1].set_title('Reconstruction')

ax[2].plot(q_dist.mean()[i])
ax[2].set_title('Each Dimension of the Latent Variable')

plt.show()
# %% LDS Train Code
def create_train_step(
    key: jnr.PRNGKey, model: nn.Module, optimizer: optax.GradientTransformation
):
    params = model.init(key, setup_batch, jnr.split(jnr.PRNGKey(0), setup_batch.shape[0]))
    opt_state = optimizer.init(params)

    def loss_fn(params, x, key, kl_weight):
        bs = x.shape[0]

        recon, z_recon, z_hat, f_dist, q_dist, p_dist, marginal_loglik = model.apply(params, x, jnr.split(key, x.shape[0]))

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

    @jax.jit
    def train_step(params, opt_state, x, key, kl_weight):
        losses, grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x, key, kl_weight)

        loss, (mse_loss, kl_loss) = losses
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss, mse_loss, kl_loss

    return train_step, params, opt_state

# %% Create LDS Model
key, model_key = jnr.split(key)

model = Batched_SVAE_LDS(latent_dims=latent_dims)
optimizer = optax.adam(learning_rate=1e-3)
# optimizer = optax.sgd(learning_rate=1e-3)

train_step, params, opt_state = create_train_step(model_key, model, optimizer)
params['params'].update({'encoder': warmup_params['params']['encoder'],
                         'decoder': warmup_params['params']['decoder']})
# %% Print LDS Parameters
print("learned parameters", params["params"].keys())
print("A", params["params"]["kf_A"])
print("b", params["params"]["kf_b"])
print("Q", vec_to_cov_cholesky.forward(params["params"]["kf_Q"]))
# %% LDS Training
mngr = ocp.CheckpointManager(checkpoint_path/"lds", options=ocp_options)

running_loss = []
running_mse = []
running_kl = []

pbar = tqdm(range(epochs + kl_ramp))
for epoch in pbar:
    total_loss, total_mse, total_kl = 0.0, 0.0, 0.0
    for i, batch in enumerate(train_dataloader):
        key, subkey = jnr.split(key)

        batch = process_batch(batch)
        params1, opt_state, loss, mse_loss, kl_loss = train_step(
            params, opt_state, batch, subkey, min(epoch / kl_ramp, 1) * kl_weight
        )

        contains_nan = False
        for a in jax.tree.flatten(params1)[0]:
            if jnp.isnan(jnp.sum(a)):
                contains_nan=True
        
        if not contains_nan:
            params = params1
        else:
            raise Exception("Parameters contain NaN: Halting Training")


        total_loss += loss
        total_mse += mse_loss
        total_kl += kl_loss

    pbar.set_postfix_str(f"Loss: {total_loss/len(train_dataloader):.2f}")
    running_loss.append(total_loss / len(train_dataloader))
    running_mse.append(total_mse / len(train_dataloader))
    running_kl.append(total_kl / len(train_dataloader))
    mngr.save(epoch, args=ocp.args.StandardSave(params))
mngr.wait_until_finished()
# %% LDS Loss Curves
plt.plot(running_loss, label='Total Loss')
plt.plot(running_mse, label='MSE Loss')
plt.plot(running_kl, label='KL Loss')

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
# %% LDS Evaluation
restored_params = mngr.restore(mngr.latest_step(), params)

def create_pred_step(model: nn.Module, params):
    @jax.jit
    def pred_step(x, key):
        x_recon, z_recon, z_hat, f_dist, q_dist, p_dist, marginal_loglik = model.apply(params, x, jnr.split(key, x.shape[0]))
        return x_recon, z_recon, z_hat, f_dist, q_dist, p_dist, marginal_loglik
    
    return pred_step

sample_batch = process_batch(next(iter(test_dataloader)))
pred_step = create_pred_step(model, restored_params)
# %% Print LDS Parameters
print("learned parameters", restored_params["params"].keys())
print("A", restored_params["params"]["kf_A"])
print("b", restored_params["params"]["kf_b"])
print("Q", vec_to_cov_cholesky.forward(restored_params["params"]["kf_Q"]))
# %% LDS Reconstruction
recon, z_recon, z_hat, f_dist, q_dist, p_dist, marginal_loglik = pred_step(sample_batch, key)
f, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
f.tight_layout()
i = 0

ax[0].imshow(sample_batch[i].T, aspect='auto', vmin=-0.3, vmax=1.3)
ax[0].set_title('Sequence')

ax[1].imshow(recon[i].T, aspect='auto', vmin=-0.3, vmax=1.3)
ax[1].set_title('Reconstruction')

ax[2].plot(q_dist.mean()[i])
ax[2].set_title('Latent Posterior Mean')

ax[3].plot(jax.vmap(jnp.diag)(q_dist.covariance()[i]))
ax[3].set_title('Latent Posterior Covariance (diagonal elements)')

plt.show()
# %% LDS Imputation
masked_batch = sample_batch.at[:,10:40].set(0)
recon, z_recon, z_hat, f_dist, q_dist, p_dist, marginal_loglik = pred_step(masked_batch, key)

f, ax = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
f.tight_layout()
i = 20

ax[0].imshow(masked_batch[i].T, aspect='auto', vmin=-0.3, vmax=1.3)
ax[0].set_title('Masked Sequence')

ax[1].imshow(recon[i].T, aspect='auto', vmin=-0.3, vmax=1.3)
ax[1].set_title('Reconstruction')

ax[2].plot(q_dist.mean()[i])
ax[2].set_title('Latent Posterior Mean')

ax[3].plot(jax.vmap(jnp.diag)(q_dist.covariance()[i]))
ax[3].set_title('Latent Posterior Covariance (diagonal elements)')

ax[4].plot(z_recon[i])
ax[4].set_title('Latent Posterior Samples')

plt.show()
# %%
