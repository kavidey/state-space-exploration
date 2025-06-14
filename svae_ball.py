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
from jax import Array

import jax.random as jnr
import numpy as np

from flax import linen as nn

import optax
import orbax.checkpoint as ocp

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
# tfpk = tfp.math.psd_kernels

from lib.distributions import MVN_kl_divergence
from lib.priors import KalmanFilter_MOTPDA

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_platform_name', 'cpu')
# %%
dset_len = 1024
embedding_dim = 10
num_balls = 3
pos_sorted = True
pos_dims = 2 + pos_sorted

key = jnr.PRNGKey(42)

warmup_epochs = 30
warmup_kl_weight = 0.01

epochs = 200
batch_size = 128
latent_dims = 2*2

kl_weight = 0.05
kl_ramp = 5

A_init_epsilon = 0.01
Q_init_stdev = 0.02
model_name = f"svae_ball.ptr_{warmup_epochs}_{warmup_kl_weight}.klw_{kl_weight:.2f}.ep_{epochs}"

checkpoint_path = (Path("vae_checkpoints") / f"{model_name}_{time.strftime('%Y%m%d-%H%M%S')}").absolute()
checkpoint_path.mkdir(parents=True)

ocp_options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2)
# %% Dataset Preparation
torch.manual_seed(47)

dataset_dir = Path("dataset") / "billiard"
train_dset = jnp.load(dataset_dir/"train.npz")
test_dset = jnp.load(dataset_dir/"test.npz")
# %%
def get_embedding(pos, embedding_dim, n=1000):
    return jnp.sin(pos/n**(2*np.arange(embedding_dim)/embedding_dim))
get_embedding_b = jax.vmap(jax.vmap(get_embedding, (0, None, None)), (0, None, None))

for i in jnp.linspace(-10, 10, 20):
    plt.plot(get_embedding(i, 10, n=4))
# %%
if False:
    train = []
    test = []
    for i in range(dset_len):
        positions = jnp.reshape(train_dset['y'][i][..., :2], (-1, 6))
        positions = positions[:, :2*num_balls]
        
        positions_sorted = []
        for j in range(positions.shape[0]):
            positions_sorted.append(
                jnp.concatenate((
                    jnp.reshape(
                        get_embedding_b(positions[j].reshape((num_balls,2)), embedding_dim, 4),
                        (num_balls, -1)),
                    jnp.expand_dims(jnp.arange(num_balls), axis=-1)), axis=1)
            )
        positions_and_metadata = jnp.concatenate([jnp.reshape((positions-5)/5, (positions.shape[0], num_balls, 2)), jnp.repeat(jnp.arange(num_balls)[None, ..., None], positions.shape[0], axis=0)], axis=-1)
        positions_sorted = jnp.concatenate([jnp.array(positions_sorted)] + [positions_and_metadata], axis=2)

        train.append(np.asarray(positions_sorted))
    
    train = np.array(train)

    for i in range(100):
        positions = jnp.reshape(test_dset['y'][i][..., :2], (-1, 6))
        positions = positions[:, :2*num_balls]

        positions_sorted = []
        for j in range(positions.shape[0]):
            positions_sorted.append(
                jnp.concatenate((
                    jnp.reshape(
                        get_embedding_b(positions[j].reshape((num_balls,2)), embedding_dim, 4),
                        (num_balls, -1)),
                    jnp.expand_dims(jnp.arange(num_balls), axis=-1)), axis=1)
            )
        positions_and_metadata = jnp.concatenate([jnp.reshape((positions-5)/5, (positions.shape[0], num_balls, 2)), jnp.repeat(jnp.arange(num_balls)[None, ..., None], positions.shape[0], axis=0)], axis=-1)
        positions_sorted = jnp.concatenate([jnp.array(positions_sorted)] + [positions_and_metadata], axis=2)

        test.append(np.asarray(positions_sorted))
    test = np.array(test)

    np.savez(dataset_dir / f"train_embedding_{num_balls}ball.npz", train)
    np.savez(dataset_dir / f"test_embedding_{num_balls}ball.npz", test)
# %%
if False:
    train = []
    test = []
    for i in range(dset_len):
        key, tmp_key = jnr.split(key)
        start_stop = jnr.uniform(tmp_key, (num_balls, 2, 2)) * 10 - 5

        positions = []
        for i in range(num_balls):
            positions.extend([
                jnp.linspace(start_stop[i, 0, 0], start_stop[i, 1, 0]), # x
                jnp.linspace(start_stop[i, 0, 1], start_stop[i, 1, 1]), # y
            ])
        positions = np.array(positions)

        position_vec = jnp.hstack([
            get_embedding_b(p, embedding_dim, 4) for p in positions
        ] + [positions.T])

        train.append(np.asarray(position_vec))

    for i in range(100):
        key, tmp_key = jnr.split(key)
        start_stop = jnr.uniform(tmp_key, (num_balls, 2, 2)) * 10 - 5

        positions = []
        for i in range(num_balls):
            positions.extend([
                jnp.linspace(start_stop[i, 0, 0], start_stop[i, 1, 0]), # x
                jnp.linspace(start_stop[i, 0, 1], start_stop[i, 1, 1]), # y
            ])
        positions = np.array(positions)

        position_vec = jnp.hstack([
            get_embedding_b(p, embedding_dim, 4) for p in positions
        ] + [positions.T])

        test.append(np.asarray(position_vec))

    train = np.array(train)
    test = np.array(test)

    np.savez(dataset_dir / "train_embedding_straight.npz", train)
    np.savez(dataset_dir / "test_embedding_straight.npz", test)
# %%
train = np.load(dataset_dir/f"train_embedding_{num_balls}ball.npz")['arr_0']
test = np.load(dataset_dir/f"test_embedding_{num_balls}ball.npz")['arr_0']

train_dataloader = torch.utils.data.DataLoader(torch.tensor(np.asarray(train)), batch_size=batch_size, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(torch.tensor(np.asarray(test)), batch_size=batch_size, shuffle=False)

process_batch = lambda x: jnp.array(x, dtype='float64')
setup_batch = process_batch(next(iter(train_dataloader)))
# %% Network Definitions
def initializer_diag_with_noise(epsilon: float):
    '''
    Generates an initializer for the A matrix of a kalman filter.
    The generated matrix is I + epsilon * normal

    Args:
        epsilon: noise factor
    
    Returns:
        (Array, tuple) -> jnp.array
    '''
    def initializer(rng: Array, shape):
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
    def __call__(self, x: Array):
        x = nn.Dense(20, name="enc_fc1")(x)
        x = nn.relu(x)
        xhat = nn.Dense(latent_dims, name="enc_fc2_xhat")(x)

        # learning logvar = log(sigma^2) ensures that sigma is positive and helps with learning small numbers
        logvar = nn.Dense(latent_dims, name="enc_fc2_logvar")(x)
        sigma = jnp.exp(logvar * 0.5)

        return xhat, jax.vmap(jnp.diag)(sigma)

Batched_Encoder = nn.vmap(Encoder, 
    in_axes=0, out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False}
)

class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z):
        # z = nn.Dense(4, name="dec_fc1")(z)
        # z = nn.relu(z)
        z = nn.Dense(pos_dims, name="dec_fc2")(z)
        return z

class SVAE_LDS(nn.Module):
    latent_dims: int

    def setup(self):
        # Initialize VAE components
        self.encoder = Batched_Encoder(latent_dims=self.latent_dims, name="encoder")
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

        z_t_sub_1 = (z_hat[0][0], z_hat[1][0])
        z_hat = (z_hat[0][1:], z_hat[1][1:])

        f_dist, q_dist, p_dist, marginal_loglik = jax.vmap(self.track_single_object, in_axes=(None, 0), out_axes=1)(z_hat, z_t_sub_1)

        z_recon = jnr.multivariate_normal(z_rng, q_dist[0], q_dist[1])
        x_recon = self.decoder(z_recon)

        return x_recon, z_recon, z_hat, f_dist, q_dist, p_dist, marginal_loglik
    
    def track_single_object(self, z_hat, z_t_sub_1):
        f_dist, p_dist, marginal_loglik = KalmanFilter_MOTPDA.run_forward(z_hat, z_t_sub_1, self.A, self.b, self.Q(), jnp.eye(self.latent_dims))
        q_dist = KalmanFilter_MOTPDA.run_backward(f_dist, self.A, self.b, self.Q(), jnp.eye(self.latent_dims))

        return f_dist, q_dist, p_dist, marginal_loglik

    def Q(self):
        return vec_to_cov_cholesky.forward(self.Q_param)
    
Batched_SVAE_LDS = nn.vmap(SVAE_LDS, 
    in_axes=0, out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False}
)
# %% LDS Train Code
def create_train_step(
    key: Array, model: nn.Module, optimizer: optax.GradientTransformation
):
    params = model.init(key, setup_batch[..., :-pos_dims], jnr.split(jnr.PRNGKey(0), setup_batch.shape[0]))
    opt_state = optimizer.init(params)

    def loss_fn(params, x, key, kl_weight):
        bs = x.shape[0]

        recon, z_recon, z_hat, f_dist, q_dist, p_dist, marginal_loglik = model.apply(params, x[..., :-pos_dims], jnr.split(key, x.shape[0])) # type: ignore
        x = x[:, 1:]
        def unbatched_loss(x, recon, z_hat, q_dist, f_dist, p_dist, marginal_loglik):
            mse_loss = optax.l2_loss(recon, x[..., -pos_dims:])
            
            def observation_likelihood(z_hat, q_z, p_z):
                k = z_hat[0].shape[-1]
                # -1/2 ( k*log(2pi) + log(det(Sigma_i)) + (x_i - mu_i)^T @ Sigma_i^-1 @ (x_i - mu_i) + tr(P_i Sigma_i^-1) )
                mean_diff = z_hat[0] - q_z[0]
                inv_cov = jnp.linalg.inv(z_hat[1])
                return -(1/2) * (k * jnp.log(2*jnp.pi) + jnp.log(jnp.linalg.det(z_hat[1])) + mean_diff.T @ inv_cov @ mean_diff + jnp.linalg.trace(q_z[1] @ inv_cov))
            
            kl_loss = jax.vmap(observation_likelihood)(z_hat, q_dist, p_dist) - marginal_loglik
            
            return mse_loss, kl_loss
        
        losses = jax.vmap(jax.vmap(unbatched_loss))(x, recon, z_hat, q_dist, f_dist, p_dist, marginal_loglik)
        mse_loss = jnp.sum(losses[0]) / (bs * x.shape[1] * num_balls)
        kl_loss = jnp.sum(losses[1]) / (bs * x.shape[1] * num_balls)

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

pbar = tqdm(range(epochs))
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
recon, z_recon, z_hat, f_dist, q_dist, p_dist, marginal_loglik = pred_step(sample_batch[..., :-pos_dims], key)
f, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
f.tight_layout()
i = 0
j = 0

ax[0].imshow(sample_batch[i,:,j].T, aspect='auto', vmin=-0.3, vmax=1.3)
ax[0].set_title('Sequence')

ax[1].imshow(recon[i,:,j].T, aspect='auto', vmin=-0.3, vmax=1.3)
ax[1].set_title('Reconstruction')

ax[2].plot(q_dist[0][i,:,j])
ax[2].set_title('Latent Posterior Mean')

ax[3].plot(jax.vmap(jnp.diag)(q_dist[1][i,:,j]))
ax[3].set_title('Latent Posterior Covariance (diagonal elements)')

plt.show()
# %%
for j in range(num_balls):
    plt.plot(sample_batch[i,:,j, -pos_dims], sample_batch[i,:,j, -pos_dims+1], c='black', linewidth=2)
    plt.plot(recon[i,:,j,0], recon[i,:,j,1])
plt.xlim(-1, 1)
plt.ylim(-1, 1)
# %% LDS Imputation
# mask = jnp.zeros(sample_batch.shape[:2]).at[:, 20:30].set(1)
# masked_batch = sample_batch * jnp.logical_not(mask)[:, :, None]
# recon, z_recon, z_hat, f_dist, q_dist, p_dist, marginal_loglik = pred_step(sample_batch[..., :pos_dims*num_balls*embedding_dim], mask, key)

# f, ax = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
# f.tight_layout()

# ax[0].imshow(masked_batch[i].T, aspect='auto', vmin=-0.3, vmax=1.3)
# ax[0].set_title('Masked Sequence')

# ax[1].imshow(recon[i].T, aspect='auto', vmin=-0.3, vmax=1.3)
# ax[1].set_title('Reconstruction')

# ax[2].plot(q_dist[0][i])
# ax[2].set_title('Latent Posterior Mean')

# ax[3].plot(jax.vmap(jnp.diag)(q_dist[1][i]))
# ax[3].set_title('Latent Posterior Covariance (diagonal elements)')

# ax[4].plot(z_recon[i])
# ax[4].set_title('Latent Posterior Samples')

# plt.show()
# %%
# masked_with_none = sample_batch.copy().at[jnp.nonzero(jnp.logical_not(mask))].set(jnp.nan)

# for j in range(num_balls):
#     plt.plot(masked_with_none[i,:,pos_dims*num_balls*embedding_dim+pos_dims*j], masked_with_none[i,:,pos_dims*num_balls*embedding_dim+pos_dims*j+1], c='grey', linewidth=6)
#     plt.plot(sample_batch[i,:,pos_dims*num_balls*embedding_dim+pos_dims*j], sample_batch[i,:,pos_dims*num_balls*embedding_dim+pos_dims*j+1], c='black', linewidth=2)
#     plt.plot(recon[i,:,pos_dims*j], recon[i,:,pos_dims*j+1])
# plt.xlim(-5, 5)
# plt.ylim(-5, 5)
# %%
