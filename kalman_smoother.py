# %%
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
import numpy.random as npr

import torch

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from tensorflow_probability.substrates.jax import distributions as tfd

from flax import linen as nn

import optax
import orbax.checkpoint as ocp

# %%

key = random.PRNGKey(42)

epochs = 1000
batch_size = 32

latent_dims = 4
kl_weight = 10
model_name = f"svae_lds.klw_{kl_weight:.2f}"

checkpoint_path = (Path("vae_checkpoints") / model_name).absolute()
checkpoint_path.mkdir(exist_ok=True, parents=True)

options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2)
mngr = ocp.CheckpointManager(checkpoint_path, options=options)
# %%
# Create data set
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
# %%
test_dset = np.stack(inputs)
plt.imshow(train_dset[0].T)
# %%
torch.manual_seed(47)

train_dataloader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size)

process_batch = jnp.array
setup_batch = next(iter(train_dataloader))


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
        sigma = jnp.exp(logvar * 0.5)

        return tfd.MultivariateNormalFullCovariance(xhat, jnp.diag(sigma))


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z):
        z = nn.Dense(400, name="dec_fc1")(z)
        z = nn.relu(z)
        z = nn.Dense(784, name="dec_fc2")(z)
        return z


class SVAE_LDS(nn.Module):
    latent_dims: int

    def setup(self):
        # Initialize VAE components
        self.encoder = Encoder(latent_dims=latent_dims, name="encoder")
        self.decoder = Decoder(name="decoder")

        # Initialize Kalman Filter matrices
        self.A = self.param(
            "kl_A", nn.initializers.xavier_uniform(), (latent_dims, latent_dims)
        )
        self.b = self.param("kl_b", nn.initializers.zeros, (latent_dims,))

    @nn.compact
    def __call__(self, x, z_rng):
        z_t_sub_1 = tfd.MultivariateNormalFullCovariance(
            jnp.zeros((latent_dims,)), jnp.eye(latent_dims)
        )

        x_recon = []
        q_dist = []
        p_dist = []

        for x_t in x:
            # Prediction
            z_t_given_t_sub_1 = self.predict(z_t_sub_1, self.A, self.b)

            # Update
            z_t_given_t = self.update(z_t_given_t_sub_1, x_t)

            z_rng, z_t_rng = random.split(z_rng)
            x_hat = self.decoder(z_t_given_t.sample(1, seed=z_t_rng))

            x_recon.append(x_hat)
            q_dist.append(z_t_given_t)
            p_dist.append(z_t_given_t_sub_1)

        x_recon = jnp.vstack(x_recon)

        return x_recon, q_dist, p_dist

    def predict(self, z_t, A, b):
        """
        P(z_t+1 | x_t, ..., x_1) = P(z_t+1 | z_t)
        """
        # z_t|t-1 = A @ z_t-1|t-1 + b
        mu = A @ z_t.mean() + b

        Q = jnp.zeros((latent_dims, latent_dims))

        # P_t|t-1 = A @ P_t-1|t-1 @ A^T + Q
        sigma = A @ z_t.covariance() @ A.T + Q

        return tfd.MultivariateNormalFullCovariance(mu, sigma)

    def update(
        self, z_t_given_t_sub_1: tfd.MultivariateNormalFullCovariance, x_t: jnp.array
    ):
        """
        Kalman filter update step
        P(z_t+1 | x_t+1, ... , x_1) ~= P(x_t+1 | z_t+1) * P(z_t+1 | x_t, ... x_1)

        Args:
            z_t_given_t_sub_1 (tfd.MultivariateNormalFullCovariance): z_t|t-1
            x_t (tfd.MultivariateNormalFullCovariance): x_t

        Returns:
            tfd.MultivariateNormalFullCovariance: z_t|t
        """

        H = jnp.eye(latent_dims)
        # R = jnp.zeros((latent_dims, latent_dims))

        # K_t = P_t|t-1 @ H^T @ (H @ P_t|t-1 @ H^T + R) ^ -1
        # K_t = z_t_given_t_sub_1.covariance() @ H.T @ jnp.linalg.inv(H @ z_t_given_t_sub_1.covariance() @ H.T + R)

        # If we assume that H is the identity, the the above simplifies to the identity
        K_t = jnp.eye(latent_dims)

        # z_t|t = z_t|t-1 + K_t @ (x_t - H @ z_t|t-1)
        mu = z_t_given_t_sub_1.mean() + K_t @ (
            x_t.mean() - H @ z_t_given_t_sub_1.mean()
        )

        # P_t|t = P_t|t-1 - K_t @ H @ P_t|t-1 = (I - K_t @ H) @ P_t|t-1
        sigma = (jnp.eye(latent_dims) - K_t @ H) @ z_t_given_t_sub_1

        return tfd.MultivariateNormalFullCovariance(mu, sigma)

model = SVAE_LDS(latent_dims=latent_dims)
model.init(key, setup_batch, random.PRNGKey)
# %%
process_batch = jnp.array
setup_batch = next(iter(train_dataloader))

def kl_divergence(q: tfd.MultivariateNormalFullCovariance, p: tfd.MultivariateNormalFullCovariance):
    mu_0 = q.mean()
    sigma_0 = q.covariance()

    mu_1 = p.mean()
    sigma_1 = p.covariance()

    k = mu_0.shape[0]

    # \frac{1}{2} (\text{tr}(\Sigma_1^{-1}\Sigma_0) + (\mu_1 - \mu_0)^T \Sigma_1^{-1} (\mu_1-\mu_0)-k+\log(\frac{\det \Sigma_1}{\det \Sigma_0}))
    return 0.5 * jnp.trace(jnp.linalg.inv(sigma_1) @ sigma_0) + (mu_1 - mu_0).T @ jnp.linalg.inv(sigma_1) @ (mu_1 - mu_0) - k + jnp.log(jnp.linalg.det(sigma_1) / jnp.linalg.det(sigma_1))

def create_train_step(key: jax.random.PRNGKey, model: nn.Module, optimizer: optax.GradientTransformation):
    params = model.init(key, setup_batch, random.PRNGKey)
    opt_state = optimizer.init(params)

    def loss_fn(params, x, key):
        recon, q_dist, p_dist = model.apply(params, x, key)

        mse_loss = optax.l2_loss(recon, x).sum() / x.shape[0]

        kl_loss = 0
        for q_z, p_z in zip(q_dist, p_dist):
            kl_loss += kl_divergence(q_z, p_z)
        
        loss = mse_loss + kl_weight * kl_loss
        return loss, (mse_loss, kl_loss)
    
    @jax.jit
    def train_step(params, opt_state, x, key):
        losses, grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x, key)
        
        loss, (mse_loss, kl_loss) = losses
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss, mse_loss, kl_weight
    
    return train_step, params, opt_state
# %%
key, model_key = jax.random.split(key)

model = SVAE_LDS(latent_dims=latent_dims)
optimizer = optax.adamw(learning_rate=1e-4)

train_step, params, opt_state = create_train_step(model_key, model, optimizer)
# %%
running_loss = []

pbar = tqdm(range(epochs))
for epoch in pbar:
    total_loss, total_mse, total_kl = 0.0, 0.0, 0.0
    for i, (batch, c) in enumerate(train_dataloader):
        key, subkey = jax.random.split(key)

        batch = batch.numpy().reshape(batch_size, 784)
        params, opt_state, loss, mse_loss, kl_loss = train_step(
            params, opt_state, batch, subkey
        )

        total_loss += loss
        total_mse += mse_loss
        total_kl += kl_loss

        pbar.set_postfix_str(f"Loss: {total_loss/i:.2f}")
    running_loss.append(total_loss/len(train_dataloader))
    mngr.save(epoch, args=ocp.args.StandardSave(params))
mngr.wait_until_finished()
# %%
plt.plot(running_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()