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

from flax import linen as nn

import optax
import orbax.checkpoint as ocp

# jax.config.update("jax_debug_nans", True)
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
plt.show()
# %%
torch.manual_seed(47)

train_dataloader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size)

process_batch = jnp.array
setup_batch = process_batch(next(iter(train_dataloader)))


# %%
class MultivariateNormalFullCovariance:
    def __init__(self, mean: jnp.ndarray, covariance: jnp.ndarray):
        self.__mean = mean
        self.__covariance = covariance

    def mean(self):
        return self.__mean

    def covariance(self):
        return self.__covariance

    @jax.jit
    def sample(self, seed):
        d = self.__mean.shape[-1]

        # https://juanitorduz.github.io/multivariate_normal/
        # Could also use https://jax.readthedocs.io/en/latest/_autosummary/jax.random.multivariate_normal.html
        epsilon = 0.001
        K = self.__covariance + jnp.identity(d) * epsilon

        L = jnp.linalg.cholesky(K)
        u = random.normal(seed, (d,))

        return self.__mean + jnp.dot(u, L)

    def __repr__(self) -> str:
        return f"MultivariateNormalFullCovariance(mu={self.__mean}, sigma={self.__covariance})"
    
    def _tree_flatten(self):
        children = (self.__mean, self.__covariance)  # arrays / dynamic values
        aux_data = {}  # static values
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

jax.tree_util.register_pytree_node(MultivariateNormalFullCovariance,
                               MultivariateNormalFullCovariance._tree_flatten,
                               MultivariateNormalFullCovariance._tree_unflatten)

# %%
def diag_3d(x):
    '''
    Takes a 3d tensor and turns the last dimension the items on a diagonal matrix, like jnp.diag

    Args:
        x: jnp.array
    
    Returns:
        jnp.array
    '''
    return jnp.einsum("ijk,kl->ijkl", x, jnp.eye(x.shape[-1]))

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

        return MultivariateNormalFullCovariance(xhat, diag_3d(sigma))


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z):
        z = nn.Dense(6, name="dec_fc1")(z)
        z = nn.relu(z)
        z = nn.Dense(10, name="dec_fc2")(z)
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
        self.Q = self.param(
            "kl_Q", nn.initializers.xavier_uniform(), (latent_dims, latent_dims)
        )

    def __call__(self, x, z_rng):
        bs = x.shape[0]

        # swap batch to second dim so it is (seq_len, batch, data)
        x = jnp.permute_dims(x, (1, 0, 2))
        z = self.encoder(x)

        z_t_sub_1 = MultivariateNormalFullCovariance(
            jnp.zeros((bs, latent_dims)), jnp.stack([jnp.eye(latent_dims)] * bs)
        )

        Q = self.Q.T @ self.Q

        def kf_forward(carry, z_t):
            z_rng, z_t_sub_1 = carry

            # Prediction
            z_t_given_t_sub_1 = self.predict(z_t_sub_1, self.A, self.b, Q)

            # Update
            z_t_given_t = self.update(z_t_given_t_sub_1, z_t)

            # Sample and decode
            z_rng, z_t_rng = random.split(z_rng)
            z_hat = z_t_given_t.sample(z_t_rng)

            return (z_rng, z_t_given_t), (z_hat, z_t_given_t, z_t_given_t_sub_1) # carry, (z_recon, q_dist, p_dist)
        
        _, result = jax.lax.scan(kf_forward, (z_rng, z_t_sub_1), z)
        z_recon, q_dist, p_dist = result
        x_recon = self.decoder(jnp.permute_dims(z_recon, (1, 0, 2)))

        return x_recon, q_dist, p_dist

    def predict(self, z_t, A, b, Q):
        """
        P(z_t+1 | x_t, ..., x_1) = P(z_t+1 | z_t)
        """
        # z_t|t-1 = A @ z_t-1|t-1 + b
        mu = jnp.dot(z_t.mean(), A) + b

        # P_t|t-1 = A @ P_t-1|t-1 @ A^T + Q
        sigma = A @ z_t.covariance() @ A.T + Q

        return MultivariateNormalFullCovariance(mu, sigma)

    def update(
        self, z_t_given_t_sub_1: MultivariateNormalFullCovariance, x_t: jnp.array
    ):
        """
        Kalman filter update step
        P(z_t+1 | x_t+1, ... , x_1) ~= P(x_t+1 | z_t+1) * P(z_t+1 | x_t, ... x_1)

        Args:
            z_t_given_t_sub_1 (MultivariateNormalFullCovariance): z_t|t-1
            x_t (MultivariateNormalFullCovariance): x_t

        Returns:
            MultivariateNormalFullCovariance: z_t|t
        """

        H = jnp.eye(latent_dims)
        R = jnp.eye(latent_dims)

        # K_t = P_t|t-1 @ H^T @ (H @ P_t|t-1 @ H^T + R) ^ -1
        K_t = z_t_given_t_sub_1.covariance() @ H.T @ jnp.linalg.inv(H @ z_t_given_t_sub_1.covariance() @ H.T + R)

        # z_t|t = z_t|t-1 + K_t @ (x_t - H @ z_t|t-1)
        # Extra expand_dims and squeeze are necessary to make the matmul dimensions work
        mu = z_t_given_t_sub_1.mean() + jnp.squeeze(K_t @ jnp.expand_dims(x_t.mean() - jnp.dot(z_t_given_t_sub_1.mean(), H), -1))

        # P_t|t = P_t|t-1 - K_t @ H @ P_t|t-1 = (I - K_t @ H) @ P_t|t-1
        sigma = (jnp.eye(latent_dims) - K_t @ H) @ z_t_given_t_sub_1.covariance()

        return MultivariateNormalFullCovariance(mu, sigma)


model = SVAE_LDS(latent_dims=latent_dims)
# params = model.init(key, setup_batch, random.PRNGKey(0))
# %%
def kl_divergence(
    q: MultivariateNormalFullCovariance, p: MultivariateNormalFullCovariance
):
    mu_0 = q.mean()
    sigma_0 = q.covariance()

    mu_1 = p.mean()
    sigma_1 = p.covariance()

    k = mu_0.shape[0]

    # \frac{1}{2} (\text{tr}(\Sigma_1^{-1}\Sigma_0) + (\mu_1 - \mu_0)^T \Sigma_1^{-1} (\mu_1-\mu_0)-k+\log(\frac{\det \Sigma_1}{\det \Sigma_0}))
    a = jnp.trace(jnp.linalg.inv(sigma_1) @ sigma_0, axis1=1, axis2=2)
    mean_diff = jnp.expand_dims(mu_1 - mu_0, -1)
    b = (mean_diff.swapaxes(1, 2) @ jnp.linalg.inv(sigma_1) @ mean_diff).squeeze()
    c = jnp.log(jnp.linalg.det(sigma_1) / jnp.linalg.det(sigma_0))
    return 0.5 * (a + b - k + c)


def create_train_step(
    key: jax.random.PRNGKey, model: nn.Module, optimizer: optax.GradientTransformation
):
    params = model.init(key, setup_batch, random.PRNGKey(0))
    opt_state = optimizer.init(params)

    def loss_fn(params, x, key):
        recon, q_dist, p_dist = model.apply(params, x, key)

        mse_loss = optax.l2_loss(recon, x).sum() / x.shape[0]

        def kl_wrapper(_, dists):
            q_z, p_z = dists
            return None, kl_divergence(q_z, p_z)
        _, kl_loss = jax.lax.scan(kl_wrapper, None, (q_dist, p_dist))
        
        kl_loss = jnp.sum(kl_loss) / x.shape[0]

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
running_mse = []
running_kl = []

pbar = tqdm(range(epochs))
for epoch in pbar:
    total_loss, total_mse, total_kl = 0.0, 0.0, 0.0
    for i, batch in enumerate(train_dataloader):
        key, subkey = jax.random.split(key)

        batch = process_batch(batch)
        params, opt_state, loss, mse_loss, kl_loss = train_step(
            params, opt_state, batch, subkey
        )

        total_loss += loss
        total_mse += mse_loss
        total_kl += kl_loss

        pbar.set_postfix_str(f"Loss: {total_loss/i:.2f}")
    running_loss.append(total_loss / len(train_dataloader))
    running_mse.append(total_mse / len(train_dataloader))
    running_kl.append(total_kl / len(train_dataloader))
    mngr.save(epoch, args=ocp.args.StandardSave(params))
mngr.wait_until_finished()
# %%
plt.plot(running_loss, label='Total Loss')
plt.plot(running_mse, label='MSE Loss')
plt.plot(running_kl, label='KL Loss')

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# %%
params = model.init(key, setup_batch, random.PRNGKey(0))

x=setup_batch

recon, q_dist, p_dist = model.apply(params, x, random.PRNGKey(1))

mse_loss = optax.l2_loss(recon, x).sum() / x.shape[0]
print("mse_loss", mse_loss)

kl_loss = 0
for q_z, p_z in zip(q_dist, p_dist):
    kld = kl_divergence(q_z, p_z)
    kl_loss += kld

kl_loss = jnp.sum(kl_loss) / x.shape[0]
print("kl_loss", kl_loss)

loss = mse_loss + kl_weight * kl_loss
# %%
