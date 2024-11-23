# %%
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as random

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from lib.distributions import MultivariateNormalFullCovariance

key = random.PRNGKey(42)
# %% [markdown]
# Example from: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LinearGaussianStateSpaceModel#examples
#
# Consider a simple tracking model, in which a two-dimensional latent state represents the position of a vehicle,
# and at each timestep we see a noisy observation of this position (e.g., a GPS reading).
# The vehicle is assumed to move by a random walk with standard deviation step_std at each step,
# and observation noise level std. We build the marginal distribution over noisy observations as a state space model:
# %%
ndims = 2
step_std = 1.0
noise_std = 5.0

transition_matrix = jnp.eye(ndims)
transition_noise = tfd.MultivariateNormalDiag(
    scale_diag=step_std**2 * jnp.ones([ndims])
)
observation_matrix = jnp.eye(ndims)
observation_noise = tfd.MultivariateNormalDiag(
    scale_diag=noise_std**2 * jnp.ones([ndims])
)
initial_state_prior = tfd.MultivariateNormalDiag(scale_diag=jnp.ones([ndims]))

model = tfd.LinearGaussianStateSpaceModel(
    num_timesteps=100,
    transition_matrix=transition_matrix,
    transition_noise=transition_noise,
    observation_matrix=observation_matrix,
    observation_noise=observation_noise,
    initial_state_prior=initial_state_prior,
)
# %%
latent_dims = ndims

class KalmanFilter:
    def __call__(self, z, prior, z_rng, A, b, Q, H):
        z_t_sub_1 = prior

        def kf_forward(carry, z_t):
            z_rng, z_t_sub_1 = carry

            # Prediction
            z_t_given_t_sub_1 = self.predict(z_t_sub_1, A, b, Q)

            # Update
            z_t_given_t = self.update(z_t_given_t_sub_1, z_t, H)

            # Sample and decode
            z_rng, z_t_rng = random.split(z_rng)
            z_hat = z_t_given_t.sample(z_t_rng)

            # jax.debug.print("z_t_given_t_sub_1: {z_t_given_t_sub_1}", z_t_given_t_sub_1=z_t_given_t_sub_1)
            # jax.debug.print("z_t_given_t: {z_t_given_t}", z_t_given_t=z_t_given_t)

            return (z_rng, z_t_given_t), (z_hat, z_t_given_t, z_t_given_t_sub_1) # carry, (z_recon, q_dist, p_dist)
        
        _, result = jax.lax.scan(kf_forward, (z_rng, z_t_sub_1), z)
        z_recon, q_dist, p_dist = result

        return z_recon, q_dist, p_dist

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
        self, z_t_given_t_sub_1: MultivariateNormalFullCovariance, x_t: jnp.array, H: jnp.array
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

        # K_t = P_t|t-1 @ H^T @ (H @ P_t|t-1 @ H^T + R) ^ -1
        K_t = z_t_given_t_sub_1.covariance() @ H.T @ jnp.linalg.inv(H @ z_t_given_t_sub_1.covariance() @ H.T + x_t.covariance())

        # z_t|t = z_t|t-1 + K_t @ (x_t - H @ z_t|t-1)
        # Extra expand_dims and squeeze are necessary to make the matmul dimensions work
        mu = z_t_given_t_sub_1.mean() + K_t @ (x_t.mean() - H @ z_t_given_t_sub_1.mean())

        # P_t|t = P_t|t-1 - K_t @ H @ P_t|t-1 = (I - K_t @ H) @ P_t|t-1
        sigma = (jnp.eye(latent_dims) - K_t @ H) @ z_t_given_t_sub_1.covariance()

        return MultivariateNormalFullCovariance(mu, sigma)
# %%
key, tmpkey = random.split(key)
x = model.sample(1, tmpkey)  # Sample from the prior on sequences of observations.
# %%
# Compute the filtered posterior on latent states given observations,
# and extract the mean and covariance for the current (final) timestep.
_, filtered_means, filtered_covs, _, _, _, _ = model.forward_filter(x)
current_location_posterior = tfd.MultivariateNormalTriL(
    loc=filtered_means[..., -1, :],
    scale_tril=jnp.linalg.cholesky(filtered_covs[..., -1, :, :]),
)

# Run a smoothing recursion to extract posterior marginals for locations
# at previous timesteps.
posterior_means, posterior_covs = model.posterior_marginals(x)
initial_location_posterior = tfd.MultivariateNormalTriL(
    loc=posterior_means[..., 0, :],
    scale_tril=jnp.linalg.cholesky(posterior_covs[..., 0, :, :]),
)
# %%
kf = KalmanFilter()

z = jax.vmap(lambda x: MultivariateNormalFullCovariance(x, observation_noise.covariance()))(x[0])

key, tmpkey = random.split(key)
prior = MultivariateNormalFullCovariance(initial_state_prior.mean(), initial_state_prior.covariance())
_, _, our_filtered_dists = kf(z, prior, tmpkey, transition_matrix, jnp.zeros((latent_dims)), transition_noise.covariance(), observation_matrix)
our_filtered_means = our_filtered_dists.mean()
# %%
fig, axs = plt.subplots(1, 2, figsize=(10,5))

axs[0].plot(x[0, :, 0], x[0, :, 1], label="Samples")
axs[0].plot(filtered_means[0, :, 0], filtered_means[0, :, 1], label="TFP Filtered Means")
axs[0].plot(posterior_means[0, :, 0], posterior_means[0, :, 1], label="TFP Smoothed Means")
axs[0].plot(our_filtered_means[:, 0], our_filtered_means[:, 1], label="Our Smoothed Means")
axs[0].legend()
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")

axs[1].plot(filtered_means[0, :, 0], filtered_means[0, :, 1], label="TFP Filtered Means")
axs[1].plot(our_filtered_means[:, 0], our_filtered_means[:, 1], label="Our Smoothed Means")
axs[1].legend()
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")

axs[1].plot()
plt.show()
# %%
