# %%
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as random

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from lib.distributions import MultivariateNormalFullCovariance
from lib.priors import KalmanFilter

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

z = jax.vmap(lambda x: MultivariateNormalFullCovariance(x, observation_noise.covariance()))(x[0])

key, tmpkey = random.split(key)
prior = MultivariateNormalFullCovariance(initial_state_prior.mean(), initial_state_prior.covariance())
_, _, our_filtered_dists = KalmanFilter.run(z, prior, tmpkey, transition_matrix, jnp.zeros((ndims)), transition_noise.covariance(), observation_matrix)
our_filtered_means = our_filtered_dists.mean()
# %%
fig, axs = plt.subplots(1, 2, figsize=(10,5))

axs[0].plot(x[0, :, 0], x[0, :, 1], label="Samples")
axs[0].plot(filtered_means[0, :, 0], filtered_means[0, :, 1], label="TFP Filtered Means")
axs[0].plot(our_filtered_means[:, 0], our_filtered_means[:, 1], label="Our Smoothed Means")
axs[0].plot(posterior_means[0, :, 0], posterior_means[0, :, 1], label="TFP Smoothed Means")
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
