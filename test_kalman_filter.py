# %%
from typing import Tuple
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as random

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from lib.distributions import MultivariateNormalFullCovariance
from lib.priors import KalmanFilter

jax.config.update("jax_enable_x64", True)
# %% [markdown]
# Example from: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LinearGaussianStateSpaceModel#examples
#
# Consider a simple tracking model, in which a two-dimensional latent state represents the position of a vehicle,
# and at each timestep we see a noisy observation of this position (e.g., a GPS reading).
# The vehicle is assumed to move by a random walk with standard deviation step_std at each step,
# and observation noise level std. We build the marginal distribution over noisy observations as a state space model:
# %%
timesteps = 1
ndims = 2
step_std = 1.0
noise_std = 2.0

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
    num_timesteps=timesteps,
    transition_matrix=transition_matrix,
    transition_noise=transition_noise,
    observation_matrix=observation_matrix,
    observation_noise=observation_noise,
    initial_state_prior=initial_state_prior,
)

# From: https://github.com/zziz/kalman-filter
class KalmanFilter2(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = jnp.eye(self.n) if Q is None else Q
        self.R = jnp.eye(self.n) if R is None else R
        self.P = jnp.eye(self.n) if P is None else P
        self.x = jnp.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = jnp.dot(self.F, self.x) #+ jnp.dot(self.B, u)
        self.P = jnp.dot(jnp.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - jnp.dot(self.H, self.x)
        S = self.R + jnp.dot(self.H, jnp.dot(self.P, self.H.T))
        K = jnp.dot(jnp.dot(self.P, self.H.T), jnp.linalg.inv(S))
        self.x = self.x + jnp.dot(K, y)
        I = jnp.eye(self.n)
        self.P = jnp.dot(jnp.dot(I - jnp.dot(K, self.H), self.P), 
        	(I - jnp.dot(K, self.H)).T) + jnp.dot(jnp.dot(K, self.R), K.T)

kf = KalmanFilter2(F=transition_matrix, B=None, H=observation_matrix, Q=transition_noise.covariance(), R=observation_noise.covariance(), P=initial_state_prior.covariance(), x0=initial_state_prior.mean())
# %%
key = random.PRNGKey(42)
key, tmpkey = random.split(key)
x = model.sample(1, tmpkey)  # Sample from the prior on sequences of observations.

# x = jnp.expand_dims(jnp.vstack((jnp.linspace(0, 1000, timesteps),)*2).T, 0)

model.log_prob(x[0])
# %%
# Compute the filtered posterior on latent states given observations,
# and extract the mean and covariance for the current (final) timestep.
_, filtered_means, filtered_covs, predicted_means, predicted_covs, _, _ = model.forward_filter(x)
# current_location_posterior = tfd.MultivariateNormalTriL(
#     loc=filtered_means[..., -1, :],
#     scale_tril=jnp.linalg.cholesky(filtered_covs[..., -1, :, :]),
# )

# Run a smoothing recursion to extract posterior marginals for locations
# at previous timesteps.
posterior_means, posterior_covs = model.posterior_marginals(x)
initial_location_posterior = tfd.MultivariateNormalTriL(
    loc=posterior_means[..., 0, :],
    scale_tril=jnp.linalg.cholesky(posterior_covs[..., 0, :, :]),
)
# %%
predictions = []
for z in x:
    kf.predict()
    print(kf.x, kf.P)
    kf.update(z[0])
    print(kf.x, kf.P)
# %%
# Predict Step
mu_t = transition_matrix @ jnp.array([0,0])
sigma_t = transition_matrix @ initial_state_prior.covariance() @ transition_matrix.T + transition_noise.covariance()
print("Predict")
print(mu_t, sigma_t)
print(predicted_means[0], predicted_covs[0])

hat_x_t = mu_t
S_t = sigma_t + observation_noise.covariance()
K_t = sigma_t @ jnp.linalg.inv(S_t)

mu = mu_t + K_t @ (x[0,0] - hat_x_t)
sigma = sigma_t - K_t @ S_t @ K_t
print("Update")
print(mu, sigma)
print(filtered_means[0], filtered_covs[0])
# %%
z = jax.vmap(lambda x: MultivariateNormalFullCovariance(x, observation_noise.covariance()))(x[0])

key, tmpkey = random.split(key)
prior = MultivariateNormalFullCovariance(initial_state_prior.mean(), initial_state_prior.covariance())
our_filtered_dists, our_predicted_dists = KalmanFilter.run_forward(z, prior, transition_matrix, jnp.zeros((ndims)), transition_noise.covariance(), observation_matrix)
our_filtered_means = our_filtered_dists.mean()
our_predicted_means = our_predicted_dists.mean()
_, our_posterior_dists = KalmanFilter.run_backward(our_filtered_dists, tmpkey, transition_matrix, jnp.zeros((ndims)), transition_noise.covariance(), observation_matrix)
our_posterior_means = our_posterior_dists.mean()
# %%
fig, axs = plt.subplots(1, 3, figsize=(10,5))

axs[0].plot(x[0, :, 0], x[0, :, 1], label="Samples", marker='.')
axs[0].plot(filtered_means[0, :, 0], filtered_means[0, :, 1], label="TFP Filtered Means", marker='.')
axs[0].plot(our_filtered_means[:, 0], our_filtered_means[:, 1], label="Our Filtered Means", marker='.')
axs[0].plot(posterior_means[0, :, 0], posterior_means[0, :, 1], label="TFP Smoothed Means", marker='.')
axs[0].legend()
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")

axs[1].plot(filtered_means[0, :, 0], filtered_means[0, :, 1], label="TFP Filtered Means", marker='.', c='tab:blue', linewidth=3)
axs[1].scatter(filtered_means[0, 0, 0], filtered_means[0, 0, 1], c='tab:blue')
axs[1].plot(our_filtered_means[:, 0], our_filtered_means[:, 1], label="Our Filtered Means", marker='.', c='tab:orange', linewidth=3)
axs[1].scatter(our_filtered_means[0, 0], our_filtered_means[0, 1], c='tab:orange')

axs[1].plot(predicted_means[0, :, 0], predicted_means[0, :, 1], label="TFP Predicted Means", marker='.', c='tab:green')
# axs[1].scatter(predicted_means[0, 0, 0], predicted_means[0, 0, 1], c='tab:green')
axs[1].plot(our_predicted_means[:, 0], our_predicted_means[:, 1], label="Our Predicted Means", marker='.', c='tab:red')
# axs[1].scatter(our_predicted_means[0, 0], our_predicted_means[0, 1], c='tab:red')

axs[1].legend()
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")

axs[2].plot(posterior_means[0, :, 0], posterior_means[0, :, 1], label="TFP Smoothed Means", marker='.', c='tab:blue')
axs[2].scatter(posterior_means[0, 0, 0], posterior_means[0, 0, 1], c='tab:blue')
axs[2].plot(our_posterior_means[:, 0], our_posterior_means[:, 1], label="Our Smoothed Means", marker='.', c='tab:orange')
axs[2].scatter(our_posterior_means[0, 0], our_posterior_means[0, 1], c='tab:orange')
axs[2].legend()
axs[2].set_xlabel("x")
axs[2].set_ylabel("y")

plt.show()
# %%
latent_dims = ndims
z_hat = z
q_dist = our_posterior_dists
p_dist = our_predicted_dists
# %%
def kl_divergence(
    q: MultivariateNormalFullCovariance, p: MultivariateNormalFullCovariance
):
    mu_0 = q.mean()
    sigma_0 = q.covariance()

    mu_1 = p.mean()
    sigma_1 = p.covariance()

    k = mu_0.shape[-1]

    # \frac{1}{2} (\text{tr}(\Sigma_1^{-1}\Sigma_0) + (\mu_1 - \mu_0)^T \Sigma_1^{-1} (\mu_1-\mu_0)-k+\log(\frac{\det \Sigma_1}{\det \Sigma_0}))
    a = jnp.trace(jnp.linalg.inv(sigma_1) @ sigma_0)
    mean_diff = mu_1 - mu_0
    b = mean_diff.T @ jnp.linalg.inv(sigma_1) @ mean_diff
    # print(f"Working KL: {b-k}")
    c = jnp.log(jnp.linalg.det(sigma_1) / jnp.linalg.det(sigma_0))
    return 0.5 * (a + b - k + c)

def observation_likelihood(z_hat: MultivariateNormalFullCovariance, q_z: MultivariateNormalFullCovariance, p_z: MultivariateNormalFullCovariance):
    k = z_hat.mean().shape[-1]
    # -1/2 ( k*log(2pi) + log(det(Sigma_i)) + (x_i - mu_i)^T @ Sigma_i^-1 @ (x_i - mu_i) + tr(P_i Sigma_i^-1) )
    mean_diff = z_hat.mean() - q_z.mean()
    inv_cov = jnp.linalg.inv(z_hat.covariance())
    return -(1/2) * (k * jnp.log(2*jnp.pi) + jnp.log(jnp.linalg.det(z_hat.covariance())) + mean_diff.T @ inv_cov @ mean_diff + jnp.linalg.trace(q_z.covariance() @ inv_cov))
 
# The first term has a different equation the next ones because p(z_1) is known in closed form
# -1/2 * (k*log(2pi) + log(det(Sigma_i))) - log(p(x))
z_hat1 = MultivariateNormalFullCovariance(z_hat.mean()[0], z_hat.covariance()[0])
p_z1 = MultivariateNormalFullCovariance(jnp.zeros((latent_dims)), jnp.eye(latent_dims))
q_z1 = MultivariateNormalFullCovariance(q_dist.mean()[0], q_dist.covariance()[0])
kl_loss_0 = observation_likelihood(z_hat1.multiply(p_z1)[1], q_z1, p_z1) - q_z1.multiply(p_z1)[0]

# Correct KL Divergence
kl_divergence(q_z1, p_z1)

# Calculate the rest of the terms
# def kl_wrapper(q_z_sub_1: MultivariateNormalFullCovariance, dists: Tuple[MultivariateNormalFullCovariance, MultivariateNormalFullCovariance, MultivariateNormalFullCovariance]):
#     z_hat, q_z, p_z = dists

#     # p(z_i|x_{1:i-1}) = \int p(z_i|z_{1:i-1}) p(z_{i-1}|x_{1:i-1}) dz_{i-1}
#     p_zi_given_x1toisub1 = p_z.multiply(q_z_sub_1)

#     # p(x_i) = \int p(z_i|x_{1:i-1}) p(x_i|z_i) dz_i
#     log_p_x = p_zi_given_x1toisub1[0] + q_z.multiply(p_zi_given_x1toisub1[1])[0]
#     kl = observation_likelihood(z_hat, q_z, p_z) - log_p_x

#     return q_z, kl

# _, kl_loss_after0 = jax.lax.scan(kl_wrapper,
#     q_z1,
#     (
#         MultivariateNormalFullCovariance(z_hat.mean()[1:], z_hat.covariance()[1:]),
#         MultivariateNormalFullCovariance(q_dist.mean()[1:], q_dist.covariance()[1:]),
#         MultivariateNormalFullCovariance(p_dist.mean()[1:], p_dist.covariance()[1:])
#     )
# )
# kl_loss = jnp.append(jnp.array(kl_loss_0), kl_loss_after0)

# kl_loss
# %%
post = p_z1.multiply(z_hat1)[1]
kl_divergence(post, p_z1)
# %%
observation_likelihood(z_hat1, post, p_z1) - z_hat1.multiply(p_z1)[0]
# %%
