# %%
from typing import Tuple
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jnr

from tensorflow_probability.substrates import jax as tfp

from dynamax.utils.plotting import plot_uncertainty_ellipses
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.linear_gaussian_ssm import lgssm_smoother, lgssm_filter

from lib.distributions import MultivariateNormalFullCovariance
from lib.priors import KalmanFilter

jax.config.update("jax_enable_x64", True)
# %% [markdown]
# dynamax code modified from the following example: https://probml.github.io/dynamax/notebooks/linear_gaussian_ssm/kf_tracking.html#
# %%
timesteps = 10
ndims = 2

step_std = 0.1
noise_std = 0.1

theta = -0.1
transition_matrix = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],[jnp.sin(theta), jnp.cos(theta)]]) #jnp.eye(ndims)
transition_noise = jnp.eye(ndims) * step_std
observation_matrix = jnp.eye(ndims)
observation_noise = jnp.eye(ndims) * noise_std
initial_mean = jnp.ones(ndims) * 5
initial_covariance = jnp.eye(ndims)

lgssm = LinearGaussianSSM(ndims, ndims)
params, _ = lgssm.initialize(jnr.PRNGKey(0),
                             initial_mean=initial_mean,
                             initial_covariance=initial_covariance,
                             dynamics_weights=transition_matrix,
                             dynamics_covariance=transition_noise,
                             emission_weights=observation_matrix,
                             emission_covariance=observation_noise)
# %%
key = jnr.PRNGKey(42)
key, tmpkey = jnr.split(key)
z_, x = lgssm.sample(params, key, timesteps)

# Plot Data
observation_marker_kwargs = {"marker": "o", "markerfacecolor": "none", "markeredgewidth": 2, "markersize": 8}
fig1, ax1 = plt.subplots()
ax1.plot(*z_[:, :2].T, marker="s", color="C0", label="true state")
ax1.plot(*x.T, ls="", **observation_marker_kwargs, color="tab:green", label="emissions")
ax1.legend(loc="upper left")
ax1.axis("equal")
# %%
lgssm_posterior = lgssm.filter(params, x)
filtered_means = lgssm_posterior.filtered_means
filtered_covs = lgssm_posterior.filtered_covariances
print(lgssm_posterior.marginal_loglik)
# %%
lgssm_posterior = lgssm.smoother(params, x)
posterior_means = lgssm_posterior.smoothed_means
posterior_covs = lgssm_posterior.smoothed_covariances
# %%
z = jax.vmap(lambda x: MultivariateNormalFullCovariance(x, observation_noise))(x)

key, tmpkey = jnr.split(key)
prior = MultivariateNormalFullCovariance(initial_mean, initial_covariance)
our_filtered_dists, our_predicted_dists = KalmanFilter.run_forward(z, prior, transition_matrix, jnp.zeros((ndims)), transition_noise, observation_matrix)
our_filtered_means = our_filtered_dists.mean()
our_filtered_covs = our_filtered_dists.covariance()
our_predicted_means = our_predicted_dists.mean()
our_predicted_covs = our_predicted_dists.covariance()
_, our_posterior_dists = KalmanFilter.run_backward(our_filtered_dists, tmpkey, transition_matrix, jnp.zeros((ndims)), transition_noise, observation_matrix)
our_posterior_means = our_posterior_dists.mean()
our_posterior_covs = our_posterior_dists.covariance()
# %%
fig, ax = plt.subplots()
ax.plot(*x.T, ls="", **observation_marker_kwargs, color="tab:green", label="observed")
ax.plot(*z_[:, :2].T, ls="--", color="darkgrey", label="true state")

ax.plot(filtered_means[:, 0], filtered_means[:, 1], color="tab:red", label="dynamax", linewidth=4)
plot_uncertainty_ellipses(filtered_means, filtered_covs, ax, **{"edgecolor": "tab:red", "linewidth": 0.5})

ax.plot(our_filtered_means[:, 0], our_filtered_means[:, 1], color="tab:blue", label="ours")
plot_uncertainty_ellipses(our_filtered_means, our_filtered_covs, ax, **{"edgecolor": "tab:blue", "linewidth": 0.5})

ax.plot(our_predicted_means[:, 0], our_predicted_means[:, 1], color="tab:orange", label="ours predicted")
plot_uncertainty_ellipses(our_predicted_means, our_predicted_covs, ax, **{"edgecolor": "tab:orange", "linewidth": 0.5})


ax.axis("equal")
ax.legend(loc="upper left")
ax.set_title("Filtered Posterior Comparison")
# %%
fig, ax = plt.subplots()
ax.plot(*x.T, ls="", **observation_marker_kwargs, color="tab:green", label="observed")
ax.plot(*z_[:, :2].T, ls="--", color="darkgrey", label="true state")

ax.plot(posterior_means[:, 0], posterior_means[:, 1], color="tab:red", label="dynamax", linewidth=4)
plot_uncertainty_ellipses(posterior_means, posterior_covs, ax, **{"edgecolor": "tab:red", "linewidth": 0.5})

ax.plot(our_posterior_means[:, 0], our_posterior_means[:, 1], color="tab:blue", label="ours")
plot_uncertainty_ellipses(our_posterior_means, our_posterior_covs, ax, **{"edgecolor": "tab:blue", "linewidth": 0.5})

ax.axis("equal")
ax.legend(loc="upper left")
ax.set_title("Smoothed Posterior Comparison")
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
kl_loss_0 = observation_likelihood(z_hat1, q_z1, p_z1) - z_hat1.multiply(p_z1)[0]
print(kl_loss_0)

# Correct KL Divergence
print(kl_divergence(q_z1, p_z1))

# Calculate the rest of the terms
def kl_wrapper(q_z_sub_1: MultivariateNormalFullCovariance, dists: Tuple[MultivariateNormalFullCovariance, MultivariateNormalFullCovariance, MultivariateNormalFullCovariance]):
    z_hat, q_z, p_z = dists

    # p(z_i|x_{1:i-1}) = \int p(z_i|z_{1:i-1}) p(z_{i-1}|x_{1:i-1}) dz_{i-1}
    p_zi_given_x1toisub1 = p_z.multiply(q_z_sub_1)

    # p(x_i) = \int p(z_i|x_{1:i-1}) p(x_i|z_i) dz_i
    log_p_x = p_zi_given_x1toisub1[0] + z_hat.multiply(p_zi_given_x1toisub1[1])[0]
    kl = observation_likelihood(z_hat, q_z, p_z) - log_p_x

    return q_z, kl

_, kl_loss_after0 = jax.lax.scan(kl_wrapper,
    q_z1,
    (
        MultivariateNormalFullCovariance(z_hat.mean()[1:], z_hat.covariance()[1:]),
        MultivariateNormalFullCovariance(q_dist.mean()[1:], q_dist.covariance()[1:]),
        MultivariateNormalFullCovariance(p_dist.mean()[1:], p_dist.covariance()[1:])
    )
)
kl_loss = jnp.append(jnp.array(kl_loss_0), kl_loss_after0)

kl_loss
# %%
post = p_z1.multiply(z_hat1)[1]
kl_divergence(post, p_z1)
# %%
observation_likelihood(z_hat1, post, p_z1) - z_hat1.multiply(p_z1)[0]
# %%