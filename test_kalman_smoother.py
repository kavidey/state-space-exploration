# %%
from typing import Tuple
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

import jax
import jax.numpy as jnp
import jax.random as jnr

from tensorflow_probability.substrates import jax as tfp

from dynamax.utils.plotting import plot_uncertainty_ellipses
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.linear_gaussian_ssm import lgssm_smoother, lgssm_filter

from lib.distributions import MultivariateNormalFullCovariance, MVN_Type
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
key = jnr.PRNGKey(0)
key, tmpkey = jnr.split(key)
z_, x = lgssm.sample(params, key, timesteps)

# Plot Data
observation_marker_kwargs = {"marker": "o", "markerfacecolor": "none", "markeredgewidth": 2, "markersize": 8}
fig1, ax1 = plt.subplots()
ax1.plot(*z_[:, :2].T, marker="s", color="C0", label="true state")
ax1.plot(*x.T, ls="", **observation_marker_kwargs, color="tab:green", label="emissions")
ax1.legend(loc="upper left")
ax1.axis("equal")

plt.show()
# %%
lgssm_posterior = lgssm.filter(params, x)
filtered_means = lgssm_posterior.filtered_means
filtered_covs = lgssm_posterior.filtered_covariances

predicted_means = jnp.vstack((jnp.expand_dims(initial_mean, axis=0), jax.vmap(lambda m: transition_matrix @ m)(lgssm_posterior.filtered_means[:-1])))
predicted_covs = jnp.vstack((jnp.expand_dims(initial_covariance, axis=0), jax.vmap(lambda cov: (transition_matrix @ cov @ transition_matrix.T) + transition_noise)(lgssm_posterior.filtered_covariances[:-1])))

print(lgssm_posterior.marginal_loglik)
# %%
lgssm_posterior = lgssm.smoother(params, x)
posterior_means = lgssm_posterior.smoothed_means
posterior_covs = lgssm_posterior.smoothed_covariances
# %%
z = jax.vmap(lambda x: MultivariateNormalFullCovariance(x, observation_noise))(x)

prior = MultivariateNormalFullCovariance(initial_mean, initial_covariance)
our_filtered_dists, our_predicted_dists, our_log_likelihood = KalmanFilter.run_forward((z.mean(), z.covariance()), (prior.mean(), prior.covariance()), transition_matrix, jnp.zeros((ndims)), transition_noise, observation_matrix, mask=jnp.zeros(timesteps))
our_filtered_means = our_filtered_dists[0]
our_filtered_covs = our_filtered_dists[1]
our_predicted_means = our_predicted_dists[0]
our_predicted_covs = our_predicted_dists[1]
our_posterior_dists = KalmanFilter.run_backward(our_filtered_dists, transition_matrix, jnp.zeros((ndims)), transition_noise, observation_matrix)
our_posterior_means = our_posterior_dists[0]
our_posterior_covs = our_posterior_dists[1]
# %%
fig, ax = plt.subplots()
ax.plot(*x.T, ls="", **observation_marker_kwargs, color="tab:green", label="observed")
ax.plot(*z_[:, :2].T, ls="--", color="darkgrey", label="true state")

ax.plot(filtered_means[:, 0], filtered_means[:, 1], color="tab:red", label="dynamax", linewidth=4)
plot_uncertainty_ellipses(filtered_means, filtered_covs, ax, **{"edgecolor": "tab:red", "linewidth": 1})

ax.plot(predicted_means[:, 0], predicted_means[:, 1], color="tab:red", linewidth=4)
plot_uncertainty_ellipses(predicted_means, predicted_covs, ax, **{"edgecolor": "tab:red", "linewidth": 1})

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
f_dist = our_filtered_dists
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

def observation_likelihood(z_hat: MVN_Type, q_z: MVN_Type, p_z: MVN_Type):
    k = z_hat[0].shape[-1]
    # -1/2 ( k*log(2pi) + log(det(Sigma_i)) + (x_i - mu_i)^T @ Sigma_i^-1 @ (x_i - mu_i) + tr(P_i Sigma_i^-1) )
    mean_diff = z_hat[0] - q_z[0]
    inv_cov = jnp.linalg.inv(z_hat[1])
    return -(1/2) * (k * jnp.log(2*jnp.pi) + jnp.log(jnp.linalg.det(z_hat[1])) + mean_diff.T @ inv_cov @ mean_diff + jnp.linalg.trace(q_z[1] @ inv_cov))

kl_loss = jax.vmap(observation_likelihood)((z_hat.mean(), z_hat.covariance()), q_dist, p_dist) - our_log_likelihood

kl_loss
# %%
print(lgssm_posterior.marginal_loglik)
print(our_log_likelihood.sum())
# %%
