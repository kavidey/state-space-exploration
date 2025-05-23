# %%
from typing import Tuple
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

import jax
import jax.numpy as jnp
import jax.random as jnr
from jax import Array

from tensorflow_probability.substrates import jax as tfp

from dynamax.utils.plotting import plot_uncertainty_ellipses
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.linear_gaussian_ssm import lgssm_smoother, lgssm_filter
from lib.priors import KalmanFilter
from lib.distributions import MVN_kl_divergence, GMM_moment_match, MVN_multiply, MVN_Type

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
initial_covariance = jnp.eye(ndims) * 0.1

lgssm = LinearGaussianSSM(ndims, ndims)
params, _ = lgssm.initialize(jnr.PRNGKey(0),
                             initial_mean=initial_mean,
                             initial_covariance=initial_covariance,
                             dynamics_weights=transition_matrix,
                             dynamics_covariance=transition_noise,
                             emission_weights=observation_matrix,
                             emission_covariance=observation_noise)
# %%
key = jnr.PRNGKey(1)
key, tmpkey = jnr.split(key)
z_, x = lgssm.sample(params, tmpkey, timesteps)
key, tmpkey = jnr.split(key)
zp_, xp = lgssm.sample(params, tmpkey, timesteps)
# rotate the other trajectory 90 deg and shift it so they overlap
xp = jnp.flip(xp, axis=1)
xp = xp.at[:, 0].set(xp[:, 0] + 4)
xp = xp.at[:, 1].set(xp[:, 1] - 4)
zp_ = jnp.flip(zp_, axis=1)
zp_ = zp_.at[:, 0].set(zp_[:, 0] + 4)
zp_ = zp_.at[:, 1].set(zp_[:, 1] - 4)

# Plot Data
observation_marker_kwargs = {"marker": "o", "markerfacecolor": "none", "markeredgewidth": 2, "markersize": 8}
fig1, ax1 = plt.subplots()
ax1.plot(*z_[:, :2].T, marker="s", color="tab:blue", label="true state (obj1)")
ax1.plot(*x.T, ls="", **observation_marker_kwargs, color="tab:green", label="emissions (obj1)")

ax1.plot(*zp_[:, :2].T, marker="s", color="tab:orange", label="true state (obj2)")
ax1.plot(*xp.T, ls="", **observation_marker_kwargs, color="tab:red", label="emissions (obj2)")

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
z = jax.vmap(lambda x: (x, observation_noise))(x)

prior = (initial_mean, initial_covariance)
our_filtered_dists, our_predicted_dists, our_log_likelihood = KalmanFilter.run_forward(z, prior, transition_matrix, jnp.zeros((ndims)), transition_noise, observation_matrix, mask=jnp.zeros(timesteps))
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
fig, ax = plt.subplots()
ax.plot(*z_[:, :2].T, marker="s", color="tab:blue", label="true state")
ax.plot(*x.T, ls="", **observation_marker_kwargs, color="tab:green", label="observed")
ax.plot(*xp.T, ls="", **observation_marker_kwargs, color="tab:green")

nobs = 2
zs = (jnp.stack((x, xp), axis=1), jnp.reshape(jnp.repeat(jnp.expand_dims(observation_noise, axis=0), timesteps*nobs, axis=0), (timesteps, nobs, ndims, ndims)))
q_1 = (jnp.array([5., 5.]), initial_covariance)
# q_1 = (jnp.array([9.5, 2.]), initial_covariance)
ax.plot(*q_1[0], marker="s", color="tab:cyan", label="initial state")
ax.legend()

def evaluate_observation(z_t, z_t_given_t_sub_1, H):
    z_t_given_t = KalmanFilter.update(z_t_given_t_sub_1, z_t, H, mask=0)
    
    # This is the same as the log likelihood calculation in KalmanFilter.forward
    z_t_given_t_sub_1_x_space = (H @ z_t_given_t_sub_1[0], H @ z_t_given_t_sub_1[1] @ H.T)
    w_k = jnp.exp(MVN_multiply(*z_t_given_t_sub_1_x_space, *z_t)[0])

    return z_t_given_t, w_k

def kf_mot_forward(carry: MVN_Type, x_t: MVN_Type, A: Array, b: Array, Q: Array, H: Array):
    z_t_sub_1 = carry

    # Prediction
    z_t_given_t_sub_1 = KalmanFilter.predict(z_t_sub_1, A, b, Q)

    # Update
    # find GMM that best represents observations
    z_t_given_t_s, w_ks = jax.vmap(lambda z_t: evaluate_observation(z_t, z_t_given_t_sub_1, observation_matrix))((x_t[0], x_t[1]))
    w_ks = w_ks / w_ks.sum()
    # approximate that with a single moment-matched gaussian
    z_t_given_t = GMM_moment_match(z_t_given_t_s, w_ks)

    # Log-Likelihood
    # project z_{t|t-1} into x (observation) space
    z_t_given_t_sub_1_x_space = (H @ z_t_given_t_sub_1[0], H @ z_t_given_t_sub_1[1] @ H.T)
    # p(x_t) = \int p(z_i|x_{1:i-1}) p(x_i|z_i) dz_i
    log_likelihood = MVN_multiply(*z_t_given_t_sub_1_x_space, *x_t)[0]

    return (z_t_given_t), (z_t_given_t, z_t_given_t_sub_1, log_likelihood) # carry, (q_dist, p_dist, log_likelihood)

kf_forward = lambda carry, x: kf_mot_forward(carry, x, transition_matrix, jnp.zeros(ndims), transition_noise, jnp.eye(ndims))
_, result = jax.lax.scan(kf_forward, q_1, zs)
q_dist, p_dist, log_likelihood = result

ax.scatter(*q_dist[0].T, color="tab:red", label="ours")
plot_uncertainty_ellipses(*q_dist, ax, **{"edgecolor": "tab:red", "linewidth": 0.5})

ax.legend()
# %%
