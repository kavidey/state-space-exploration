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
from lib.priors import KalmanFilter
from lib.distributions import MVN_kl_divergence, GMM_moment_match, MVN_multiply

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

add_cov = jax.vmap(lambda x: (x, observation_noise))
nobs = 2
zs = [add_cov(x), add_cov(xp)]
# s = (jnp.array([5, 5]), initial_covariance)
s = (jnp.array([9.5, 2]), initial_covariance)
ax.plot(*s[0], marker="s", color="tab:cyan", label="initial state")
ax.legend()

for t in range(timesteps):
    z_t_given_t_sub_1 = KalmanFilter.predict(s, transition_matrix, jnp.zeros((ndims)), transition_noise)
    # ax.scatter(*z_t_given_t_sub_1[0], color="tab:purple", label="ours")
    # plot_uncertainty_ellipses(jnp.array([z_t_given_t_sub_1[0]]), jnp.array([z_t_given_t_sub_1[1]]), ax, **{"edgecolor": "tab:purple", "linewidth": 0.5, "label": "observations"})

    w_s = jnp.zeros((nobs))
    z_t_given_t_s = (jnp.zeros((nobs, ndims)), jnp.zeros((nobs, ndims, ndims)))
    
    
    for i in range(nobs):
        z_t = (zs[i][0][t], zs[i][1][t])
        z_t_given_t = KalmanFilter.update(z_t_given_t_sub_1, z_t, observation_matrix, mask=0)
        # this is the same as the log likelihood calculation in KalmanFilter.forward
        z_t_given_t_sub_1_x_space = (observation_matrix @ z_t_given_t_sub_1[0], observation_matrix @ z_t_given_t_sub_1[1] @ observation_matrix.T)
        w_k = jnp.exp(MVN_multiply(*z_t_given_t_sub_1_x_space, *z_t)[0])

        w_s = w_s.at[i].set(w_k)
        z_t_given_t_s = (z_t_given_t_s[0].at[i].set(z_t_given_t[0]), z_t_given_t_s[1].at[i].set(z_t_given_t[1]))

        ax.scatter(*z_t_given_t[0], marker="+", color="grey")
    
    w_s = w_s / w_s.sum()

    z_t_given_t = GMM_moment_match(z_t_given_t_s, w_s)

    s = z_t_given_t

    ax.scatter(*z_t_given_t[0], color="tab:red", label="ours")
    plot_uncertainty_ellipses(jnp.array([z_t_given_t[0]]), jnp.array([z_t_given_t[1]]), ax, **{"edgecolor": "tab:red", "linewidth": 0.5, "label": "observations"})
# ax.legend()
# %%
