# %%
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial

%config InlineBackend.figure_format = 'retina'

import jax
import jax.numpy as jnp
import jax.random as jnr
from jax.tree_util import tree_map
from jax import Array

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import torch

import sys
sys.path.insert(1, 'dynamax') # import dev version of dynamax submodule
from dynamax.utils.plotting import plot_uncertainty_ellipses
from dynamax.linear_gaussian_ssm import LinearGaussianSSM, lgssm_smoother, lgssm_filter, _condition_on
from dynamax.linear_gaussian_ssm import EmissionCAVI

from lib.distributions import MVN_multiply

jax.config.update("jax_enable_x64", True)
# %%
dset_len = 1024
epochs = 100
batch_size = 1024

key = jnr.PRNGKey(43)
# %% Dataset Preparation
torch.manual_seed(47)

dataset_dir = Path("dataset") / "billiard"
train_dset = jnp.load(dataset_dir/"train.npz")
test_dset = jnp.load(dataset_dir/"test.npz")
# %%
# Only use one ball with two observations per timestep
train = jnp.concat([train_dset['y'][:1024, ..., :2]]*2, axis=-1)
test = jnp.concat([test_dset['y'][:1024, ..., :2]]*2, axis=-1)
latent_dims = 4
pos_dims = 2
num_balls = 3

noise_amt = 0.10
key, tmpkey = jnr.split(key)
train = jnp.concat((train[:, :, :1], train[:, :, :1], train[:, :, 2:3]), axis=2)
train += jnr.normal(tmpkey, train.shape) * noise_amt
key, tmpkey = jnr.split(key)
test = jnp.concat((test[:, :, :1], test[:, :, :1], test[:, :, 2:3]), axis=2)
test += jnr.normal(tmpkey, test.shape) * noise_amt

train = train/5-1
test = test/5-1

train_dataloader = torch.utils.data.DataLoader(torch.tensor(np.asarray(train)), batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(torch.tensor(np.asarray(test)), batch_size=batch_size, shuffle=False)

process_batch = lambda x: jnp.array(x, dtype='float64')
setup_batch = process_batch(next(iter(train_dataloader)))

lgssm = LinearGaussianSSM(latent_dims, pos_dims, has_emissions_bias=False, has_dynamics_bias=True)

i = 0
for j in range(num_balls):
    plt.scatter(setup_batch[i,:,j, -pos_dims], setup_batch[i,:,j, -pos_dims+1])
plt.xlim(-1, 1)
plt.ylim(-1, 1)
# %% [markdown]
# CAVI step code
# %%
@jax.jit
def cavi_step(params, raw_emissions, beta, p_d=0.9, g=1, m_t=1):
    """Perform one EM step."""
    emissions = EmissionCAVI(*raw_emissions, beta, p_d, g, m_t)
    lgssm_out = lgssm.smoother(params, emissions)
    lp = lgssm_out.marginal_loglik
    q_dist = (lgssm_out.smoothed_means, lgssm_out.smoothed_covariances)
    beta = emissions.update_beta(q_dist, params.emissions.weights)
    return beta, lp
# %% [markdown]
# manually entered kalman filter for object tracking
# %%
A = jnp.array([[1, 0, 0.1, 0], [0, 1, 0, 0.1], [0,0,1,0], [0,0,0,1]])
b = jnp.zeros((4))
Q = jnp.eye(4) * 0.01
R = jnp.eye(2) * 0.01
H = jnp.array([[1,0,0,0],[0,1,0,0]])

p_d, g, m_t = 0.9, 1, 1
cavi_iters = 5
cavi_lp_thresh = 0.5

x = (setup_batch[0,:,:,:pos_dims], jnp.broadcast_to(R, (50, num_balls, 2, 2)))
raw_emissions = (x[0][1:], x[1][1:])
fig, axs = plt.subplots(2, 1, figsize=(5,10))
axs[0].scatter(*x[0].T, color='black')
colors = ['tab:blue', 'tab:orange', 'tab:red']

# for c,i in zip(colors,range(num_balls)):
i,c = 1, colors[0]

z_0 = (jnp.append(x[0][0][i], jnp.zeros(2)), Q)

params, _ = lgssm.initialize(jnr.PRNGKey(0),
                            initial_mean=z_0[0],
                            initial_covariance=z_0[1],
                            dynamics_weights=A,
                            dynamics_covariance=Q,
                            emission_weights=H,
                            emission_covariance=R)

lps = []
emissions = EmissionCAVI(*raw_emissions, p_d=0.9, g=1.0, m_t=1.0)
beta = emissions.initial_beta()
for _ in range(cavi_iters):
    beta, lp = cavi_step(params, raw_emissions, beta)
    lps.append(lp)
    if len(lps) > 2 and lps[-1] - lps[-2] < cavi_lp_thresh:
        break

emissions = EmissionCAVI(*raw_emissions, beta=beta, p_d=0.9, g=1.0, m_t=1.0)
lgssm_out = lgssm.smoother(params, emissions)
f_dist = (lgssm_out.filtered_means, lgssm_out.filtered_covariances)
log_lik_true = lgssm_out.marginal_loglik
q_dist = (lgssm_out.smoothed_means, lgssm_out.smoothed_covariances)

axs[0].plot(*(H@q_dist[0].T), c=c)
plot_uncertainty_ellipses((H@q_dist[0].T).T, q_dist[1], axs[0], **{"edgecolor": c, "linewidth": 1})

axs[1].plot(lps, c=c)
A, b, Q, R, H
# %% [markdown]
# use EM to optimize parameters
# %%
@partial(jax.jit, static_argnames=['steps'])
def find_beta(raw_emissions, params, p_d, g, m_t, steps=5):
    emissions = EmissionCAVI(*raw_emissions, p_d=p_d, g=g, m_t=m_t)
    beta = emissions.initial_beta()

    return jax.lax.scan(lambda beta, _: cavi_step(params, raw_emissions, beta),
                        beta, xs=None, length=steps)
# %%
vec_to_cov_cholesky = tfb.Chain([
    tfb.CholeskyOuterProduct(),
    tfb.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=None)
])

key, tmpkey = jnr.split(key)
A = (jnp.eye(4) + jnr.normal(key, (4,4)) * 0.1).round(2)
b = jnp.zeros((4))
key, tmpkey = jnr.split(key)
Q = vec_to_cov_cholesky.forward(jnr.normal(key, (int(latent_dims*(latent_dims+1)/2),)) * 0.1).round(2) * 0.01
key, tmpkey = jnr.split(key)
R = vec_to_cov_cholesky.forward(jnr.normal(key, ((int(pos_dims*(pos_dims+1)/2)),)) * 0.1).round(2) * 0.01
key, tmpkey = jnr.split(key)
H = (jnp.array([[1,0,0,0],[0,1,0,0]]) + jnr.normal(key, (2,4)) * 0.05).round(2)

fig, ax = plt.subplots()
ax.scatter(*x[0].T, color='black')
colors = ['tab:blue', 'tab:orange', 'tab:red']

for c,i in zip(colors,range(num_balls)):
    z_0 = (jnp.append(x[0][0][i], jnp.zeros(2)), Q)
    params, _ = lgssm.initialize(jnr.PRNGKey(0),
                                initial_mean=z_0[0],
                                initial_covariance=z_0[1],
                                dynamics_weights=A,
                                dynamics_covariance=Q,
                                emission_weights=H,
                                emission_covariance=R)
    
    beta, lps = find_beta(raw_emissions, params, p_d=p_d, g=g, m_t=m_t, steps=cavi_iters)
    emissions = EmissionCAVI(*raw_emissions, beta=beta, p_d=p_d, g=g, m_t=m_t)

    lgssm_out = lgssm.smoother(params, emissions)
    f_dist = (lgssm_out.filtered_means, lgssm_out.filtered_covariances)
    log_lik = lgssm_out.marginal_loglik
    q_dist = (lgssm_out.smoothed_means, lgssm_out.smoothed_covariances)

    plt.plot(*(H@q_dist[0].T), c=c)
    plt.scatter([z_0[0][0]], [z_0[0][1]], c=c, s=100, alpha=0.5)
    plot_uncertainty_ellipses((H@q_dist[0].T).T, q_dist[1], ax, **{"edgecolor": c, "linewidth": 1})
params
# %%
num_iters = 50

@jax.jit
def em_step(params, m_step_state):
    """Perform one EM step."""
    beta, _ = find_beta(raw_emissions, params, p_d=p_d, g=g, m_t=m_t, steps=cavi_iters)
    emissions = EmissionCAVI(*raw_emissions, beta=beta, p_d=p_d, g=g, m_t=m_t)
    stats, ll = lgssm.e_step(params, emissions)
    batch_stats, lls = tree_map(partial(jnp.expand_dims, axis=0), (stats, ll)) # add fake batch dimension
    lp = lgssm.log_prior(params) + lls.sum()
    params, m_step_state = lgssm.m_step(params, None, batch_stats, m_step_state)
    return params, m_step_state, lp

log_probs = []
log_prob_thresh = 0.5
m_step_state = None
for _ in range(num_iters):
    params, m_step_state, marginal_logprob = em_step(params, m_step_state)
    print(params)
    log_probs.append(marginal_logprob)
    if len(log_probs) > 2 and log_probs[-1] - log_probs[-2] < log_prob_thresh:
        break

lgssm_out = lgssm.smoother(params, emissions)
f_dist = (lgssm_out.filtered_means, lgssm_out.filtered_covariances)
log_lik = lgssm_out.marginal_loglik
q_dist = (lgssm_out.smoothed_means, lgssm_out.smoothed_covariances)

fig, ax = plt.subplots()
ax.scatter(*x[0].T, color='black')
ax.plot(*(H@q_dist[0].T))
plot_uncertainty_ellipses((H@q_dist[0].T).T, q_dist[1], ax, **{"edgecolor": "tab:blue", "linewidth": 1})
params
# %%
plt.plot(log_probs)
plt.xlabel("iteration")
plt.ylabel("marginal log likelihood")
plt.axhline(log_lik_true, color='k', linestyle='--')
# %%
