# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

import jax
import jax.numpy as jnp
import jax.random as jnr
from jax import Array

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import torch

from lib.priors import KalmanFilter, KalmanFilter_MOTPDA, KalmanFilter_MOTCAVI
from lib.distributions import MVN_kl_divergence, GMM_moment_match, MVN_multiply, MVN_Type, MVN_inverse_bayes

from pykalman import KalmanFilter as PyKalmanFilter
from pykalman.standard import _smooth, _smooth_pair

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
num_balls = 2

noise_amt = 0.15
key, tmpkey = jnr.split(key)
train = jnp.concat((train[:, :, :1], train[:, :, :1]), axis=2)
train += jnr.normal(tmpkey, train.shape) * noise_amt
key, tmpkey = jnr.split(key)
test = jnp.concat((test[:, :, :1], test[:, :, :1]), axis=2)
test += jnr.normal(tmpkey, test.shape) * noise_amt

train = train/5-1
test = test/5-1

train_dataloader = torch.utils.data.DataLoader(torch.tensor(np.asarray(train)), batch_size=batch_size, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(torch.tensor(np.asarray(test)), batch_size=batch_size, shuffle=False)

process_batch = lambda x: jnp.array(x, dtype='float64')
setup_batch = process_batch(next(iter(train_dataloader)))

i = 0
for j in range(num_balls):
    plt.scatter(setup_batch[i,:,j, -pos_dims], setup_batch[i,:,j, -pos_dims+1])
plt.xlim(-1, 1)
plt.ylim(-1, 1)
# %% [markdown]
# manually entered kalman filter for object tracking
# %%
A = jnp.array([[1, 0, 0.1, 0], [0, 1, 0, 0.1], [0,0,1,0], [0,0,0,1]])
b = jnp.zeros((4))
Q = jnp.eye(4) * 0.1
R = jnp.eye(2) * 0.1
H = jnp.array([[1,0,0,0],[0,1,0,0]])

x = (setup_batch[0,:,0,:num_balls], jnp.broadcast_to(R, (50, 2,2)))
z_t_sub_1 = (jnp.append(x[0][0], jnp.zeros(2)), Q)
f_dist, p_dist, log_lik = KalmanFilter.run_forward((x[0][1:], x[1][1:]), z_t_sub_1, A, b, Q, H, jnp.zeros(49))
q_dist, _ = KalmanFilter.run_backward(f_dist, A, b, Q, H)

pkf = PyKalmanFilter(A, H, Q, R, b, jnp.zeros((2,)), z_t_sub_1[0], z_t_sub_1[1],
    em_vars=[
        "transition_matrices",
        "observation_matrices",
        "transition_covariance",
        "observation_covariance",
        "observation_offsets",
        "initial_state_mean",
        "initial_state_covariance",
    ],
)
pkf_q_dist = pkf.smooth(x[0][1:])

plt.scatter(*x[0].T, color='black')
plt.plot(*(H@q_dist[0].T))
plt.plot(*(H@pkf_q_dist[0].T), '--', linewidth=1)
A, b, Q, R, H
# %% [markdown]
# use EM to optimize parameters
# %%
vec_to_cov_cholesky = tfb.Chain([
    tfb.CholeskyOuterProduct(),
    tfb.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=None)
])

key, tmpkey = jnr.split(key)
# A = (jnp.eye(4) + jnr.normal(key, (4,4)) * 0.5).round(2)
# b = jnp.zeros((4))
# key, tmpkey = jnr.split(key)
Q = vec_to_cov_cholesky.forward(jnr.normal(key, int(latent_dims*(latent_dims+1)/2)) * 0.1).round(2)
# key, tmpkey = jnr.split(key)
# R = vec_to_cov_cholesky.forward(jnr.normal(key, ((int(pos_dims*(pos_dims+1)/2)))) * 0.1).round(2)
# key, tmpkey = jnr.split(key)
# H = (jnp.array([[1,0,0,0],[0,1,0,0]]) + jnr.normal(key, (2,4)) * 0.5).round(2)

f_dist, p_dist, log_lik = KalmanFilter.run_forward((x[0][1:], x[1][1:]), z_t_sub_1, A, b, Q, H, jnp.zeros(49))
q_dist, J_t = KalmanFilter.run_backward(f_dist, A, b, Q, H)

pkf = PyKalmanFilter(A, H, Q, R, b, jnp.zeros((2,)), z_t_sub_1[0], z_t_sub_1[1],
    em_vars=[
        "transition_matrices",
        "observation_matrices",
        "transition_covariance",
        "observation_covariance",
        "observation_offsets",
        "initial_state_mean",
        "initial_state_covariance",
    ],
)
pkf_q_dist = pkf.smooth(x[0][1:])

plt.scatter(*x[0].T, color='black')
plt.plot(*(H@q_dist[0].T))
plt.plot(*(H@pkf_q_dist[0].T), '--', linewidth=1)
A, b, Q, R, H
# %%
pkf_cross_cov = _smooth_pair(q_dist[1], J_t)[1:]
cross_cov = KalmanFilter.cross_covariance(q_dist, J_t)
jnp.allclose(pkf_cross_cov, cross_cov)
# %%
# @jax.jit
# def single_iter(carry, i):
#     A, b, Q, R, H = carry

#     f_dist, p_dist, log_lik = KalmanFilter.run_forward((x[0][1:], x[1][1:]), z_t_sub_1, A, b, Q, H, jnp.zeros(49))
#     q_dist, J_t = KalmanFilter.run_backward(f_dist, A, b, Q, H)

#     # H, R, A, Q, _ = KalmanFilter.m_step_update((x[0][1:], x[1][1:]), z_t_sub_1, p_dist, f_dist, q_dist, A, Q, H, R)
#     _, _, A, _, _ = KalmanFilter.m_step_update((x[0][1:], x[1][1:]), z_t_sub_1, p_dist, f_dist, q_dist, J_t, A, Q, H, R)

#     return (A, b, Q, R, H), i
# (A, b, Q, R, H), _ = jax.lax.scan(single_iter, (A, b, Q, R, H), jnp.arange(1000))

for i in range(1):
    pkf = pkf.em(X=x[0][1:], n_iter=1, em_vars=["transition_covariance"])
    print(pkf.transition_covariance)
    pkf_q_dist = pkf.smooth(x[0][1:])

    f_dist, p_dist, log_lik = KalmanFilter.run_forward((x[0][1:], x[1][1:]), z_t_sub_1, A, b, Q, H, jnp.zeros(49))
    q_dist, J_t = KalmanFilter.run_backward(f_dist, A, b, Q, H)

    # H, R, A, Q, _ = KalmanFilter.m_step_update((x[0][1:], x[1][1:]), z_t_sub_1, p_dist, f_dist, q_dist, J_t, A, Q, H, R)
    _, _, _, Q, _ = KalmanFilter.m_step_update((x[0][1:], x[1][1:]), z_t_sub_1, p_dist, f_dist, q_dist, J_t, A, Q, H, R)
    print(Q)

    # print(jnp.any(jnp.isnan(jnp.linalg.cholesky(Q))))


plt.scatter(*x[0].T, color='black')
plt.plot(*(H@q_dist[0].T))
plt.plot(*(H@pkf_q_dist[0].T), '--', linewidth=1)
A, b, Q, R, H
# %%
print("q_dist cov trace", jnp.trace(q_dist[1], axis1=1, axis2=2))
print("p_dist cov trace", jnp.trace(p_dist[1], axis1=1, axis2=2))

print(jnp.allclose(p_dist[1], jnp.moveaxis(p_dist[1], -1, -2)))
print(jnp.allclose(q_dist[1], jnp.moveaxis(q_dist[1], -1, -2)))
# %%
