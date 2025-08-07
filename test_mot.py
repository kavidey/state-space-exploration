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
from pykalman.standard import _em_transition_covariance, _em_transition_matrix, _smooth_pair, _filter, _smooth, _loglikelihoods

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
Q = jnp.eye(4) * 0.05
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
# A = (jnp.eye(4) + jnr.normal(key, (4,4)) * 0.1).round(2)
# b = jnp.zeros((4))
# key, tmpkey = jnr.split(key)
Q = vec_to_cov_cholesky.forward(jnr.normal(key, (int(latent_dims*(latent_dims+1)/2),)) * 0.1).round(2)
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
# k = f_dist[0].shape[-1]
# T = f_dist[0].shape[0]
# elementwise_outer = jax.vmap(jnp.outer)

# sigma_t_and_t_sub_1_given_T = KalmanFilter.cross_covariance(q_dist, J_t)
# mu_t_given_T = q_dist[0]

# P_t = q_dist[1] + elementwise_outer(mu_t_given_T, mu_t_given_T)
# P_t_and_t_sub_1 = sigma_t_and_t_sub_1_given_T + elementwise_outer(mu_t_given_T[1:], mu_t_given_T[:-1])

# A_new = jnp.linalg.solve(P_t[:-1].sum(axis=0).T, P_t_and_t_sub_1.sum(axis=0).T).T
# pkf_A = _em_transition_matrix(b, q_dist[0], q_dist[1], jnp.concat((jnp.zeros((1,k,k)), sigma_t_and_t_sub_1_given_T)))

# print(f"A all close: {jnp.allclose(A_new, pkf_A)}")
# # print(A_new.round(3))
# # print(pkf_A.round(3))

# pkf_Q = _em_transition_covariance(A, b, q_dist[0], q_dist[1], jnp.concat((jnp.zeros((1,k,k)), sigma_t_and_t_sub_1_given_T)))

# # Q_new = 1/(T-1) * (
# #     P_t[1:] - P_t_and_t_sub_1 @ A.T - A_new @ jnp.moveaxis(P_t_and_t_sub_1, -1, -2) + A @ P_t[:-1] @ A.T
# # ).sum(axis=0)
# # Q_new = (1/(T-1)) * (P_t[1:].sum(axis=0) - A_new @ P_t_and_t_sub_1.sum(axis=0).T)
# Q_new = jnp.zeros((k, k))
# for t in range(T - 1):
#     err = (
#         q_dist[0][t + 1]
#         - jnp.dot(A, q_dist[0][t])
#         - b
#     )
#     Vt1t_A = jnp.dot(jnp.concat((jnp.zeros((1,k,k)), sigma_t_and_t_sub_1_given_T))[t+1], A.T)
#     Q_new += (
#         jnp.outer(err, err)
#         + jnp.dot(
#             A,
#             jnp.dot(q_dist[1][t], A.T),
#         )
#         + q_dist[1][t + 1]
#         - Vt1t_A
#         - Vt1t_A.T
#     )
# Q_new = (1/(T-1)) * Q_new

# print(f"Q all close: {jnp.allclose(Q_new, pkf_Q)}")
# print(Q_new.round(2))
# print(pkf_Q.round(2))
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
m_step_update = jax.jit(KalmanFilter.m_step_update)

log_likelihood = []
# %%
for i in range(10):
    pkf = pkf.em(X=x[0][1:], n_iter=1, em_vars=["transition_covariance"])

    # f_dist, p_dist, log_lik = KalmanFilter.run_forward((x[0][1:], x[1][1:]), z_t_sub_1, A, b, Q, H, jnp.zeros(49))
    p_dist_0, p_dist_1, _, f_dist_0, f_dist_1, = _filter(A, H, Q, R, b, jnp.zeros(2), z_t_sub_1[0], z_t_sub_1[1], x[0][1:])
    p_dist = (p_dist_0, p_dist_1)
    f_dist = (f_dist_0, f_dist_1)
    
    q_dist, J_t = KalmanFilter.run_backward(f_dist, A, b, Q, H)
    # q_dist_0, q_dist_1, J_t = _smooth(A, *f_dist, *p_dist)
    # q_dist = (q_dist_0, q_dist_1)

    log_likelihood.append(KalmanFilter.joint_log_likelihood((x[0][1:], x[1][1:]), q_dist, A, b, Q, H))
    # log_likelihood.append(_loglikelihoods(H, jnp.zeros((2)), R, *q_dist, x[0][1:]).sum())

    H_new, R_new, A_new, Q_new, _ = m_step_update((x[0][1:], x[1][1:]), z_t_sub_1, p_dist, f_dist, q_dist, J_t, A, b, Q, H, R)
    # sigma_t_and_t_sub_1_given_T = KalmanFilter.cross_covariance(q_dist, J_t)
    # sigma_t_and_t_sub_1_given_T = _smooth_pair(q_dist[1], J_t)[1:]
    # Q_new = _em_transition_covariance(A, b, q_dist[0], q_dist[1], jnp.concat((jnp.zeros((1,4,4)), sigma_t_and_t_sub_1_given_T)))

    print(f"Q all close: {jnp.allclose(Q_new, pkf.transition_covariance)}")
    print(Q_new.round(2))
    print(pkf.transition_covariance.round(2))

    # H = H_new
    # R = R_new
    # A = A_new
    Q = Q_new

    print(jnp.any(jnp.isnan(jnp.linalg.cholesky(Q))))

# f_dist, p_dist, log_lik = KalmanFilter.run_forward((x[0][1:], x[1][1:]), z_t_sub_1, A, b, Q, H, jnp.zeros(49))
# q_dist, J_t = KalmanFilter.run_backward(f_dist, A, b, Q, H)
p_dist_0, p_dist_1, _, f_dist_0, f_dist_1, = _filter(A, H, Q, R, b, jnp.zeros(2), z_t_sub_1[0], z_t_sub_1[1], x[0][1:])
p_dist = (p_dist_0, p_dist_1)
f_dist = (f_dist_0, f_dist_1)
q_dist_0, q_dist_1, J_t = _smooth(A, *f_dist, *p_dist)

pkf_q_dist = pkf.smooth(x[0][1:])

plt.scatter(*x[0].T, color='black')
plt.plot(*(H@q_dist[0].T))
plt.plot(*(H@pkf_q_dist[0].T), '--', linewidth=1)
# A, b, Q, R, H
# %%
plt.plot(log_likelihood)
# %%
