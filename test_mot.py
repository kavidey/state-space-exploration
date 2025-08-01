# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

import jax
import jax.numpy as jnp
import jax.random as jnr
from jax import Array

import torch

from lib.priors import KalmanFilter, KalmanFilter_MOTPDA, KalmanFilter_MOTCAVI
from lib.distributions import MVN_kl_divergence, GMM_moment_match, MVN_multiply, MVN_Type, MVN_inverse_bayes

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
q_dist = KalmanFilter.run_backward(f_dist, A, b, Q, H)

plt.scatter(*x[0].T, color='black')
plt.plot(*(H@q_dist[0].T))
A, b, Q, R, H
# %% [markdown]
# use EM to optimize parameters
# %%
key, tmpkey = jnr.split(key)
A = jnp.eye(4) + jnr.normal(key, (4,4)) * 0.1
b = jnp.zeros((4))
key, tmpkey = jnr.split(key)
Q = jnp.eye(4) * 0.1 + jnr.normal(key, (4,4)) * 0.01
key, tmpkey = jnr.split(key)
R = jnp.eye(2) * 0.1 + jnr.normal(key, (2,2)) * 0.01
key, tmpkey = jnr.split(key)
H = jnp.array([[1,0,0,0],[0,1,0,0]]) + jnr.normal(key, (2,4)) * 0.1

f_dist, p_dist, log_lik = KalmanFilter.run_forward((x[0][1:], x[1][1:]), z_t_sub_1, A, b, Q, H, jnp.zeros(49))
q_dist = KalmanFilter.run_backward(f_dist, A, b, Q, H)

plt.scatter(*x[0].T, color='black')
plt.plot(*(H@q_dist[0].T))
A, b, Q, R, H
# %%
# @jax.jit
# def single_iter(carry, i):
#     A, b, Q, R, H = carry

#     f_dist, p_dist, log_lik = KalmanFilter.run_forward((x[0][1:], x[1][1:]), z_t_sub_1, A, b, Q, H, jnp.zeros(49))
#     q_dist = KalmanFilter.run_backward(f_dist, A, b, Q, H)

#     H, R, A, Q, _ = KalmanFilter.m_step_update((x[0][1:], x[1][1:]), z_t_sub_1, p_dist, f_dist, q_dist, A, Q, H, R)

#     return (A, b, Q, R, H), i

# (A, b, Q, R, H), _ = jax.lax.scan(single_iter, (A, b, Q, R, H), jnp.arange(2))

for i in range(2):
    f_dist, p_dist, log_lik = KalmanFilter.run_forward((x[0][1:], x[1][1:]), z_t_sub_1, A, b, Q, H, jnp.zeros(49))
    q_dist = KalmanFilter.run_backward(f_dist, A, b, Q, H)

    H, R, A, Q, _ = KalmanFilter.m_step_update((x[0][1:], x[1][1:]), z_t_sub_1, p_dist, f_dist, q_dist, A, Q, H, R)

plt.scatter(*x[0].T, color='black')
plt.plot(*(H@q_dist[0].T))
A, b, Q, R, H
# %%
