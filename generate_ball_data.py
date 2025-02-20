# %% [markdown]
# This script is modified from the RTRBM code by Ilya Sutskever from
# http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar
# 
# and
# 
# https://github.com/stelzner/Visual-Interaction-Networks/blob/master/create_billards_data.py
# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
# %%
shape_std = np.shape
def shape(A):
    if isinstance(A, np.ndarray):
        return shape_std(A)
    else:
        return A.shape()

size_std = np.size

def size(A):
    if isinstance(A, np.ndarray):
        return size_std(A)
    else:
        return A.size()

det = np.linalg.det

def new_speeds(m1, m2, v1, v2):
    new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2

def norm(x): return np.sqrt((x ** 2).sum())

def sigmoid(x): return 1. / (1. + np.exp(-x))

SIZE = 10
# %%
# size of bounding box: SIZE X SIZE.

def bounce_n(T=128, n=2, r=None, m=None):
    if r is None:
        r = np.array([1.2] * n)
    if m is None:
        m = np.array([1] * n)
    # r is to be rather small.
    X = np.zeros((T, n, 2), dtype='float')
    y = np.zeros((T, n, 2), dtype='float')
    v = np.random.randn(n, 2)
    v = v / norm(v) * .5
    good_config = False
    while not good_config:
        x = 2 + np.random.rand(n, 2) * 8
        good_config = True
        for i in range(n):
            for z in range(2):
                if x[i][z] - r[i] < 0:
                    good_config = False
                if x[i][z] + r[i] > SIZE:
                    good_config = False

        # that's the main part.
        for i in range(n):
            for j in range(i):
                if norm(x[i] - x[j]) < r[i] + r[j]:
                    good_config = False

    eps = .5
    for t in range(T):
        # for how long do we show small simulation

        v_prev = np.copy(v)

        for i in range(n):
            X[t, i] = x[i]
            y[t, i] = v[i]

        for mu in range(int(1 / eps)):

            for i in range(n):
                x[i] += eps * v[i]

            for i in range(n):
                for z in range(2):
                    if x[i][z] - r[i] < 0:
                        v[i][z] = abs(v[i][z])  # want positive
                    if x[i][z] + r[i] > SIZE:
                        v[i][z] = -abs(v[i][z])  # want negative

            for i in range(n):
                for j in range(i):
                    if norm(x[i] - x[j]) < r[i] + r[j]:
                        # the bouncing off part:
                        w = x[i] - x[j]
                        w = w / norm(w)

                        v_i = np.dot(w.transpose(), v[i])
                        v_j = np.dot(w.transpose(), v[j])

                        new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)

                        v[i] += w * (new_v_i - v_i)
                        v[j] += w * (new_v_j - v_j)

    return X, y

def ar(x, y, z):
    return z / 2 + np.arange(x, y, z, dtype='float')

def draw_image(X, res, r=None):
    T, n = shape(X)[0:2]
    if r is None:
        r = np.array([1.2] * n)

    A = np.zeros((T, res, res, 3), dtype='float')

    [I, J] = np.meshgrid(ar(0, 1, 1. / res) * SIZE, ar(0, 1, 1. / res) * SIZE)

    for t in range(T):
        for i in range(n):
            A[t, :, :, i] += np.exp(-(((I - X[t, i, 0]) ** 2 +
                                    (J - X[t, i, 1]) ** 2) /
                                   (r[i] ** 2)) ** 4)

        A[t][A[t] > 1] = 1
    return A

def bounce_mat(res, n=2, T=128, r=None):
    if r is None:
        r = np.array([1.2] * n)
    x, y = bounce_n(T, n, r)
    A = draw_image(x, res, r)
    return A, y

def bounce_vec(res, n=2, T=128, r=None, m=None):
    if r is None:
        r = np.array([1.2] * n)
    x, y = bounce_n(T, n, r, m)
    V = draw_image(x, res, r)
    y = np.concatenate((x, y), axis=2)
    return V.reshape(T, res, res, 3), y

def show_sample(logdir, V):
    T = V.shape[0]
    for t in tqdm(range(T), desc="Saving sample sequence"):
        plt.imshow(V[t])
        # Save it
        fname = logdir  / (str(t) + '.png')
        plt.savefig(fname)
# %%
dataset_dir = Path("dataset") / "billiard"
dataset_dir.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

res = 32
T = 100
N = 1000
# %%
dat = np.empty((N, T, res, res, 3), dtype=float)
dat_y = np.empty((N, T, 3, 4), dtype=float)
for i in tqdm(range(N), desc="train"):
    dat[i], dat_y[i] = bounce_vec(res=res, n=3, T=T)
np.savez(dataset_dir / "train.npz", X=dat, y=dat_y)

N = 200
dat = np.empty((N, T, res, res, 3), dtype=float)
dat_y = np.empty((N, T, 3, 4), dtype=float)
for i in tqdm(range(N), desc="test"):
    dat[i], dat_y[i] = bounce_vec(res=res, n=3, T=T)
np.savez(dataset_dir / "test.npz", X=dat, y=dat_y)

# show one video
sample_dir = dataset_dir / "sample"
sample_dir.mkdir(exist_ok=True)
show_sample(sample_dir,dat[0])
# %%
