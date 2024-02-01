# %%
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
# %%
xs = np.random.normal(size=(100,))
noise = np.random.normal(scale=0.1, size=(100,))
ys = xs * 3 + 1 + noise

plt.scatter(xs, ys)
plt.show()
# %%
def model(theta, x):
    w, b = theta
    return w * x + b

def loss_fn(theta, x, y):
    prediction = model(theta, x)
    return jnp.mean((prediction-y)**2)

@jax.jit
def update(theta, x, y, lr=0.1):
    return theta - lr * jax.grad(loss_fn)(theta, x, y)
# %%
theta = jnp.array([1., 1.])

for _ in range(1000):
    theta = update(theta, xs, ys)

plt.scatter(xs, ys)
plt.plot(xs, model(theta, xs), c="black")

w, b = theta
print(f"w = {w:<.2f}")
print(f"b = {b:<.2f}")
# %%
