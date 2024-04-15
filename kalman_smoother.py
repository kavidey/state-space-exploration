# %%
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
# %%
latent_dims = 10

# %%
def multiply(
    a: tfd.MultivariateNormalFullCovariance, b: tfd.MultivariateNormalFullCovariance
):
    # sigma = 1 / (1/sigma_a + 1/sigma_b)
    sigma = jnp.linalg.inv(
        (jnp.linalg.inv(a.covariance()) + jnp.linalg.inv(b.covariance()))
    )
    # mu = sigma * (mu_a / sigma_a + mu_b / sigma_b)
    mu = sigma * (a.mean() @ jnp.linalg.inv(a.covariance()) + b.mean() @ jnp.linalg.inv(b.covariance()))

    return tfd.MultivariateNormalFullCovariance(mu, sigma)

def predict(z, A, b):
    '''
    P(z_t+1 | x_t, ..., x_1) = P(z_t+1 | z_t)
    '''
    mu = A @ z.mean() + b
    Q = jnp.zeros((latent_dims, latent_dims))
    sigma = A @ z.covariance() @ jnp.linalg.inv(A) + Q
    return tfd.MultivariateNormalFullCovariance(mu, sigma)

def update(z, x, A, b):
    '''
    P(z_t+1 | x_t+1, ... , x_1) ~= P(x_t+1 | z_t+1) * P(z_t+1 | x_t, ... x_1)
    '''
    pred = predict(z, A, b)

    R = jnp.zeros((latent_dims, latent_dims))
    K = pred.covariance() @ jnp.linalg.inv(pred.covariance() + R)
    
    sigma = pred.covariance() - K @ pred.covariance()
    mu = pred.mu() + K @ (x - pred.mu())

    return tfd.MultivariateNormalFullCovariance(mu, sigma)
# %%
def encoder(x):
    '''
    Returns the fake observation xhat and the covariance of the latent distribution z
    P(x_t | z_t)
    '''
    return jnp.zeros((latent_dims,)), jnp.identity(latent_dims)

# %%
def kalman_filter(x, A, b):
    xhat = jax.lax.map(encoder, x)