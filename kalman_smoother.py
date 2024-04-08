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


def predict(z: tfd.MultivariateNormalFullCovariance, A, b):
    '''
    Finds P(z_t+1 | z_t)
    '''
    # mu = Az + b (linear update step)
    mu = A @ z.loc + b
    sigma = z.covariance()
    
    return tfd.MultivariateNormalFullCovariance(mu, sigma)

def fake_problem(xhat, sigma):
    '''
    Evaluates P(xhat_t | z_t) to find a distribution over z
    This is possible because xhat_i and z are both known (calculated using the encoder)
    '''

    return tfd.MultivariateNormalFullCovariance(xhat, sigma)

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
    z = [
        predict(
            tfd.MultivariateNormalFullCovariance(jnp.zeros((latent_dims,)), jnp.identity(latent_dims)),
            A,
            b,
        )
    ]
    for i in range(1, len(x)):
        z[i] = multiply(predict(z[i - 1]), z[i - 1])


# def kalman_smoother(x, A, b):
