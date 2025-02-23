from typing import Tuple
import jax
import jax.numpy as jnp
import jax.random as jnr

class MultivariateNormalFullCovariance:
    def __init__(self, mean: jnp.ndarray, covariance: jnp.ndarray):
        self.__mean = mean
        self.__covariance = covariance

    def mean(self):
        return self.__mean

    def covariance(self):
        return self.__covariance

    @jax.jit
    def sample(self, seed):

        # https://juanitorduz.github.io/multivariate_normal/
        # Could also use https://jax.readthedocs.io/en/latest/_autosummary/jax.random.multivariate_normal.html
        # d = self.__mean.shape[-1]
        # epsilon = 0.0001
        # K = self.__covariance + jnp.identity(d) * epsilon

        # L = jnp.linalg.cholesky(K)
        # u = jnr.normal(seed, (d,))

        # return self.__mean + jnp.dot(u, L)

        return jnr.multivariate_normal(seed, self.__mean, self.__covariance)

    def __repr__(self) -> str:
        return f"MultivariateNormalFullCovariance(mu={self.__mean}, sigma={self.__covariance})"
    
    def _tree_flatten(self):
        children = (self.__mean, self.__covariance)  # arrays / dynamic values
        aux_data = {}  # static values
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    
    def multiply(self, other: "MultivariateNormalFullCovariance") -> Tuple[float, "MultivariateNormalFullCovariance"]:
        """
        Calculates the product of this distribution and another 
        
        Source: the matrix cookbook eqn 371

        Returns:
            float: log of the normalization constant
            MultivariateNormalFullCovariance: updated distribution
        """
        # c = (1/(jnp.sqrt(jnp.linalg.det(2*jnp.pi*(self.covariance() + other.covariance()))))) * \
        #     jnp.exp(-(1/2)*(self.mean()-other.mean()) @ jnp.linalg.inv(self.covariance() + other.covariance()) @ (self.mean()-other.mean()))
        k = self.mean().shape[-1]
        mean_diff = self.mean() - other.mean()
        cov = self.covariance() + other.covariance()
        c = -(1/2) * (k * jnp.log(2*jnp.pi) + jnp.log(jnp.linalg.det(cov)) + mean_diff.T @ jnp.linalg.inv(cov) @ mean_diff)

        s_cov_inv = jnp.linalg.inv(self.covariance())
        o_cov_inv = jnp.linalg.inv(other.covariance())

        cov = jnp.linalg.inv(s_cov_inv+o_cov_inv)
        mean = cov @ (s_cov_inv @ self.mean().T + o_cov_inv @ other.mean().T)
        
        return c, MultivariateNormalFullCovariance(mean, cov)

    def log_likelihood(self, x):
        k = self.mean().shape[-1]
        mean_diff = self.mean() - x
        log_likelihood = -(1/2) * (k * jnp.log(2*jnp.pi) + jnp.log(jnp.linalg.det(self.covariance())) + mean_diff.T @ jnp.linalg.inv(self.covariance()) @ mean_diff)
        return log_likelihood

jax.tree_util.register_pytree_node(MultivariateNormalFullCovariance,
                               MultivariateNormalFullCovariance._tree_flatten,
                               MultivariateNormalFullCovariance._tree_unflatten)

def kl_divergence(
    q: MultivariateNormalFullCovariance, p: MultivariateNormalFullCovariance
):
    mu_0 = q.mean()
    sigma_0 = q.covariance()

    mu_1 = p.mean()
    sigma_1 = p.covariance()

    k = mu_0.shape[-1]

    # \frac{1}{2} (\text{tr}(\Sigma_1^{-1}\Sigma_0) + (\mu_1 - \mu_0)^T \Sigma_1^{-1} (\mu_1-\mu_0)-k+\log(\frac{\det \Sigma_1}{\det \Sigma_0}))
    a = jnp.trace(jnp.linalg.inv(sigma_1) @ sigma_0)
    mean_diff = mu_1 - mu_0
    b = mean_diff.T @ jnp.linalg.inv(sigma_1) @ mean_diff
    c = jnp.log(jnp.linalg.det(sigma_1) / jnp.linalg.det(sigma_0))
    return 0.5 * (a + b - k + c)
