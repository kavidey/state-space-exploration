from typing import Tuple
import jax
import jax.numpy as jnp

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
        d = self.__mean.shape[-1]

        # https://juanitorduz.github.io/multivariate_normal/
        # Could also use https://jax.readthedocs.io/en/latest/_autosummary/jax.random.multivariate_normal.html
        epsilon = 0.0001
        K = self.__covariance + jnp.identity(d) * epsilon

        L = jnp.linalg.cholesky(K)
        u = jax.random.normal(seed, (d,))

        return self.__mean + jnp.dot(u, L)

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
        c = -(1/2) * (jnp.log(jnp.linalg.det(2*jnp.pi*(self.covariance() + other.covariance()))) + (self.mean()-other.mean()) @ jnp.linalg.inv(self.covariance() + other.covariance()) @ (self.mean()-other.mean()))

        s_cov_inv = jnp.linalg.inv(self.covariance())
        o_cov_inv = jnp.linalg.inv(other.covariance())

        cov = jnp.linalg.inv(s_cov_inv+o_cov_inv)
        mean = cov @ (s_cov_inv @ self.mean().T + o_cov_inv @ other.mean().T)
        
        return c, MultivariateNormalFullCovariance(mean, cov)

jax.tree_util.register_pytree_node(MultivariateNormalFullCovariance,
                               MultivariateNormalFullCovariance._tree_flatten,
                               MultivariateNormalFullCovariance._tree_unflatten)