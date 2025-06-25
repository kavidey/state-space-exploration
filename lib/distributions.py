from typing import Tuple
import jax
import jax.numpy as jnp
import jax.random as jnr
from jax.experimental import checkify
from jax import Array

MVN_Type = tuple[Array, Array]

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
    
    def multiply(self, other: "MultivariateNormalFullCovariance") -> tuple[float, "MultivariateNormalFullCovariance"]:
        """
        Calculates the product of this distribution and another 
        
        Source: the matrix cookbook eqn 371

        Returns:
            float: log of the normalization constant
            MultivariateNormalFullCovariance: updated distribution
        """

        c, (mean, cov) = MVN_multiply(self.mean(), self.covariance(), other.mean(), other.covariance())
        
        return c, MultivariateNormalFullCovariance(mean, cov)

    def log_likelihood(self, x):
        return MVN_log_likelihood(self.mean(), self.covariance(), x)

jax.tree_util.register_pytree_node(MultivariateNormalFullCovariance,
                               MultivariateNormalFullCovariance._tree_flatten,
                               MultivariateNormalFullCovariance._tree_unflatten)

def MVN_multiply(m1: Array, c1: Array, m2: Array, c2: Array) -> tuple[float, MVN_Type]:
    '''
    Calculates the product of gaussian densities

    Source: the matrix cookbook eqn 371


    Parameter
    ---------
    m1: Array
        mean of first gaussian
    c1: Array
        covariance of first gaussian
    m2: Array
        mean of second gaussian
    c2: Array
        covariance of second gaussian

    Returns
    -------
    tuple[float, tuple[Array, Array]]
        float, product_gaussian

    '''

    k = m1.shape[-1]
    mean_diff = m1 - m2
    cov = c1 + c2
    # c = -(1/2) * (k * jnp.log(2*jnp.pi) + jnp.log(jnp.linalg.det(cov)) + mean_diff.T @ jnp.linalg.inv(cov) @ mean_diff)
    c = -(1/2) * (k * jnp.log(2*jnp.pi) + jnp.log(jnp.linalg.det(cov)) + mean_diff.T @ jnp.linalg.solve(cov, mean_diff))

    s_cov_inv = jnp.linalg.inv(c1)
    o_cov_inv = jnp.linalg.inv(c2)

    cov = jnp.linalg.inv(s_cov_inv+o_cov_inv)
    mean = cov @ (s_cov_inv @ c1.T + o_cov_inv @ c2.T)

    return c, (mean, cov)

def MVN_inverse_bayes(prior: MVN_Type, posterior: MVN_Type):
    '''
    Determines the gaussian likelihood function given a gaussian posterior and prior

    Derivation is simple using natural parameters

    Parameter
    ---------
    prior: tuple[Array, Array]
        gaussian prior function
    posterior: tuple[Array, Array]
        gaussian posterior function
    
    Returns
    -------
    tuple[Array, Array]
        gaussian likelihood function
    '''
    # if \Sigma_posterior^-1 - \Sigma_prior^-1 is not positive semidefinite then this is an invalid covariance matrix
    issue = jnp.any(jnp.isnan(jnp.linalg.cholesky(jnp.linalg.inv(posterior[1]) - jnp.linalg.inv(prior[1]))))

    # \Sigma_l = (\Sigma_posterior^-1 - \Sigma_prior^-1)^-1
    likelihood_sigma = jnp.linalg.inv(jnp.linalg.inv(posterior[1]) - jnp.linalg.inv(prior[1]))
    # \mu_l = \Sigma_l @ (\Sigma_posterior^-1 @ \mu_posterior - \Sigma_prior^-1 @ \mu_prior)
    likelihood_mu = likelihood_sigma @ (jnp.linalg.inv(posterior[1]) @ posterior[0] - jnp.linalg.inv(prior[1]) @ prior[0])
    
    return (likelihood_mu, likelihood_sigma)

def MVN_log_likelihood(mean: Array, cov: Array, x: Array) -> float:
    '''
    Calculates the likelihood of the observation under the gaussian distribution
    

    Parameter
    ---------
    mean: Array
        mean of distribution
    cov: Array
        covariance of distribution
    x: Array
        observed sample
    
    Returns
    ------
    float: log likelihood of observing x under the distribution
    '''
    k = mean.shape[-1]
    mean_diff = mean - x
    # log_likelihood = -(1/2) * (k * jnp.log(2*jnp.pi) + jnp.log(jnp.linalg.det(cov)) + mean_diff.T @ jnp.linalg.inv(cov) @ mean_diff)
    log_likelihood = -(1/2) * (k * jnp.log(2*jnp.pi) + jnp.log(jnp.linalg.det(cov)) + mean_diff.T @ jnp.linalg.solve(cov, mean_diff))
    return log_likelihood

def MVN_kl_divergence(mu_0: Array, sigma_0: Array, mu_1: Array, sigma_1: Array) -> float:
    k = mu_0.shape[-1]

    # \frac{1}{2} (\text{tr}(\Sigma_1^{-1}\Sigma_0) + (\mu_1 - \mu_0)^T \Sigma_1^{-1} (\mu_1-\mu_0)-k+\log(\frac{\det \Sigma_1}{\det \Sigma_0}))
    # a = jnp.trace(jnp.linalg.inv(sigma_1) @ sigma_0)
    a = jnp.trace(jnp.linalg.solve(sigma_1, sigma_0))
    mean_diff = mu_1 - mu_0
    # b = mean_diff.T @ jnp.linalg.inv(sigma_1) @ mean_diff
    b = mean_diff.T @ jnp.linalg.solve(sigma_1, mean_diff)
    c = jnp.log(jnp.linalg.det(sigma_1) / jnp.linalg.det(sigma_0))
    return 0.5 * (a + b - k + c)

def GMM_moment_match(dists: MVN_Type, weights: Array) -> MVN_Type:
    ''' Finds a gaussian with moments matching a multivariate distribution

    Test cases:
    ```python
    from dynamax.utils.plotting import plot_uncertainty_ellipses
    dists = (jnp.array([[-3], [3]]), jnp.array([[2], [2]]))
    weights = jnp.array([0.5, 0.5])

    # should be ([0], [11])
    print(GMM_moment_match(dists, weights))


    dist1 = (jnp.array([1, 0]), jnp.array([[1, 0], [0, 1]])*0.1)
    dist2 = (jnp.array([-1, 0]), jnp.array([[1, 0], [0, 1]])*0.1)
    weight = jnp.array([0.5, 0.5])

    dist1 = (jnp.array([1, 1]), jnp.array([[1, 0], [0, 1]])*0.1)
    dist2 = (jnp.array([-1, -1]), jnp.array([[1, 0], [0, 1]])*0.1)
    weight = jnp.array([0.5, 0.5])

    dist1 = (jnp.array([1, 1]), jnp.array([[1, 0], [0, 1]])*0.1)
    dist2 = (jnp.array([-1, -1]), jnp.array([[1, 0], [0, 1]])*0.1)
    weight = jnp.array([0.95, 0.05])

    combined = GMM_moment_match((jnp.array([dist1[0], dist2[0]]), jnp.array([dist1[1], dist2[1]])), weight)

    fig, ax = plt.subplots()
    # plot_uncertainty_ellipses from dynamax.utils.plotting
    plot_uncertainty_ellipses([dist1[0], dist2[0]], [dist1[1], dist2[1]], ax)
    plot_uncertainty_ellipses([combined[0]], [combined[1]], ax, edgecolor='tab:red')

    ax.set_xlim(-5,6)
    ax.set_ylim(-5,6)
    plt.show()
    ```
    '''
    # \mu = \Sum w_{k,i} mu_i
    mu = weights @ dists[0]

    # \Sigma = \Sum w_{k,i} * (\Sigma_i + (\mu - \mu_i) @ (\mu - \mu_i)^T)
    mean_diff = mu - dists[0]
    sigma = jnp.average(dists[1] + jax.vmap(jnp.outer)(mean_diff, mean_diff), weights=weights, axis=0)

    return (mu, sigma)