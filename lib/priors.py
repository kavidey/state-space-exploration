from typing import Tuple
import jax
import jax.numpy as jnp

from lib.distributions import MultivariateNormalFullCovariance

class KalmanFilter:
    @staticmethod
    def run(z, z_t_sub_1, z_rng, A, b, Q, H):
        kf_forward = lambda carry, z_t: KalmanFilter.forward(carry, z_t, A, b, Q, H)
        
        _, result = jax.lax.scan(kf_forward, (z_rng, z_t_sub_1), z)
        z_recon, q_dist, p_dist = result

        return z_recon, q_dist, p_dist
    
    @staticmethod
    def forward(carry, z_t, A, b, Q, H):
        z_rng, z_t_sub_1 = carry

        # Prediction
        z_t_given_t_sub_1 = KalmanFilter.predict(z_t_sub_1, A, b, Q)

        # Update
        z_t_given_t = KalmanFilter.update(z_t_given_t_sub_1, z_t, H)

        # Sample and decode
        z_rng, z_t_rng = jax.random.split(z_rng)
        z_hat = z_t_given_t.sample(z_t_rng)

        # jax.debug.print("z_t_given_t_sub_1: {z_t_given_t_sub_1}", z_t_given_t_sub_1=z_t_given_t_sub_1)
        # jax.debug.print("z_t_given_t: {z_t_given_t}", z_t_given_t=z_t_given_t)

        return (z_rng, z_t_given_t), (z_hat, z_t_given_t, z_t_given_t_sub_1) # carry, (z_recon, q_dist, p_dist)
    
    @staticmethod
    def predict(z_t, A, b, Q):
        """
        P(z_t+1 | x_t, ..., x_1) = P(z_t+1 | z_t)
        """
        # z_t|t-1 = A @ z_t-1|t-1 + b
        mu = jnp.dot(z_t.mean(), A) + b

        # P_t|t-1 = A @ P_t-1|t-1 @ A^T + Q
        sigma = A @ z_t.covariance() @ A.T + Q

        return MultivariateNormalFullCovariance(mu, sigma)

    @staticmethod
    def update(
        z_t_given_t_sub_1: MultivariateNormalFullCovariance, x_t: jnp.array, H: jnp.array
    ):
        """
        Kalman filter update step
        P(z_t+1 | x_t+1, ... , x_1) ~= P(x_t+1 | z_t+1) * P(z_t+1 | x_t, ... x_1)

        Args:
            z_t_given_t_sub_1 (MultivariateNormalFullCovariance): z_t|t-1
            x_t (MultivariateNormalFullCovariance): x_t

        Returns:
            MultivariateNormalFullCovariance: z_t|t
        """
        k = H.shape[0]
        # K_t = P_t|t-1 @ H^T @ (H @ P_t|t-1 @ H^T + R) ^ -1
        K_t = z_t_given_t_sub_1.covariance() @ H.T @ jnp.linalg.inv(H @ z_t_given_t_sub_1.covariance() @ H.T + x_t.covariance())

        # z_t|t = z_t|t-1 + K_t @ (x_t - H @ z_t|t-1)
        # Extra expand_dims and squeeze are necessary to make the matmul dimensions work
        mu = z_t_given_t_sub_1.mean() + K_t @ (x_t.mean() - H @ z_t_given_t_sub_1.mean())

        # P_t|t = P_t|t-1 - K_t @ H @ P_t|t-1 = (I - K_t @ H) @ P_t|t-1
        sigma = (jnp.eye(k) - K_t @ H) @ z_t_given_t_sub_1.covariance()

        return MultivariateNormalFullCovariance(mu, sigma)