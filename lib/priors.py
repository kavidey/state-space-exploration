from typing import Tuple
import jax
import jax.numpy as jnp

from lib.distributions import MultivariateNormalFullCovariance

class KalmanFilter:
    @staticmethod
    def run_forward(z, z_t_sub_1, A, b, Q, H):
        """
        Run Kalman Filter forward pass on a sequence of distributions and return results
        """
        kf_forward = lambda carry, z_t: KalmanFilter.forward(carry, z_t, A, b, Q, H)

        z_1 = MultivariateNormalFullCovariance(z.mean()[0], z.covariance()[0])
        q_1 = KalmanFilter.update(z_t_sub_1,
                                  z_1,
                                  H)
        p_1 = z_t_sub_1
        log_likelihood1 = z_1.multiply(p_1)[0]
        if z.mean().shape[0] > 1:
            _, result = jax.lax.scan(kf_forward, (q_1),
                                    MultivariateNormalFullCovariance(z.mean()[1:], z.covariance()[1:]))
            q_dist, p_dist, log_likelihood = result

            q_dist = MultivariateNormalFullCovariance(
                jnp.vstack((jnp.expand_dims(q_1.mean(), axis=0), q_dist.mean())),
                jnp.vstack((jnp.expand_dims(q_1.covariance(), axis=0), q_dist.covariance())),
            )

            p_dist = MultivariateNormalFullCovariance(
                jnp.vstack((jnp.expand_dims(p_1.mean(), axis=0), p_dist.mean())),
                jnp.vstack((jnp.expand_dims(p_1.covariance(), axis=0), p_dist.covariance())),
            )

            log_likelihood = jnp.append(jnp.array(log_likelihood1), log_likelihood)

            return q_dist, p_dist, log_likelihood
        else:
            return MultivariateNormalFullCovariance(
                jnp.expand_dims(q_1.mean(), axis=0),
                jnp.expand_dims(q_1.covariance(), axis=0)
            ), MultivariateNormalFullCovariance(
                jnp.expand_dims(p_1.mean(), axis=0),
                jnp.expand_dims(p_1.covariance(), axis=0),
            ), jnp.array([log_likelihood1])
    
    @staticmethod
    def forward(carry, z_t, A, b, Q, H):
        """
        Single iteration of Kalman Filter forward pass
        """
        z_t_sub_1 = carry

        # Prediction
        z_t_given_t_sub_1 = KalmanFilter.predict(z_t_sub_1, A, b, Q)

        # Update
        z_t_given_t = KalmanFilter.update(z_t_given_t_sub_1, z_t, H)

        # Log-Likelihood
        log_likelihood = MultivariateNormalFullCovariance(H @ z_t_given_t_sub_1.mean(), z_t.covariance() + H @ z_t_given_t_sub_1.covariance() @ H.T).log_likelihood(z_t.mean())
        return (z_t_given_t), (z_t_given_t, z_t_given_t_sub_1, log_likelihood) # carry, (q_dist, p_dist)
    
    @staticmethod
    def predict(z_t, A, b, Q):
        """
        P(z_t+1 | x_t, ..., x_1) = P(z_t+1 | z_t)
        """
        # z_t|t-1 = A @ z_t-1|t-1 + b
        mu = A @ z_t.mean() + b

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

        # hat_x_t = H @ z_t|t-1
        hat_x_t = H @ z_t_given_t_sub_1.mean()

        # S_t = H @ P_t|t-1 @ H^T + R
        S_t = H @ z_t_given_t_sub_1.covariance() @ H.T + x_t.covariance()
        # K_t = P_t|t-1 @ H^T @ S^-1
        K_t = z_t_given_t_sub_1.covariance() @ H.T @ jnp.linalg.inv(S_t)

        # z_t|t = z_t|t-1 + K_t @ (x_t - H @ z_t|t-1)
        mu = z_t_given_t_sub_1.mean() + K_t @ (x_t.mean() - hat_x_t)

        # P_t|t = P_t|t-1 - K_t @ S @ K_t^T
        sigma = z_t_given_t_sub_1.covariance() - K_t @ S_t @ K_t.T

        return MultivariateNormalFullCovariance(mu, sigma)
    
    @staticmethod
    def run_backward(p_dist, z_rng, A, b, Q, H):
        """
        Run Kalman Filter forward pass on a sequence of distributions and return results
        """
        tmpkey, z_rng = jax.random.split(z_rng)
        kf_forward = lambda carry, z_t: KalmanFilter.backward(carry, z_t, A, b, Q, H)
        
        p_dist_T = MultivariateNormalFullCovariance(p_dist.mean()[-1], p_dist.covariance()[-1])
        p_dist_1_to_T_sub_1 = MultivariateNormalFullCovariance(p_dist.mean()[:-1], p_dist.covariance()[:-1])
        _, result = jax.lax.scan(kf_forward, (z_rng, p_dist_T), p_dist_1_to_T_sub_1, reverse=True)
        z_hat, p_dist = result

        p_dist = MultivariateNormalFullCovariance(
            jnp.vstack((p_dist.mean(), jnp.expand_dims(p_dist_T.mean(), 0))),
            jnp.vstack((p_dist.covariance(), jnp.expand_dims(p_dist_T.covariance(), 0)))
        )

        z_hat = jnp.vstack((z_hat, jnp.expand_dims(p_dist_T.sample(tmpkey), 0)))

        return z_hat, p_dist
    
    @staticmethod
    def backward(carry, z_t: MultivariateNormalFullCovariance, A, b, Q, H):
        z_rng, z_t_plus_1 = carry

        # A @ P_t @ A^T + Q
        P_pred = A @ z_t.covariance() @ A.T + Q

        # Kalman Gain
        # P_t @ A^T @ P_pred^-1
        K = z_t.covariance() @ A.T @ jnp.linalg.inv(P_pred)

        # mu_t|T = mu_t|1:t + K @ (mu_t+1|T - A @ mu_i|1:t)
        mu = z_t.mean() + K @ (z_t_plus_1.mean() - (A @ z_t.mean() + b)) # TODO: CHECK THE +b HERE

        # P_t|T = P_t|1:t + K @ (P_t+1|T - P_t|1:t) @ K^T
        sigma = z_t.covariance() + K @ (z_t_plus_1.covariance() - z_t.covariance()) @ K.T

        z_t_given_T = MultivariateNormalFullCovariance(mu, sigma)
        
        # Sample and decode
        z_rng, z_t_rng = jax.random.split(z_rng)
        z_hat = z_t_given_T.sample(z_t_rng)

        return (z_rng, z_t_given_T), (z_hat, z_t_given_T) # carry, (z_recon, posterior_dist)