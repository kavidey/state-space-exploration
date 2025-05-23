import jax
import jax.numpy as jnp
from jax import Array

from lib.distributions import MVN_log_likelihood, MVN_multiply
class KalmanFilter:
    @staticmethod
    def run_forward(z, z_t_sub_1, A, b, Q, H, mask):
        """
        Run Kalman Filter forward pass on a sequence of distributions and return results

        Parameter
        ---------
        z: tuple[Array, Array]
            list of observations to run the filter on represented as (mean, covariance)
        z_t_sub_1: tuple[Array, Array]
            prior on z[0] represented as (mean, covariance)
        A: Array
            state transition matrix
        b: Array
            state transition offset
        Q: Array
            process noise covariance matrix
        H: Array
            observation matrix

        Returns
        -------
        tuple[tuple[Array, Array], tuple[Array, Array], Array]
            q_dist, p_dist, log_likelihood

        """
        kf_forward = lambda carry, xs: KalmanFilter.forward(carry, xs[0], A, b, Q, H, xs[1]) # z_t = xs[0], mask = xs[1]

        z_1 = (z[0][0], z[1][0])
        q_1 = KalmanFilter.update(z_t_sub_1,
                                  z_1,
                                  H,
                                  mask[0])
        p_1 = z_t_sub_1
        log_likelihood1 = MVN_multiply(z_1[0], z_1[1], p_1[0], p_1[1])[0]
        if z[0].shape[0] > 1:
            _, result = jax.lax.scan(kf_forward, (q_1), ((z[0][1:], z[1][1:]), mask[1:]))
            q_dist, p_dist, log_likelihood = result

            q_dist = (
                jnp.vstack((jnp.expand_dims(q_1[0], axis=0), q_dist[0])),
                jnp.vstack((jnp.expand_dims(q_1[1], axis=0), q_dist[1])),
            )

            p_dist = (
                jnp.vstack((jnp.expand_dims(p_1[0], axis=0), p_dist[0])),
                jnp.vstack((jnp.expand_dims(p_1[1], axis=0), p_dist[1])),
            )

            log_likelihood = jnp.append(jnp.array(log_likelihood1), log_likelihood)

            return q_dist, p_dist, log_likelihood
        else:
            return (
                jnp.expand_dims(q_1[0], axis=0),
                jnp.expand_dims(q_1[1], axis=0)
            ), (
                jnp.expand_dims(p_1[0], axis=0),
                jnp.expand_dims(p_1[1], axis=0),
            ), jnp.array([log_likelihood1])
    
    @staticmethod
    def forward(carry, z_t, A, b, Q, H, mask=0):
        """
        Single iteration of Kalman Filter forward pass
        """
        z_t_sub_1 = carry

        # Prediction
        z_t_given_t_sub_1 = KalmanFilter.predict(z_t_sub_1, A, b, Q)

        # Update
        z_t_given_t = KalmanFilter.update(z_t_given_t_sub_1, z_t, H, mask=mask)

        # Log-Likelihood
        log_likelihood = MVN_log_likelihood(
            H @ z_t_given_t_sub_1[0],
            z_t[1] + H @ z_t_given_t_sub_1[1] @ H.T,
            z_t[0]
        )
        return (z_t_given_t), (z_t_given_t, z_t_given_t_sub_1, log_likelihood) # carry, (q_dist, p_dist, log_likelihood)
    
    @staticmethod
    def predict(z_t, A, b, Q):
        """
        P(z_t+1 | x_t, ..., x_1) = P(z_t+1 | z_t)
        """
        # z_t|t-1 = A @ z_t-1|t-1 + b
        mu = A @ z_t[0] + b

        # P_t|t-1 = A @ P_t-1|t-1 @ A^T + Q
        sigma = A @ z_t[1] @ A.T + Q

        return (mu, sigma)

    @staticmethod
    def update(
        z_t_given_t_sub_1: tuple[Array, Array], x_t: Array, H: Array, mask = 0
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
        hat_x_t = H @ z_t_given_t_sub_1[0]

        # S_t = H @ P_t|t-1 @ H^T + R
        S_t = H @ z_t_given_t_sub_1[1] @ H.T + x_t[1]
        # K_t = P_t|t-1 @ H^T @ S^-1
        # K_t = z_t_given_t_sub_1[1] @ H.T @ jnp.linalg.inv(S_t)
        K_t = z_t_given_t_sub_1[1] @ (jnp.linalg.solve(S_t.T, H)).T # (AB^-1)^T = (B^-1)^T A^T = (B^T)^-1 A^T

        # z_t|t = z_t|t-1 + K_t @ (x_t - H @ z_t|t-1)
        mu = z_t_given_t_sub_1[0] + K_t @ (x_t[0] - hat_x_t)

        # P_t|t = P_t|t-1 - K_t @ S @ K_t^T
        sigma = z_t_given_t_sub_1[1] - K_t @ S_t @ K_t.T

        return jax.lax.cond(mask, lambda: z_t_given_t_sub_1, lambda: (mu, sigma))
    
    @staticmethod
    def run_backward(p_dist, A, b, Q, H) -> tuple[Array, Array]:
        """
        Run Kalman Filter backward pass on a sequence of distributions and return results

        Parameter
        ---------
        p_dist: tuple[Array, Array]
            predicted state distribution represented as (mean, covariance)
        A: Array
            state transition matrix
        b: Array
            state transition offset
        Q: Array
            process noise covariance matrix
        H: Array
            observation matrix
        
        Returns
        -------
        q_dist: tuple[Array, Array]
            smoothed state distribution represented as (mean, covariance)
        """
        kf_backward = lambda carry, z_t: KalmanFilter.backward(carry, z_t, A, b, Q, H)
        
        q_dist_T = (p_dist[0][-1], p_dist[1][-1])
        q_dist_1_to_T_sub_1 = (p_dist[0][:-1], p_dist[1][:-1])
        _, q_dist = jax.lax.scan(kf_backward, (q_dist_T), q_dist_1_to_T_sub_1, reverse=True)

        q_dist = (
            jnp.vstack((q_dist[0], jnp.expand_dims(q_dist_T[0], 0))),
            jnp.vstack((q_dist[1], jnp.expand_dims(q_dist_T[1], 0)))
        )

        return q_dist
    
    @staticmethod
    def backward(carry, z_t: tuple[Array, Array], A, b, Q, H):
        """
        Kalman Filter Smooth Step
        """
        z_t_plus_1 = carry

        # A @ P_t @ A^T + Q
        P_pred = A @ z_t[1] @ A.T + Q

        # Kalman Gain
        # P_t @ A^T @ P_pred^-1
        # K = z_t[1] @ A.T @ jnp.linalg.inv(P_pred)
        K = z_t[1] @ (jnp.linalg.solve(P_pred.T, A)).T # (AB^-1)^T = (B^-1)^T A^T = (B^T)^-1 A^T

        # mu_t|T = mu_t|1:t + K @ (mu_t+1|T - A @ mu_i|1:t)
        mu = z_t[0] + K @ (z_t_plus_1[0] - (A @ z_t[0] + b))

        # P_t|T = P_t|1:t + K @ (P_t+1|T - P_t+1|1:t) @ K^T
        sigma = z_t[1] + K @ (z_t_plus_1[1] - P_pred) @ K.T

        z_t_given_T = (mu, sigma)

        return (z_t_given_T), (z_t_given_T) # carry, (posterior_dist)