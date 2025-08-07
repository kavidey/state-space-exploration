import jax
import jax.numpy as jnp
from jax import Array

from lib.distributions import MVN_Type, MVN_log_likelihood, MVN_multiply, MVN_inverse_bayes, GMM_moment_match
class KalmanFilter:
    @staticmethod
    def run_forward(x: MVN_Type, z_t_sub_1: MVN_Type, A: Array, b: Array, Q: Array, H: Array, mask):
        """
        Run Kalman Filter forward pass on a sequence of distributions and return results

        Parameter
        ---------
        x: tuple[Array, Array]
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
        mask: Array
            list of sample to mask observations for

        Returns
        -------
        tuple[tuple[Array, Array], tuple[Array, Array], Array]
            q_dist, p_dist, log_likelihood

        """
        kf_forward = lambda carry, xs: KalmanFilter.forward(carry, xs[0], A, b, Q, H, xs[1]) # z_t = xs[0], mask = xs[1]

        x_1 = (x[0][0], x[1][0])
        q_1 = KalmanFilter.update(z_t_sub_1,
                                  x_1,
                                  H,
                                  mask[0])
        p_1 = z_t_sub_1

        # p(x_1) = \int p(z_1) p(x_1|z_1) dz_1
        # TODO: should we project p into x space or vice versa?
        log_likelihood1 = MVN_multiply(*x_1, *(H @ p_1[0], H @ p_1[1] @ H.T))[0]
        
        if x[0].shape[0] > 1:
            _, result = jax.lax.scan(kf_forward, (q_1), ((x[0][1:], x[1][1:]), mask[1:]))
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
    def forward(carry: MVN_Type, x_t: MVN_Type, A: Array, b: Array, Q: Array, H: Array, mask=0):
        """
        Single iteration of Kalman Filter forward pass
        """
        z_t_sub_1 = carry

        # Prediction
        z_t_given_t_sub_1 = KalmanFilter.predict(z_t_sub_1, A, b, Q)

        # Update
        z_t_given_t = KalmanFilter.update(z_t_given_t_sub_1, x_t, H, mask=mask)

        # Log-Likelihood
        # project z_{t|t-1} into x (observation) space
        z_t_given_t_sub_1_x_space = (H @ z_t_given_t_sub_1[0], H @ z_t_given_t_sub_1[1] @ H.T)
        # p(x_t) = \int p(z_i|x_{1:i-1}) p(x_i|z_i) dz_i
        log_likelihood = MVN_multiply(*z_t_given_t_sub_1_x_space, *x_t)[0]
        return (z_t_given_t), (z_t_given_t, z_t_given_t_sub_1, log_likelihood) # carry, (q_dist, p_dist, log_likelihood)
    
    @staticmethod
    def predict(z_t: MVN_Type, A: Array, b: Array, Q: Array):
        """
        Kalman filter predict step
        P(z_t+1 | x_t, ..., x_1) = P(z_t+1 | z_t)
        """
        # z_t|t-1 = A @ z_t-1|t-1 + b
        mu = A @ z_t[0] + b

        # \Sigma_t|t-1 = A @ \Sigma_t-1|t-1 @ A^T + Q
        sigma = A @ z_t[1] @ A.T + Q

        return (mu, sigma)

    @staticmethod
    def update(
        z_t_given_t_sub_1: MVN_Type, x_t: MVN_Type, H: Array, mask=0
    ):
        """
        Kalman filter update step
        P(z_t+1 | x_t+1, ... , x_1) ~= P(x_t+1 | z_t+1) * P(z_t+1 | x_t, ... x_1)

        Parameter
        ---------
            z_t_given_t_sub_1: tuple[Array, Array]
                distribution for z_t|t-1
            x_t: tuple[Array, Array]
                distribution for x_t

        Returns:
            MultivariateNormalFullCovariance: z_t|t
        """

        # hat_x_t = H @ z_t|t-1
        hat_x_t = H @ z_t_given_t_sub_1[0]

        # S_t = H @ \Sigma_t|t-1 @ H^T + R
        S_t = H @ z_t_given_t_sub_1[1] @ H.T + x_t[1]
        # K_t = \Sigma_t|t-1 @ H^T @ S^-1
        # K_t = z_t_given_t_sub_1[1] @ H.T @ jnp.linalg.inv(S_t)
        K_t = z_t_given_t_sub_1[1] @ (jnp.linalg.solve(S_t.T, H)).T # (AB^-1)^T = (B^-1)^T A^T = (B^T)^-1 A^T

        # z_t|t = z_t|t-1 + K_t @ (x_t - H @ z_t|t-1)
        mu = z_t_given_t_sub_1[0] + K_t @ (x_t[0] - hat_x_t)

        # \Sigma_t|t = \Sigma_t|t-1 - K_t @ S @ K_t^T
        sigma = z_t_given_t_sub_1[1] - K_t @ S_t @ K_t.T

        return jax.lax.cond(mask, lambda: z_t_given_t_sub_1, lambda: (mu, sigma))
    
    @staticmethod
    def run_backward(f_dist: MVN_Type, A: Array, b: Array, Q: Array, H: Array) -> tuple[MVN_Type, Array]:
        """
        Run Kalman Filter backward pass on a sequence of distributions and return results

        Parameter
        ---------
        f_dist: tuple[Array, Array]
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
        
        q_dist_T = (f_dist[0][-1], f_dist[1][-1])
        q_dist_1_to_T_sub_1 = (f_dist[0][:-1], f_dist[1][:-1])
        _, (q_dist, K_t) = jax.lax.scan(kf_backward, (q_dist_T), q_dist_1_to_T_sub_1, reverse=True)

        q_dist = (
            jnp.vstack((q_dist[0], jnp.expand_dims(q_dist_T[0], 0))),
            jnp.vstack((q_dist[1], jnp.expand_dims(q_dist_T[1], 0)))
        )

        return q_dist, K_t
    
    @staticmethod
    def backward(carry, z_t: MVN_Type, A: Array, b: Array, Q: Array, H: Array):
        """
        Kalman Filter Smooth Step
        """
        z_t_plus_1 = carry

        # A @ \Sigma_t @ A^T + Q
        P_pred = A @ z_t[1] @ A.T + Q

        # Kalman Gain
        # \Sigma_t @ A^T @ \Sigma_pred^-1
        # J = z_t[1] @ A.T @ jnp.linalg.inv(P_pred)
        J = z_t[1] @ (jnp.linalg.solve(P_pred.T, A)).T # (AB^-1)^T = (B^-1)^T A^T = (B^T)^-1 A^T

        # mu_t|T = mu_t|1:t + K @ (mu_t+1|T - A @ mu_i|1:t)
        mu = z_t[0] + J @ (z_t_plus_1[0] - (A @ z_t[0] + b))

        # \Sigma_t|T = \Sigma_t|1:t + K @ (\Sigma_t+1|T - \Sigma_t+1|1:t) @ K^T
        sigma = z_t[1] + J @ (z_t_plus_1[1] - P_pred) @ J.T

        z_t_given_T = (mu, sigma)

        return (z_t_given_T), (z_t_given_T, J) # carry, (posterior_dist, J_t)

    @staticmethod
    def cross_covariance(q_dist: MVN_Type, J_t):
        """
        Calculates pairwise covariance between t and t-1

        Parameter
        ---------
        q_dist: tuple[Array, Array]
            updated state distribution represented as (mean, covariance)
        J_t: Array
            sequence of kalman gains
        
        Returns
        -------
        sigma_t_and_t_sub_1_given_T: Array
            pairwise covariance, 0th entry is missing
        """

        r"""
        Alternative algorithm
        ```
        K_T = p_dist[1][-1] @ H.T @ jnp.linalg.inv(H @ p_dist[1][-1] @ H.T + R)

        # \Sigma_{T,T-1|T} = (I - K_T@H) @ A @ \Sigma_{T-1|T-1}
        sigma_T_and_T_sub_1_given_T_last = (jnp.eye(k) - K_T@H) @ A @ f_dist[1][-2]

        # recursively calculate previous items in sequence using `scan`
        def cross_cov_recurse(carry, x):
            # \Sigma_{t+1,t|T}
            sigma_t_add_1_and_t_given_T = carry
            
            # (\Sigma_{t|t}, \Sigma_{t-1|t-1}, \Sigma_{t|t-1})
            sigma_t_given_t, sigma_t_sub_1_given_t_sub_1, sigma_t_given_t_sub_1 = x

            # J_{t-1} = \Sigma_{t-1|t-1} @ A^T @ \Sigma_{t|t-1}^{-1}
            J_t_sub_1 = sigma_t_sub_1_given_t_sub_1 @ A.T @ jnp.linalg.inv(sigma_t_given_t_sub_1)  # TODO: optimize with solve

            # \Sigma_{t,t-1|T} = \Sigma_{t|t} @ J_{t-1}^T + J_{t-1} @ (\Sigma_{t+1,t|T} - A@\Sigma_{t|t}) @ J_{t-1}^T
            sigma_t_and_t_sub_1_given_T = sigma_t_given_t @ J_t_sub_1.T + J_t_sub_1@(sigma_t_add_1_and_t_given_T - A@sigma_t_given_t)@J_t_sub_1.T
            return sigma_t_and_t_sub_1_given_T, sigma_t_and_t_sub_1_given_T
        #                                                                                                  (\Sigma_{t|t},  \Sigma_{t-1|t-1}, \Sigma_{t|t-1})
        _, sigma_t_and_t_sub_1_given_T = jax.lax.scan(cross_cov_recurse, sigma_T_and_T_sub_1_given_T_last, (f_dist[1][1:], f_dist[1][:-1],   p_dist[1][1:]), reverse=True)
        sigma_t_and_t_sub_1_given_T = jnp.concat((sigma_t_and_t_sub_1_given_T, sigma_T_and_T_sub_1_given_T_last[None]), axis=0)
        sigma_t_and_t_sub_1_given_T = jnp.flip(sigma_t_and_t_sub_1_given_T, axis=0)
        ```
        """

        # \Sigma_{t, t-1 | T} = \Sigma_{t|T} @ J_t-1
        return q_dist[1][1:] @ jnp.moveaxis(J_t, -1, -2)

    @staticmethod
    def joint_log_likelihood(x_t: MVN_Type, z_t: MVN_Type, A: Array, b: Array, Q: Array, H: Array):
        T = x_t[0].shape[0]
        # MVN_log_likelihood_mapped = jax.vmap(MVN_log_likelihood)
        # log_likelihood = (
        #     jnp.log(MVN_log_likelihood_mapped(jax.vmap(lambda z_t: KalmanFilter.predict(z_t, A, b, Q)[0])((z_t[0][:-1], z_t[1][:-1])), jnp.broadcast_to(Q, (z_t[1][1:].shape)), z_t[0][1:])).sum()
        #     + jnp.log(MVN_log_likelihood_mapped(H@z_t[0], x_t[1], x_t[0])).sum()
        # )
        # return log_likelihood
        log_likelihood = 0
        for t in range(1, T):
            log_likelihood += MVN_log_likelihood(A @ z_t[0][t-1] + b, Q, z_t[0][t])
        for t in range(T):
            log_likelihood += MVN_log_likelihood(H @ z_t[0][t], x_t[1][t], x_t[0][t])

        return log_likelihood

    @staticmethod
    def m_step_update(x: MVN_Type, z_t_sub_1: MVN_Type, p_dist: MVN_Type, f_dist: MVN_Type, q_dist: MVN_Type, J_t: Array, A: Array, b: Array, Q: Array, H: Array, R: Array):
        r"""
        Updates model parameters for kalman filter using the EM algorithm

        Parameter
        ---------
        x: tuple[Array, Array]
            observations represented as (mean, covariance)
        z_t_sub_1: tuple[Array, Array]
            latent prior represented as (mean, covariance)
        p_dist: tuple[Array, Array]
            predicted state distribution represented as (mean, covariance)
        f_dist: tuple[Array, Array]
            updated state distribution represented as (mean, covariance)
        q_dist: tuple[Array, Array]
            smoothed state distribution represented as (mean, covariance)
        K_t: Array
            sequence of kalman gains
        
        
        Returns
        -------
        (H, R, A, Q, (z_t_sub_1)): (Array, Array, Array, Array, tuple[Array, Array])
            updated model parameters
        """
        k = f_dist[0].shape[-1]
        T = f_dist[0].shape[0]

        elementwise_outer = jax.vmap(jnp.outer)

        ### Step 1: calculate cross-covariance recursion, aka 1-lag smoother
        sigma_t_and_t_sub_1_given_T = KalmanFilter.cross_covariance(q_dist, J_t)

        ### Step 2: calculate posterior expectations
        # E[z_t|x_{1:T}] \mu_{t|T}
        mu_t_given_T = q_dist[0]
        
        # E[z_t @ z_t^T | x_{1:T}] = \Sigma_{t|T} + \mu_{t|T} @ \mu_{t|T}^T = P_t
        P_t = q_dist[1] + elementwise_outer(mu_t_given_T, mu_t_given_T)

        # E[z_t @ z_{t-1}^T | x_{1:T}] = \Sigma_{t,t-1|T} + \mu_{t|T}\mu_{t-1|T}^T = P_{t,t-1}
        P_t_and_t_sub_1 = sigma_t_and_t_sub_1_given_T + elementwise_outer(mu_t_given_T[1:], mu_t_given_T[:-1])

        ### Step 3: calculate M step update
        # H_{new} =  (\Sum_{t=1}^T x_t @ \mu_{t|T}^T) @ (\Sum_{t=1}^T P_t)^{-1}
        H_new = elementwise_outer(x[0], mu_t_given_T).sum(axis=0) @ jnp.linalg.inv(P_t.sum(axis=0)) # TODO: optimize with solve)

        # R_{new} = (1/T) * \Sum_{t=1}^T (x_t@x_t^T - H_new @ \mu_{t|T} @ x_t^T)
        R_new = (1/T) * (elementwise_outer(x[0], x[0]) - H_new @ elementwise_outer(mu_t_given_T, x[0])).sum(axis=0)

        # A_{new} = (\Sum_{t=1}^T P_{t,t-1}) @ (\Sum_{t=1}^T P_t)^{-1}
        # A_new = P_t_and_t_sub_1.sum(axis=0) @ jnp.linalg.inv(P_t[:-1].sum(axis=0))
        A_new = jnp.linalg.solve(P_t[:-1].sum(axis=0).T, P_t_and_t_sub_1.sum(axis=0).T).T

        
        # Q_{new} = (1/(T-1)) * (\Sum_{t=1}^T P_t - A_{new} @ \Sum_{t=1}^T P_{t,t-1}^T)
        # OURS
        Q_new = (1/(T-1)) * (P_t[1:].sum(axis=0) - A_new @ P_t_and_t_sub_1.sum(axis=0).T)
        # THEIRS
        # Q_new = jnp.zeros((k, k))
        # for t in range(T - 1):
        #     err = (
        #         q_dist[0][t + 1]
        #         - jnp.dot(A, q_dist[0][t])
        #         - b
        #     )
        #     Vt1t_A = jnp.dot(jnp.concat((jnp.zeros((1,k,k)), sigma_t_and_t_sub_1_given_T))[t+1], A.T)
        #     Q_new += (
        #         jnp.outer(err, err)
        #         + jnp.dot(
        #             A,
        #             jnp.dot(q_dist[1][t], A.T),
        #         )
        #         + q_dist[1][t + 1]
        #         - Vt1t_A
        #         - Vt1t_A.T
        #     )
        # Q_new = (1/(T-1)) * Q_new

        mu_0_new = q_dist[0][0]
        sigma_0_new = q_dist[1][0]

        return H_new, R_new, A_new, Q_new, (mu_0_new, sigma_0_new)

class KalmanFilter_MOTPDA(KalmanFilter):
    @staticmethod
    def run_forward(x: MVN_Type, z_t_sub_1: MVN_Type, A: Array, b: Array, Q: Array, H: Array, mask=None):
        """
        Run Kalman Filter forward pass on a sequence of distributions and return results

        Parameter
        ---------
        x: tuple[Array, Array]
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
        mask: None
            unused

        Returns
        -------
        tuple[tuple[Array, Array], tuple[Array, Array], Array]
            q_dist, p_dist, log_likelihood

        """
        kf_forward = lambda carry, x: KalmanFilter_MOTPDA.forward(carry, x, A, b, Q, H, 0)

        x_1 = (x[0][0], x[1][0])
        q_1, w_k1 = KalmanFilter_MOTPDA.update(z_t_sub_1,
                                  x_1,
                                  H,
                                  0)
        p_1 = z_t_sub_1

        # Log-Likelihood
        # project z_{t|t-1} and z_{t|t} into x (observation) space
        p_1_x_space = (H @ p_1[0], H @ p_1[1] @ H.T)
        q_1_x_space = (H @ q_1[0], H @ q_1[1] @ H.T)

        # get the "effective" observation with moments matching the true GMM
        x_1_effective = MVN_inverse_bayes(p_1_x_space, q_1_x_space)
        # p(x_t) = \int p(z_i|x_{1:i-1}) p(x_i|z_i) dz_i
        log_likelihood1 = MVN_multiply(*p_1_x_space, *x_1_effective)[0]
        
        if x[0].shape[0] > 1:
            _, result = jax.lax.scan(kf_forward, (q_1), (x[0][1:], x[1][1:]))
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
    def forward(carry: MVN_Type, x_t: MVN_Type, A: Array, b: Array, Q: Array, H: Array, mask=0):
        """
        Single iteration of Kalman Filter forward pass
        """
        z_t_sub_1 = carry

        # Prediction
        z_t_given_t_sub_1 = KalmanFilter.predict(z_t_sub_1, A, b, Q)

        # Update
        z_t_given_t, w_ks = KalmanFilter_MOTPDA.update(z_t_given_t_sub_1, x_t, H)

        # Log-Likelihood
        # project z_{t|t-1} and z_{t|t} into x (observation) space
        z_t_given_t_sub_1_x_space = (H @ z_t_given_t_sub_1[0], H @ z_t_given_t_sub_1[1] @ H.T)
        z_t_given_t_x_space = (H @ z_t_given_t[0], H @ z_t_given_t[1] @ H.T)
        # get the "effective" observation with moments matching the true GMM
        x_t_effective = MVN_inverse_bayes(z_t_given_t_sub_1_x_space, z_t_given_t_x_space)
        # p(x_t) = \int p(z_i|x_{1:i-1}) p(x_i|z_i) dz_i
        log_likelihood = MVN_multiply(*z_t_given_t_sub_1_x_space, *x_t_effective)[0]

        return (z_t_given_t), (z_t_given_t, z_t_given_t_sub_1, log_likelihood) # carry, (q_dist, p_dist, log_likelihood)
    
    @staticmethod
    def update(
        z_t_given_t_sub_1: MVN_Type, x_t: MVN_Type, H: Array, mask=0
    ):
        # find GMM that best represents observations
        z_t_given_t_s, w_ks = jax.vmap(lambda z_t: KalmanFilter_MOTPDA.evaluate_observation(z_t, z_t_given_t_sub_1, H))(x_t)
        # normalize list to fix underflow issues
        w_ks = w_ks - jnp.max(w_ks) # jax.lax.stop_gradient(jnp.max(w_ks))
        # jax.debug.print("{x}", x=w_ks)
        # sharpen
        w_ks = w_ks * 10
        # move out of log space
        w_ks = jnp.exp(w_ks)
        
        # w_ks = jnp.array([1, 0]) # hardcode to always use the first observation

        # jax.debug.print("{x} {y}", x=w_ks, y=w_ks / w_ks.sum())
        w_ks = w_ks / w_ks.sum() # jax.lax.stop_gradient(w_ks.sum())
        
        # approximate that with a single moment-matched gaussian
        z_t_given_t = GMM_moment_match(z_t_given_t_s, w_ks)

        return z_t_given_t, w_ks
    
    @staticmethod
    def evaluate_observation(z_t: MVN_Type, z_t_given_t_sub_1: MVN_Type, H: Array):
        z_t_given_t = KalmanFilter.update(z_t_given_t_sub_1, z_t, H, mask=0)
        
        # This is the same as the log likelihood calculation in KalmanFilter.forward
        z_t_given_t_sub_1_x_space = (H @ z_t_given_t_sub_1[0], H @ z_t_given_t_sub_1[1] @ H.T)
        w_k = MVN_multiply(*z_t_given_t_sub_1_x_space, *z_t)[0]

        return z_t_given_t, w_k

class KalmanFilter_MOTCAVI(KalmanFilter):
    @staticmethod
    def run_forward(x: MVN_Type, z_t_sub_1: MVN_Type, A: Array, b: Array, Q: Array, H: Array, beta: Array):
        """
        Run Kalman Filter forward pass on a sequence of distributions and return results

        Parameter
        ---------
        x: tuple[Array, Array]
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
        beta: Array
            observation weights

        Returns
        -------
        tuple[tuple[Array, Array], tuple[Array, Array], Array]
            q_dist, p_dist, log_likelihood

        """
        kf_forward = lambda carry, X: KalmanFilter_MOTCAVI.forward(carry, X[0], A, b, Q, H, X[1])

        x_1 = (x[0][0], x[1][0])
        beta_1 = beta[0]
        q_1 = KalmanFilter_MOTCAVI.update(z_t_sub_1,
                                  x_1,
                                  H,
                                  beta_1)
        p_1 = z_t_sub_1

        # Log-Likelihood
        # project z_{t|t-1} and z_{t|t} into x (observation) space
        p_1_x_space = (H @ p_1[0], H @ p_1[1] @ H.T)
        q_1_x_space = (H @ q_1[0], H @ q_1[1] @ H.T)

        # get the "effective" observation with moments matching the true GMM
        x_1_effective = MVN_inverse_bayes(p_1_x_space, q_1_x_space)
        # p(x_t) = \int p(z_i|x_{1:i-1}) p(x_i|z_i) dz_i
        log_likelihood1 = MVN_multiply(*p_1_x_space, *x_1_effective)[0]
        
        if x[0].shape[0] > 1:
            _, result = jax.lax.scan(kf_forward, (q_1), ((x[0][1:], x[1][1:]), beta[1:]))
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
    def forward(carry: MVN_Type, x_t: MVN_Type, A: Array, b: Array, Q: Array, H: Array, beta: Array):
        """
        Single iteration of Kalman Filter forward pass
        """
        z_t_sub_1 = carry

        # Prediction
        z_t_given_t_sub_1 = KalmanFilter.predict(z_t_sub_1, A, b, Q)

        # Update
        z_t_given_t = KalmanFilter_MOTCAVI.update(z_t_given_t_sub_1, x_t, H, beta)

        # Log-Likelihood
        # project z_{t|t-1} and z_{t|t} into x (observation) space
        z_t_given_t_sub_1_x_space = (H @ z_t_given_t_sub_1[0], H @ z_t_given_t_sub_1[1] @ H.T)
        z_t_given_t_x_space = (H @ z_t_given_t[0], H @ z_t_given_t[1] @ H.T)
        # get the "effective" observation with moments matching the true GMM
        x_t_effective = MVN_inverse_bayes(z_t_given_t_sub_1_x_space, z_t_given_t_x_space)
        # p(x_t) = \int p(z_i|x_{1:i-1}) p(x_i|z_i) dz_i
        log_likelihood = MVN_multiply(*z_t_given_t_sub_1_x_space, *x_t_effective)[0]

        return (z_t_given_t), (z_t_given_t, z_t_given_t_sub_1, log_likelihood) # carry, (q_dist, p_dist, log_likelihood)
    
    @staticmethod
    def update(
        z_t_given_t_sub_1: MVN_Type, x_t: MVN_Type, H: Array, beta: Array
    ):
        
        # hat_x_t = H @ z_t|t-1
        hat_x_t = H @ z_t_given_t_sub_1[0]

        # S_t = H @ P_t|t-1 @ H^T + R
        # S_t = H @ z_t_given_t_sub_1[1] @ H.T + x_t[1]
        S_t = H @ z_t_given_t_sub_1[1] @ H.T + (1/(1-beta[0])) * x_t[1][0] # R is a model parameter, so we get it from the first observation

        # K_t = P_t|t-1 @ H^T @ S^-1
        # K_t = z_t_given_t_sub_1[1] @ H.T @ jnp.linalg.inv(S_t)
        K_t = z_t_given_t_sub_1[1] @ (jnp.linalg.solve(S_t.T, H)).T # (AB^-1)^T = (B^-1)^T A^T = (B^T)^-1 A^T

        # z_t|t = z_t|t-1 + K_t @ (\Sum(\beta^(k)_t x^(k)_t) / (1-\beta^(0)) - @ z_t|t-1)
        mu = z_t_given_t_sub_1[0] + K_t @ ((beta[1:]@x_t[0]).sum(axis=-1) / (1-beta[0]) - hat_x_t)

        # P_t|t = P_t|t-1 - K_t @ S @ K_t^T
        sigma = z_t_given_t_sub_1[1] - K_t @ S_t @ K_t.T

        return (mu, sigma)
