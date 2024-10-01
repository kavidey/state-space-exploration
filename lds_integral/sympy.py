# %%
from sympy import symbols, sqrt, pi, exp, Pow, log, integrate
# %%
def gaussian(x, mu, sigma):
    return (1/(sigma*sqrt(2*pi))) * exp(-(1/2)*Pow((x-mu)/sigma, 2))

def kl_divergence(mu1, sigma1, mu2, sigma2):
    return log(sigma2/sigma1) + (Pow(sigma1, 2)+Pow(mu1 - mu2, 2))/(2*Pow(sigma2, 2)) - 1/2

def predict(z, A, b, P, Q):
    P_t = Pow(A, 2)*P + Q
    z_t = A*z + b
    return z_t, P_t

def update(z, z_t_sub_1, P, H, R):
    K = (P*H) / (Pow(H,2)*P + R)
    z_t = z_t_sub_1 + K * (z - H*z)
    P_t = (1 - K * H) * P
    return z_t, P_t

# %%
A, b, H, Q, R = symbols("A b H Q R")

P_t_sub_1_given_t_sub_1 = symbols("P_{t-1|t-1}")
z_t_sub_1_given_t_sub_1 = symbols("z_{t-1|t-1}")
z_t = symbols("z_t")
# %%
z_t_given_t_sub_1, P_t_given_t_sub_1 = predict(z_t_sub_1_given_t_sub_1, A, b, P_t_sub_1_given_t_sub_1, Q)
z_t_given_t, P_t_given_t = update(z_t, z_t_given_t_sub_1, P_t_given_t_sub_1, H, R)
# %%
z_t_sub_1 = symbols("z_{t-1}")

i = integrate(gaussian(z_t_sub_1, z_t_sub_1_given_t_sub_1, P_t_given_t_sub_1) * kl_divergence(z_t_given_t, P_t_given_t, z_t_given_t_sub_1, P_t_given_t_sub_1), z_t_sub_1)
i
# %%
