# %%
import jax
import jax.numpy as jnp
import numpy as np
# %%
x = jnp.arange(10)
print(x)
# %%
long_vector = jnp.arange(1e7)
%timeit jnp.dot(long_vector, long_vector).block_until_ready()
# %%
def sum_of_squares(x):
    return jnp.sum(x**2)

sum_of_squares_dx = jax.grad(sum_of_squares)
x = jnp.asarray([1.0, 2.0, 3.0, 4.0])

print(sum_of_squares(x))
print(sum_of_squares_dx(x))
# %%
def sum_squared_error(x, y):
    return jnp.sum((x-y)**2)

sum_squared_error_dx = jax.grad(sum_squared_error)
y = jnp.asarray([1.1, 2.1, 3.1, 4.1])
print(sum_squared_error_dx(x, y))
# %%
jax.grad(sum_squared_error, argnums=(0, 1))(x, y)
# %%
jax.value_and_grad(sum_squared_error)(x, y)
# %%
def squared_error_with_aux(x, y):
    return sum_squared_error(x, y), x-y

jax.grad(squared_error_with_aux, has_aux=True)(x, y)
# %%
