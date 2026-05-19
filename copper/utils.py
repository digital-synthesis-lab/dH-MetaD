import jax.numpy as jnp
import math
from typing import List, Tuple, Union
import numpy as np
from ase import Atoms
from jax import jit
import jax.lax as lax
import jax


DEFAULT_BANDWIDTH=1
DEFAULT_BATCH = 10000
pi = jnp.pi


# def distance_matrix(A, B):
#     """
#     Calculate distance matrix
#     """
#     M = A @ B.T
#     r = jnp.diag(M).reshape(-1,1)
#     D = r + r.T - 2 * M
#     return D


@jit
def distance_matrix(A: jnp.ndarray, B: jnp.ndarray):
    """Fully vectorized continuous distance calculation."""
    norm_A = jnp.sum(A**2, axis=1, keepdims=True)  # (N, 1)
    norm_B = jnp.sum(B**2, axis=1, keepdims=True)  # (M, 1)
    
    dot_product = A @ B.T  # (N, M)
    squared_dist = norm_A + norm_B.T - 2.0 * dot_product  # (N, M)
    squared_dist = jnp.maximum(squared_dist, 1e-8)
    return jnp.sqrt(squared_dist)


@jit
def distances(positions: jnp.ndarray, neighbor_list: jnp.ndarray, cell: jnp.ndarray):
    i_indices = neighbor_list[0,:]
    j_indices = neighbor_list[1,:]

    i_pos = positions[i_indices]
    j_pos = positions[j_indices]

    displacement = j_pos - i_pos
    
    inv_cell = jnp.linalg.inv(cell)
    frac_displacement = displacement @ inv_cell.T
    frac_displacement = frac_displacement - jnp.round(frac_displacement)
    
    cart_displacement = frac_displacement @ cell.T
    distances = jnp.linalg.norm(cart_displacement, axis=1)
    
    return distances


@jit
def eam_function(dset:jnp.ndarray, params: jnp.ndarray):
    """
    Calculate eam embedding matrix
    """
    R = []
    for miu, sigma in params:
        kernel = jnp.exp((dset-miu)**2/sigma**2)
        col = jnp.sum(kernel, axis=1) / dset.shape[1]
        R.append(col)
    return jnp.column_stack(R)


@jit
def normalize(dset):
    """
    Normalize dset along rows
    """
    norm = jnp.linalg.norm(dset, axis=1, keepdims=True)
    return dset / norm


@jit
def sumexp(X: jnp.ndarray):
    return jnp.sum(jnp.exp(X), axis=1)


# @jit
# def kernel_sum(
#     x: jnp.ndarray,
#     y: jnp.ndarray,
#     h: float = DEFAULT_BANDWIDTH,
# ):
#     z = distance_matrix(x, y)
#     z = z / h
#     p_x = sumexp(-0.5 * (z**2))
#     return p_x


@jit
def kernel_sum(
    x: jnp.ndarray,
    y: jnp.ndarray,
    h: float = DEFAULT_BANDWIDTH,
    batch_size: int = DEFAULT_BATCH,
):
    M = x.shape[0]
    max_step_x = math.ceil(M / batch_size)
    N = y.shape[0]
    max_step_y = math.ceil(N / batch_size)

    p_x = jnp.zeros(M, dtype=x.dtype)

    for step_x in range(0, max_step_x):
        i = step_x * batch_size
        imax = min(i + batch_size, M)
        x_batch = x[i:imax]
        
        for step_y in range(0, max_step_y):
            j = step_y * batch_size
            jmax = min(j + batch_size, N)
            y_batch = y[j:jmax]

            z = distance_matrix(x_batch, y_batch)
            z = z / h
            z = sumexp(-0.5 * (z**2))

            p_x = p_x.at[i:imax].add(z)

    return p_x


@jit 
def delta_entropy(
    x: jnp.ndarray,
    y: jnp.ndarray,
    h: float = DEFAULT_BANDWIDTH,
):
    """
    x is the dataset with shape (N, k) and y is the reference with shape (M, k)
    """
    p_xy = kernel_sum(x, y, h=h)
    return -jnp.log(p_xy)


@jit
def similarity(
    x: jnp.ndarray,
    y: jnp.ndarray,
    h: float = DEFAULT_BANDWIDTH,
):
    """
    x is the dataset with shape (N, k) and y is the reference with shape (M, k)
    """
    dH = delta_entropy(x, y, h=h)
    return jnp.mean(dH)


### Analyzing Tools ###

def dihedrals(traj: List[Atoms], indices_list: list[Tuple]):
    """
    Indices should be a list of tuple that has 4 components
    """
    result_list = []
    for atoms in traj:
        result = []
        for indices in indices_list:
            i, j, k, l = indices
            pos_i = atoms.positions[i]
            pos_j = atoms.positions[j]
            pos_k = atoms.positions[k]
            pos_l = atoms.positions[l]

            vec_ij = pos_j - pos_i  # i -> j
            vec_jk = pos_k - pos_j  # j -> k
            vec_kl = pos_l - pos_k  # k -> l

            normal1 = np.cross(vec_ij, vec_jk)
            normal2 = np.cross(vec_jk, vec_kl)

            normal1_norm = normal1 / np.linalg.norm(normal1)
            normal2_norm = normal2 / np.linalg.norm(normal2)

            value = np.dot(normal1_norm, normal2_norm)
            cross_prod = np.cross(normal1_norm, normal2_norm)
            sign = np.dot(cross_prod, vec_jk / np.linalg.norm(vec_jk))

            value = np.clip(value, -1.0, 1.0)
            angle = np.arccos(value)

            if sign < 0:
                angle = -angle
            
            result.append(angle)
        result_list.append(result)
    return np.array(result_list)