import jax.numpy as jnp
from typing import List, Tuple, Union
from openff.interchange import Interchange
import numpy as np
from ase import Atoms
from jax import jit


DEFAULT_BANDWIDTH=0.015
DEFAULT_PARAMS=jnp.array([(1,0.1),(1,0.2),(1,0.3),(1,0.4)])
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
    squared_dist = jnp.maximum(squared_dist, 1e-16)
    return jnp.sqrt(squared_dist)


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
def g1_embed(dset: jnp.ndarray):
    kernel = 0.5*(jnp.cos(pi*dset)+1)
    return jnp.sum(kernel, axis=1)


@jit
def g2_embed(dset: jnp.ndarray, params: jnp.ndarray):
    """
    Parameters should includes a list of (eta, Rs)
    """
    G2 = []
    g1_kernel = 0.5*(jnp.cos(pi*dset)+1)
    for eta, Rs in params:
        kernel = jnp.exp(-eta*(dset-Rs)**2) * g1_kernel
        G2.append(jnp.sum(kernel, axis=1))
    return jnp.column_stack(G2)


@jit
def acsf_embed(dset: jnp.ndarray, params: jnp.ndarray):
    """
    Parameters should includes a list of (eta, Rs) that converted to jax ndarray
    """    
    G1 = g1_embed(dset)
    G2 = g2_embed(dset, params)
    return jnp.column_stack([G1, G2])


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


@jit
def kernel_sum(
    x: jnp.ndarray,
    y: jnp.ndarray,
    h: float = DEFAULT_BANDWIDTH,
):
    z = distance_matrix(x, y)
    z = z / h
    p_x = sumexp(-0.5 * (z**2))

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
 

@jit
def get_descriptor(r: jnp.ndarray, params: jnp.ndarray = DEFAULT_PARAMS):
    """
    Calculate descriptor based on atom positions
    """
    D = distance_matrix(r, r)
    M = acsf_embed(D, params)
    return M


def center(interchange: Interchange):
    positions = interchange.get_positions()
    center_of_mass = np.mean(positions, axis=0)

    box = interchange.box
    center_of_box = np.mean(box, axis=0)

    displacement = center_of_box - center_of_mass
    new_positions = positions + displacement
    interchange.positions = new_positions
    return interchange


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


def nans_at_jumps(x, y, threshold=5.0):
    """
    Insert NaN values where jumps occur. Matplotlib won't draw lines through NaNs.
    This is the fastest approach for large datasets.
    """
    # Calculate differences
    dx = np.abs(np.diff(x))
    dy = np.abs(np.diff(y))
    
    # Find jump locations
    jumps = (dx > threshold) | (dy > threshold)
    jump_indices = np.where(jumps)[0] + 1
    
    # Insert NaNs at jump positions
    x_with_nans = np.insert(x, jump_indices, np.nan)
    y_with_nans = np.insert(y, jump_indices, np.nan)
    
    return x_with_nans, y_with_nans


