import jax.numpy as jnp
from jax import jit
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.visualize import view
from functools import partial
from ase.neighborlist import NewPrimitiveNeighborList


DEFAULT_PARAMS = jnp.array([(1,1),(1,2),(1,3),(1,4)])
DEFAULT_CUTOFF = 5
DEFAULT_K = 16
DEFAULT_TOLERANCE = 40
DEFAULT_PBC = jnp.array([1,1,1])


@jit
def g1_embed(distances: jnp.ndarray, r_cut: float):
    """G1 symmetry function using neighbor list"""
    mask = (distances > 1e-6) & (distances < r_cut)
    g1_kernel = 0.5 * (jnp.cos(jnp.pi * distances / r_cut) + 1) * mask
    return g1_kernel


@jit  
def g2_embed(distances: jnp.ndarray, r_cut: float, params: jnp.ndarray = DEFAULT_PARAMS):
    """G2 symmetry function using neighbor list"""
    g1_kernel = g1_embed(distances, r_cut)
    
    G2 = []
    for eta, Rs in params:
        g2_kernel = jnp.exp(-eta * (distances - Rs)**2) * g1_kernel
        G2.append(g2_kernel)
    return jnp.stack(G2, axis=-1)


@jit
def get_distance_matrix(positions, cell, cutoff=DEFAULT_CUTOFF):
    # Pairwise displacement vectors (n_atoms, n_atoms, 3)
    # positions = wrap_positions(positions, cell)
    dr = positions[:, None, :] - positions[None, :, :]
    
    # Apply minimum image convention for periodic boundaries and wrap displacement for each dimension
    for dim in range(3):
        dr_dim = dr[:, :, dim]
        cell_length = jnp.linalg.norm(cell[dim])
        dr = dr.at[:, :, dim].set(
            dr_dim - jnp.round(dr_dim / cell_length) * cell_length
        )
    
    # Add small epsilon to avoid sqrt(0) and ensure gradient stability
    distances_sq = jnp.sum(dr * dr, axis=2)
    distances = jnp.sqrt(distances_sq + 1e-16)
    
    return distances


# def get_neighborlist(positions: jnp.ndarray, 
#                      cell: jnp.ndarray, 
#                      cutoff=DEFAULT_CUTOFF, 
#                      k=DEFAULT_K):
#     atoms = Atoms(positions=positions, cell=cell, pbc=DEFAULT_PBC)
#     i, j, d = neighbor_list('ijd', a=atoms, cutoff=cutoff)
#     matrix = jnp.ones([len(positions)]*2) * 100
#     matrix = matrix.at[i, j].set(d)
#     sorted_indices = jnp.argsort(matrix, axis=1)
#     k_nearest_indices = sorted_indices[:, :k]
#     i = jnp.repeat(jnp.arange(len(positions)), k)
#     j = k_nearest_indices.flatten()
#     return jnp.array([i,j])


# def get_neighborlist(positions: jnp.ndarray, 
#                      cell: jnp.ndarray, 
#                      cutoff=DEFAULT_CUTOFF, 
#                      k=DEFAULT_K):
#     # atoms = Atoms(positions=positions, cell=cell, pbc=DEFAULT_PBC)
#     # i, j, d = neighbor_list('ijd', a=atoms, cutoff=cutoff)
    
#     # i = jnp.array(i, dtype=jnp.int32)
#     # j = jnp.array(j, dtype=jnp.int32)
#     # d = jnp.array(d, dtype=jnp.float32)
    
#     # n = len(positions)
#     # matrix = jnp.full((n, n), jnp.inf, dtype=jnp.float32)
#     # matrix = matrix.at[i, j].set(d)
#     matrix = get_distance_matrix(positions, cell)
    
#     k_nearest_indices = jnp.argsort(matrix, axis=1)[:, 1:k+1]
    
#     i_out = jnp.repeat(jnp.arange(len(positions), dtype=jnp.int32), k)
#     j_out = k_nearest_indices.ravel()
    
#     return jnp.stack([i_out, j_out])


# def get_neighborlist(positions: jnp.ndarray, 
#                      cell: jnp.ndarray, 
#                      cutoff=DEFAULT_CUTOFF):
#     atoms = Atoms(positions=positions, cell=cell, pbc=DEFAULT_PBC)
#     i, j = neighbor_list('ij', a=atoms, cutoff=cutoff)
#     adj_matrix = jnp.zeros((len(positions), len(positions)), dtype=int)
#     adj_matrix = adj_matrix.at[i, j].set(1)
#     return i, j


@jit
def acsf_embed(distances: jnp.ndarray,
                r_cut: float = DEFAULT_CUTOFF,
                params: jnp.ndarray = DEFAULT_PARAMS,
                ):    
    # Compute embeddings for all pairs
    G1_pairs = g1_embed(distances, r_cut)  # (n_atoms, n_atoms)
    G2_pairs = g2_embed(distances, r_cut, params)  # (n_atoms, n_atoms, n_params)

    # Sum over neighbors (axis 1 = j index in pair i->j)
    G1_atoms = jnp.sum(G1_pairs, axis=1)  # (n_atoms,)
    G2_atoms = jnp.sum(G2_pairs, axis=1)  # (n_atoms, n_params)
    
    return jnp.column_stack([G1_atoms, G2_atoms])


# def acsf_embed(nl: jnp.ndarray,
#                 d: jnp.ndarray,
#                 n_atoms: int,
#                 r_cut: int = DEFAULT_CUTOFF,
#                 params: jnp.ndarray = DEFAULT_PARAMS,
#                 ):
#     """ACSF embedding using neighbor list and distances"""
#     i_indices = nl[0,:]

#     G1_pairs = g1_embed(d, r_cut)  # (n_pairs,)
#     G2_pairs = g2_embed(d, r_cut, params)  # (n_pairs, n_params)

#     G1_atoms = jnp.zeros(n_atoms)
#     G1_atoms = G1_atoms.at[i_indices].add(G1_pairs)

#     G2_atoms = jnp.zeros((n_atoms, G2_pairs.shape[1]))
#     G2_atoms = G2_atoms.at[i_indices].add(G2_pairs)

#     return jnp.column_stack([G1_atoms, G2_atoms])
