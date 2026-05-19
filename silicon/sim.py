# Standard library
import argparse
import random
import sys

# JAX
from jax.experimental import io_callback

# LAMMPS
from lammps import lammps

# ASE
from ase.io import read
from ase.units import kB

# PySAGES
import pysages
from pysages.backends.lammps import get_cache, set_cache_updater
from pysages.colvars import Distance
from pysages.colvars.core import CollectiveVariable, multicomponent
from pysages.methods import Metadynamics, MetaDLogger

# Local
from utils import *
from descriptor import *


def initialize_cache(data_file):
    cache = get_cache()
    cache.clear()
    atoms = read(data_file, format='lammps-data', atom_style='atomic')
    atoms.wrap()
    atoms.pbc = [1,1,1]
    cell = jnp.array(atoms.get_cell().array)
    cache.data = {
        'cell': cell,
        'cell_shape': cell.shape,
        'cell_dtype': cell.dtype,
        'n_atoms': len(atoms),
    }
    return cache


def generate_simulation(file_name, temperature, pressure, output_stride, args=""):
    context = lammps(cmdargs=args.split())
    context.command(f'variable T2 equal {temperature}')
    context.command(f'variable P equal {pressure}')
    context.file('in.lmp')
    context.command(f'log {file_name}.log')
    context.command(f'dump 1 all custom {output_stride} {file_name}.traj element x y z id')
    context.command(f'dump_modify 1 element Si')
    return context


def get_args(argv):
    """Process the command-line arguments to this script."""

    available_args = [
        ("time-steps", "t", int, 1e6, "Number of simulation steps"),
        ("kokkos", "k", bool, True, "Whether to use Kokkos acceleration"),
        ("log-steps", "l", int, 100, "Number of simulation steps for logging"),
        ("temperature", "T", int, 1250, "Temperature of the simulation"),
        ("pressure", "p", int, 0, "Pressure of the simulation"),
        ("height", "y", float, 0.1, "Height of gaussian bias"),
        ("width", "w", float, 16, "Width of gaussian bias"),
        ("deltaT", "dT", float, None, "DeltaT for well-tempered metadynamics"),
        ("stride", "s", int, 100, "Frequency of adding gaussian bias"),
        ("bandwidth", "b", float, 0.035, "Bandwidth for dH calculation"),
    ]
    parser = argparse.ArgumentParser(description="Example script to run pysages with lammps")

    for name, short, T, val, doc in available_args:
        if T is bool:
            action = "store_" + str(val).lower()
            parser.add_argument("--" + name, "-" + short, action=action, help=doc)
        else:
            convert = (lambda x: int(float(x))) if T is int else T
            if val is None:
                parser.add_argument("--" + name, "-" + short, type=convert, help=doc)
            else:
                parser.add_argument("--" + name, "-" + short, type=convert, default=T(val), help=doc)

    return parser.parse_args(argv)


args = get_args(sys.argv[1:])
refer = read(f'results_unbiased/Si_{args.temperature}K_{args.pressure}bar_nvt.traj', index=':', format='lammps-dump-text')[50:]
random.shuffle(refer)
refer = random.sample(refer, 1)

# Mi = []
# for atoms in refer:
#     pos = jnp.array(atoms.get_positions())
#     cell = jnp.array(atoms.get_cell())
#     dm = get_distance_matrix(pos, cell)
#     mi = acsf_embed(dm)
#     Mi.append(mi)
# Mi = jnp.vstack(Mi)
bandwidth = args.bandwidth


def update_cache(snapshot):
    cell = snapshot.box[0].copy()
    return {
        'cell': cell,
        'cell_shape': cell.shape,
        'cell_dtype': cell.dtype,
        }

def collective_variable(rs):
    """Function that calculate delta entropy"""
    global bandwidth
    
    cache = get_cache()
    cell_shape = cache.get('cell_shape')
    cell_dtype = cache.get('cell_dtype')

    cell = io_callback(
        lambda: get_cache().get('cell'),
        jnp.zeros(cell_shape, dtype=cell_dtype)
    )

    dm = get_distance_matrix(rs, cell)
    M = acsf_embed(dm)
    dH = delta_entropy(M, M, bandwidth)
    return dH


@multicomponent
class EntropyCV(CollectiveVariable):
    @property
    def function(self): 
        return lambda rs: collective_variable(rs)
    

class CustomMetaDLogger(MetaDLogger):
    def __init__(self, hills_file, forces_file, log_period):
        super().__init__(hills_file, log_period)
        self.forces_file = forces_file


    def save_forces(self, forces, bias):
        with open(f'{self.forces_file}.frc', "a+", encoding="utf8") as f:
            f.write(str(self.counter) + "\t")
            f.write("\t".join(map(str, forces.flatten())) + "\n")
        with open(f'{self.forces_file}.bias', "a+", encoding="utf8") as f:
            f.write(str(self.counter) + "\t")
            f.write("\t".join(map(str, bias.flatten())) + "\n")


    def __call__(self, snapshot, state, timestep):
        """
        Implements the logging itself. Interface as expected for Callbacks.
        """
        if self.counter >= self.log_period and self.counter % self.log_period == 0:
            idx = state.idx - 1 if state.idx > 0 else 0
            self.save_hills(state.centers[idx], state.sigmas, state.heights[idx])
            #self.save_forces(snapshot.forces, state.bias)

        self.counter += 1   


def main(argv):
    args = get_args(argv)

    # parameters                             
    height = args.height # eV unit metal
    temperature = args.temperature #K
    pressure = args.pressure #bar

    set_cache_updater(update_cache)
    # input file for lammps simulation read_data command
    cache = initialize_cache(f'structure/Si_{temperature}K_{pressure}bar_nvt.lmp')
    cvs = [EntropyCV(np.arange(cache.get('n_atoms')))]

    sigma = [args.width] * len(cvs)
    deltaT = args.deltaT
    stride = args.stride
    timesteps = args.time_steps
    ngauss = timesteps // stride
    prefix = 'results_deltaH_self/'
    output_stride = args.log_steps

    context_args = {"output_stride": args.log_steps}

    if prefix == 'results_unbiased/':
        file_name = 'unbiased'
    else:
        file_name = f'{height}_{sigma[0]}_{bandwidth}_{stride}_{temperature}_{deltaT}'

    method = Metadynamics(cvs=cvs, height=height, sigma=sigma, stride=stride, ngaussians=ngauss, deltaT=deltaT, kB=kB)
    callback = CustomMetaDLogger(f'{prefix}{file_name}.dat', f'{prefix}{file_name}', output_stride)

    context_args['file_name'] = f'{prefix}{file_name}'
    context_args['temperature'] = temperature
    context_args['pressure'] = pressure
    context_args["args"] = "-k on g 1 -sf kk -pk kokkos newton on neigh half"
    pysages.run(method, 
                generate_simulation, 
                timesteps, 
                callback, 
                context_args=context_args)


if __name__ == "__main__":
    main(sys.argv[1:])
