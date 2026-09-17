"""
Well-tempered metadynamics of alanine tetrapeptide in water, biased on the per-atom
delta entropy of the solute atoms.
"""

import numpy as np
import openmm
import openmm.app as app

from openmm import unit
from openff.units import unit as off_unit
from openff.toolkit import Molecule, ForceField
from openff.interchange import Interchange
from openff.interchange.components._packmol import pack_box, UNIT_CUBE

import pysages
from pysages.colvars.core import CollectiveVariable, multicomponent
from pysages.methods import Metadynamics, MetaDLogger

from ase.io import read
from utils import *
import argparse
import random
import os
import sys


def get_args(argv):
    """Process the command-line arguments to this script."""

    available_args = [
        ("time-steps", "t", int, 1e7, "Number of simulation steps"),
        ("log-steps", "l", int, 1000, "Number of simulation steps for logging"),
        ("temperature", "T", int, 300, "Temperature of the simulation"),
        ("height", "y", float, 1.2, "Height of gaussian bias"),
        ("width", "w", float, 16.0, "Width of gaussian bias"),
        ("deltaT", "dT", float, 5000, "DeltaT for well-tempered metadynamics"),
        ("stride", "s", int, 100, "Frequency of adding gaussian bias"),
        ("bandwidth", "b", float, 1.0, "Bandwidth for dH calculation"),
        ("n-water", "n", int, 100, "Number of water molecules packed in the box"),
        ("box-length", "L", float, 2.0, "Box length in nm"),
        ("reference", "r", str, "refer/unbiased.pdb", "Trajectory holding the reference frames"),
        ("n-reference", "N", int, 100, "Number of frames sampled from the reference trajectory"),
        ("unbiased", "u", bool, True, "Run without bias, to generate the reference trajectory"),
    ]
    parser = argparse.ArgumentParser(description="Example script to run pysages with openmm")

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

print(f'[setup] packing {args.n_water} waters around the solute in a {args.box_length} nm box', flush=True)

alanine_tetrapeptide = 'CC(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC'
water = 'O'
force_field = 'openff-2.0.0.offxml'

ala = Molecule.from_smiles(alanine_tetrapeptide, allow_undefined_stereo=True)
water = Molecule.from_smiles(water, allow_undefined_stereo=True)
ala.generate_conformers(n_conformers=1)

# the 0.9 nm nonbonded cutoff has to stay below half the periodic box length
top = pack_box(
    [ala, water],
    number_of_copies=[1, args.n_water],
    box_vectors=args.box_length * UNIT_CUBE * off_unit.nanometer,
)

ff = ForceField(force_field)
inter = Interchange.from_smirnoff(force_field=ff, topology=top)
print(f'[setup] {top.n_atoms} atoms in the box, {len(ala.atoms)} of them in the solute', flush=True)

# the only atoms the descriptor sees (the solute comes first in the topology)
protein_indices = np.arange(len(ala.atoms))
bandwidth = args.bandwidth

# box vectors in nm, as rows. The box is cubic and the runs are NVT, so it never changes
# and the descriptor can hold on to it as a constant.
cell = jnp.eye(3) * args.box_length

if not args.unbiased:
    print(f'[reference] reading {args.reference}', flush=True)
    refer = read(args.reference, index=':')
    random.seed(42)
    random.shuffle(refer)
    refer = random.sample(refer, args.n_reference)

    Mi = []
    max_extent = 0.0
    for atoms in refer:
        positions = atoms.get_positions() * unit.angstrom
        # water is masked out: the descriptor is built from the solute atoms alone
        solute = positions.value_in_unit(unit.nanometer)[protein_indices]
        mi = get_descriptor(solute, cell=cell)
        # unwrapped extent of the solute, to check the minimum image convention below
        max_extent = max(max_extent, float(jnp.max(distance_matrix(solute, solute))))
        Mi.append(mi)
    Mi = jnp.vstack(Mi)
    print(f'[reference] {len(refer)} frames sampled, reference descriptor {Mi.shape}', flush=True)

    # the minimum image convention only holds below half the shortest box length
    if max_extent > args.box_length / 2:
        print(f'[warning] the solute spans up to {max_extent:.2f} nm in the reference, beyond half '
              f'the {args.box_length} nm box -- the minimum image convention folds those distances '
              f'back into the box. Run with a larger --box-length.', flush=True)


def collective_variable(rs):
    """
    ``rs`` holds the positions of the solute atoms only -- the water is masked out before
    the descriptor is built -- so the per-atom delta entropy is a purely intramolecular
    quantity, exactly as in the vacuum runs. The pair distances respect the periodic box,
    so a solute that straddles a boundary is still described correctly.
    """
    M = get_descriptor(rs, cell=cell)
    return delta_entropy(M, Mi, bandwidth)


@multicomponent
class EntropyCV(CollectiveVariable):
    """
    ``indices`` are the atoms whose positions are passed to the descriptor. Only the
    solute atoms are handed over, so the water never enters the descriptor.
    """

    @property
    def function(self):
        return lambda rs: collective_variable(rs)


class CustomMetaDLogger(MetaDLogger):
    def __init__(self, hills_file, log_period):
        super().__init__(hills_file, log_period)

    def __call__(self, snapshot, state, timestep):
        """
        Implements the logging itself. Interface as expected for Callbacks.
        """
        if self.counter >= self.log_period and self.counter % self.log_period == 0:
            idx = state.idx - 1 if state.idx > 0 else 0
            self.save_hills(state.centers[idx], state.sigmas, state.heights[idx])
            print(f'[run] {self.counter} steps', flush=True)

        self.counter += 1


def generate_simulation(file_name, output_stride, temperature, inter=inter):
    # log info
    trj_freq = output_stride
    data_freq = output_stride

    # Integration options
    time_step = 1 * unit.femtoseconds
    temperature = temperature * unit.kelvin
    friction = 10 / unit.picoseconds

    integrator = openmm.LangevinIntegrator(temperature, friction, time_step)
    simulation = inter.to_openmm_simulation(integrator=integrator)
    simulation.context.setVelocitiesToTemperature(temperature)

    pdb_reporter = app.PDBReporter(f"{file_name}.pdb", trj_freq)
    state_data_reporter = app.StateDataReporter(
        f"{file_name}.csv",
        data_freq,
        step=True,
        potentialEnergy=True,
        kineticEnergy=True,
        temperature=True,
    )
    simulation.reporters.append(pdb_reporter)
    simulation.reporters.append(state_data_reporter)
    print(f'[setup] openmm platform: {simulation.context.getPlatform().getName()}', flush=True)

    return simulation


def main(argv):
    args = get_args(argv)

    # parameters
    height = args.height # kj/mol
    temperature = args.temperature #K
    deltaT = args.deltaT
    stride = args.stride
    timesteps = args.time_steps
    ngauss = timesteps // stride
    output_stride = args.log_steps

    if args.unbiased:
        prefix = 'refer/'
        file_name = 'unbiased'
    else:
        prefix = 'results/'

    os.makedirs(prefix, exist_ok=True)

    if args.unbiased:
        print(f'[run] unbiased simulation, {timesteps} steps at {temperature} K -> {prefix}{file_name}.pdb', flush=True)
        simulation = generate_simulation(f'{prefix}{file_name}', output_stride, temperature)
        simulation.step(timesteps)
        print('[run] finished', flush=True)
        return

    cvs = [EntropyCV(protein_indices)]
    sigma = [args.width] * len(cvs)
    file_name = f'{height}_{sigma[0]}_{bandwidth}_{stride}_{deltaT}_mask'

    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kB = kB.value_in_unit(unit.kilojoules_per_mole / unit.kelvin)

    method = Metadynamics(cvs=cvs, height=height, sigma=sigma, stride=stride, ngaussians=ngauss, deltaT=deltaT, kB=kB)
    callback = CustomMetaDLogger(f'{prefix}{file_name}.dat', f'{prefix}{file_name}', output_stride)

    context_args = {
        'file_name': f'{prefix}{file_name}',
        'output_stride': output_stride,
        'temperature': temperature,
    }
    print(f'[run] metadynamics, {timesteps} steps, {ngauss} gaussians of height {height} every {stride} steps', flush=True)
    print(f'[run] writing to {prefix}{file_name}.{{dat,pdb,csv}} (first step includes the jit compilation)', flush=True)
    pysages.run(method,
                generate_simulation,
                timesteps,
                callback,
                context_args=context_args)
    print('[run] finished', flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
