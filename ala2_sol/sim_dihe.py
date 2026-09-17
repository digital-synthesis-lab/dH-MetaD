"""
Well-tempered metadynamics of alanine dipeptide in water, biased on the backbone
dihedral angles (phi, psi) of the solute.
"""

import numpy as np
import openmm
import openmm.app as app

from math import pi
from openmm import unit
from openff.units import unit as off_unit
from openff.toolkit import Molecule, ForceField
from openff.interchange import Interchange
from openff.interchange.components._packmol import pack_box, UNIT_CUBE

import pysages
from pysages.approxfun import compute_mesh
from pysages.colvars import DihedralAngle
from pysages.methods import Metadynamics, MetaDLogger

import argparse
import os
import sys


# backbone dihedrals of the solute, in the atom order of the OpenFF topology
# (0:C 1:C 2:O 3:N 4:CA 5:CB 6:C 7:O 8:N 9:C, hydrogens after the heavy atoms)
PHI = (1, 3, 4, 6)
PSI = (3, 4, 6, 8)
DIHEDRALS = [PHI, PSI]


def get_args(argv):
    """Process the command-line arguments to this script."""

    available_args = [
        ("time-steps", "t", int, 1e7, "Number of simulation steps"),
        ("log-steps", "l", int, 1000, "Number of simulation steps for logging"),
        ("temperature", "T", int, 300, "Temperature of the simulation"),
        ("height", "y", float, 1.2, "Height of gaussian bias"),
        ("width", "w", float, 0.35, "Width of gaussian bias, in radians"),
        ("deltaT", "dT", float, 5000, "DeltaT for well-tempered metadynamics"),
        ("stride", "s", int, 500, "Frequency of adding gaussian bias"),
        ("n-water", "n", int, 100, "Number of water molecules packed in the box"),
        ("box-length", "L", float, 2.0, "Box length in nm"),
        ("grid-shape", "g", int, 50, "Points per dimension of the bias grid"),
        ("grid", "G", bool, True, "Accumulate the bias on a grid instead of summing the gaussians"),
        ("fes-shape", "f", int, 64, "Points per dimension of the saved free energy surface"),
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

alanine_dipeptide = 'CC(=O)NC(C)C(=O)NC'
water = 'O'
force_field = 'openff-2.0.0.offxml'

ala = Molecule.from_smiles(alanine_dipeptide, allow_undefined_stereo=True)
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

# the solute comes first in the topology, so the dihedral indices are the molecule's own
protein_indices = np.arange(len(ala.atoms))
assert max(max(d) for d in DIHEDRALS) < len(ala.atoms)


class CustomMetaDLogger(MetaDLogger):
    """MetaDLogger that also reports progress to stdout."""

    def __init__(self, hills_file, log_period, print_period):
        super().__init__(hills_file, log_period)
        self.print_period = print_period

    def __call__(self, snapshot, state, timestep):
        """
        Implements the logging itself. Interface as expected for Callbacks.
        """
        if self.counter >= self.log_period and self.counter % self.log_period == 0:
            idx = state.idx - 1 if state.idx > 0 else 0
            self.save_hills(state.centers[idx], state.sigmas, state.heights[idx])
            if self.counter % self.print_period == 0:
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


def save_free_energy(run_result, method, temperature, kB, file_name, shape, chunk=4096):
    """
    Evaluate the deposited bias on a dense mesh and write the free energy surface to
    ``{file_name}_fes.npz``, following the analysis in the PySAGES metadynamics example.

    The surface is reported in units of kT with its minimum shifted to zero, shaped
    ``(shape,) * len(cvs)`` so that ``A[i, j, ...]`` sits at ``axes[0][i], axes[1][j], ...``.
    """
    ncvs = len(method.cvs)
    fes_grid = pysages.Grid(
        lower=(-pi,) * ncvs,
        upper=(pi,) * ncvs,
        shape=(shape,) * ncvs,
        periodic=True,
    )
    xi = compute_mesh(fes_grid)

    # bias factor: 1 for standard metadynamics, (T + deltaT) / deltaT when well-tempered
    alpha = 1 if method.deltaT is None else (temperature + method.deltaT) / method.deltaT
    kT = kB * temperature

    result = pysages.analyze(run_result)
    metapotential = result["metapotential"]

    # chunked so that a fine mesh over many gaussians stays within device memory
    A = np.concatenate(
        [np.asarray(metapotential(xi[i:i + chunk])) for i in range(0, xi.shape[0], chunk)]
    )

    # report in kT and set the free energy minimum to zero
    A = A * -alpha / kT
    A = A - A.min()
    A = A.reshape(fes_grid.shape)

    # per-dimension CV values of the mesh, each axis monotonically increasing
    mesh = np.asarray(xi).reshape(*fes_grid.shape, ncvs)
    axes = np.stack([np.unique(mesh[..., k]) for k in range(ncvs)])

    out = f'{file_name}_fes.npz'
    np.savez(
        out,
        A=A,                                  # free energy in kT, shape (shape,) * ncvs
        axes=axes,                            # CV values along each dimension, in radians
        cv_indices=np.asarray(method_dihedrals(method)),
        heights=np.asarray(result["heights"]),
        alpha=alpha,
        kT=kT,
        temperature=temperature,
        deltaT=np.nan if method.deltaT is None else method.deltaT,
        sigma=np.asarray(method.sigma),
        gaussian_height=np.asarray(method.height),
        stride=method.stride,
    )
    print(f'[analysis] free energy surface {A.shape} -> {out}', flush=True)
    return A


def method_dihedrals(method):
    """The atom quadruple behind each of the method's dihedral CVs."""
    return [list(cv.indices.flatten()) for cv in method.cvs]


def main(argv):
    args = get_args(argv)

    # parameters
    height = args.height # kj/mol
    temperature = args.temperature #K
    deltaT = args.deltaT
    stride = args.stride
    timesteps = args.time_steps
    ngauss = timesteps // stride + 1
    output_stride = args.log_steps

    prefix = 'results_dihedral/'
    os.makedirs(prefix, exist_ok=True)

    cvs = [DihedralAngle(list(d)) for d in DIHEDRALS]
    sigma = [args.width] * len(cvs) # radians
    file_name = f'{height}_{sigma[0]}_{stride}_{deltaT}_dihedral'

    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kB = kB.value_in_unit(unit.kilojoules_per_mole / unit.kelvin)

    # Optional grid for the bias: PySAGES then looks the force up per grid cell instead
    # of summing the gaussians every step. It is off by default -- measured on this
    # system it costs ~1 ms/step more than the direct sum, because the per-step cost is
    # dominated by fixed overhead rather than by the 20001-term sum, and the lookup is
    # piecewise constant in CV space (no interpolation between cells).
    grid = None
    if args.grid:
        grid = pysages.Grid(
            lower=(-pi,) * len(cvs),
            upper=(pi,) * len(cvs),
            shape=(args.grid_shape,) * len(cvs),
            periodic=True,
        )

    method = Metadynamics(cvs=cvs, height=height, sigma=sigma, stride=stride, ngaussians=ngauss, deltaT=deltaT, kB=kB, grid=grid)
    callback = CustomMetaDLogger(f'{prefix}{file_name}.dat', stride, output_stride)

    context_args = {
        'file_name': f'{prefix}{file_name}',
        'output_stride': output_stride,
        'temperature': temperature,
    }
    print(f'[run] dihedral CVs: {DIHEDRALS}', flush=True)
    print(f'[run] metadynamics, {timesteps} steps, {ngauss} gaussians of height {height} every {stride} steps', flush=True)
    print(f'[run] writing to {prefix}{file_name}.{{dat,pdb,csv}} (first step includes the jit compilation)', flush=True)
    run_result = pysages.run(method,
                             generate_simulation,
                             timesteps,
                             callback,
                             context_args=context_args)
    print('[run] finished', flush=True)

    save_free_energy(run_result, method, temperature, kB, f'{prefix}{file_name}', args.fes_shape)


if __name__ == "__main__":
    main(sys.argv[1:])
