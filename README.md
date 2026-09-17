# Information Content is a General-Purpose Collective Variable for Enhanced Sampling

This repository contains the code and simulation inputs for **δH-MetaD**, a metadynamics
approach that uses the local information content of atomic environments as a
general-purpose collective variable (CV) for enhanced sampling.

Conventional enhanced sampling requires predefined CVs that presuppose knowledge of the
reaction coordinate, which restricts the discovery of unanticipated transition mechanisms
or intermediates. Instead, we bias simulations along the information content change (δH),
which quantifies the "novelty" or "surprise" of a sampled configuration with respect to a
chosen reference dataset, and therefore pushes the system away from configurations it has
already sampled. The method is model-free, requires no training data, and applies to both
organic and inorganic systems, enabling blind exploration of unknown metastable states in
high-dimensional configurational spaces.

A preprint of this work is available: [Li and Schwalbe-Koda, *Information Content is a
General-Purpose Collective Variable for Enhanced Sampling*](https://doi.org/10.48550/arXiv.2604.05239)

## Installation

The packages used in this project are tricky to handle, so we suggest creating a separate
environment for each simulation backend ([OpenMM](https://openmm.org/) for the organic
systems and [LAMMPS](https://www.lammps.org/) for the inorganic ones). Our
`pyproject.toml` installs only the minimum requirements — [QUESTS](https://github.com/dskoda/quests),
NumPy, ASE, Matplotlib and a [customized version of PySAGES](https://github.com/XiangruiLi-Ray/PySAGES)
— so that it does not damage an existing environment. The customized PySAGES is required
for the neighbor list reconstruction and the δH calculation; see that repository for the
modified code.

Python 3.10 or newer is required. Clone the repository and install the package:

```
git clone https://github.com/digital-synthesis-lab/dH-MetaD.git
cd dH-MetaD
pip install .
```

For the organic systems, the OpenFF Toolkit is also needed. Please follow their
[documentation](https://github.com/openforcefield/openff-toolkit) for installation.

## Simulation backend setup

The simulation backends and their PySAGES plugins are installed separately. Please follow
the instructions on the [PySAGES](https://github.com/SSAGESLabs/PySAGES) homepage for
OpenMM and LAMMPS backend and plugin installation.

JAX must also be installed separately to match your hardware (a CPU-only build is pulled
in by default). Please refer to the [PySAGES](https://github.com/SSAGESLabs/PySAGES)
homepage for the JAX installation instructions.

[`openmm-dlext`](https://github.com/SSAGESLabs/openmm-dlext) can be installed through
[conda](https://github.com/conda-forge/openmm-dlext-feedstock):

```
conda install -c conda-forge --strict-channel-priority openmm-dlext
```

The LAMMPS examples additionally require
[`lammps-dlext`](https://github.com/SSAGESLabs/lammps-dlext), which must be compiled
against your LAMMPS build. The inorganic simulations reported in the paper used LAMMPS
(v. 29 Aug 2024) with the Kokkos package enabled.

The S2 benchmark in `S2_test/` is independent of PySAGES and instead uses
[PLUMED](https://github.com/plumed/plumed2) patched into LAMMPS, together with the
`PairEntropy` CV supplied in each subdirectory.

## Examples

This repository contains all the files needed to directly implement δH-MetaD across the
following examples:

| Directory                 | System                                                                |
| ------------------------- | --------------------------------------------------------------------- |
| [`ala2/`](ala2/)         | Alanine dipeptide (Ala2) conformational change in vacuum              |
| [`ala2_sol/`](ala2_sol/) | Ala2 conformational change in water                                   |
| [`ala4/`](ala4/)         | Alanine tetrapeptide (Ala4) conformational change in vacuum           |
| [`ala4_sol/`](ala4_sol/) | Ala4 conformational change in water                                   |
| [`copper/`](copper/)     | Copper nucleation                                                     |
| [`silicon/`](silicon/)   | Silicon nucleation / glass transition                                 |
| [`carbon/`](carbon/)     | Graphite-to-diamond phase transformation                              |
| [`S2_test/`](S2_test/)   | Local crystalline order (S2) benchmark on the inorganic systems above |

Each directory holds the δH implementation (`utils.py`), the simulation driver
(`sim.py` or `sim.ipynb`), the reference configurations used to build `{X}` (`refer/`),
and the output of the biased runs (`results/`). The inorganic examples additionally ship
the force field (`ff/`), the LAMMPS input template (`in.lmp`), and the starting structures
(`str/`).

### Running the examples

The vacuum peptide examples are notebooks; open and run `ala2/sim.ipynb` or
`ala4/sim.ipynb`. The remaining examples are scripts driven from the command line, for
example:

```
cd copper
python sim.py --temperature 1100 --pressure 1 --height 0.1 --width 4 --bandwidth 0.025
```

Run `python sim.py --help` in any example directory for the full list of options (number
of time steps, bias height and width, deposition stride, well-tempered ΔT, δH kernel
bandwidth, and, for the solvated systems, the reference trajectory and the number of
reference frames). The solvated examples also provide `sim_dihe.py`, which runs the same
system with the backbone dihedrals as CVs for comparison. Results are written to
`results/` in the example directory.

## Citation

If you use δH-MetaD in your work, please cite:

```bibtex
@article{li2026information,
    title = {Information Content is a General-Purpose Collective Variable for Enhanced Sampling},
    author = {Li, Xiangrui and Schwalbe-Koda, Daniel},
    year = {2026},
    eprint = {2604.05239},
    archivePrefix = {arXiv},
    doi = {10.48550/arXiv.2604.05239},
    url = {https://doi.org/10.48550/arXiv.2604.05239},
}
```

This work builds on the following packages and methods, which should be cited alongside it:

- **QUESTS**, used for the information entropy formulation:
  D. Schwalbe-Koda, S. Hamel, B. Sadigh, F. Zhou, V. Lordi.
  *Model-free estimation of completeness, uncertainties, and outliers in atomistic machine
  learning using information theory.* Nature Communications **16**, 4014 (2025).
  [10.1038/s41467-025-59232-0](https://doi.org/10.1038/s41467-025-59232-0)
- **PySAGES**, the enhanced sampling library:
  P. F. Zubieta Rico, L. Schneider, G. R. Pérez-Lemus, R. Alessandri, et al.
  *PySAGES: flexible, advanced sampling methods accelerated with GPUs.*
  npj Computational Materials **10**, 35 (2024).
  [10.1038/s41524-023-01189-z](https://doi.org/10.1038/s41524-023-01189-z)
- **S2 fingerprint**, the entropy-based order parameter used as a baseline in `S2_test/`:
  P. M. Piaggi, M. Parrinello. *Entropy based fingerprint for local crystalline order.*
  The Journal of Chemical Physics **147**, 114112 (2017).
  [10.1063/1.4998408](https://doi.org/10.1063/1.4998408)
- **Metadynamics** and **well-tempered metadynamics**:
  A. Laio, M. Parrinello. *Escaping free-energy minima.* PNAS **99**, 12562–12566 (2002).
  [10.1073/pnas.202427399](https://doi.org/10.1073/pnas.202427399);
  A. Barducci, G. Bussi, M. Parrinello. *Well-tempered metadynamics: a smoothly converging
  and tunable free-energy method.* Physical Review Letters **100**, 020603 (2008).
  [10.1103/PhysRevLett.100.020603](https://doi.org/10.1103/PhysRevLett.100.020603)

## Acknowledgment

This work was supported by the U.S. Department of Energy (DOE), Office of Science, Office
of Basic Energy Sciences under Award Number DE-SC0025642. This research used resources of
the Argonne Leadership Computing Facility, which is a U.S. Department of Energy Office of
Science User Facility operated under contract DE-AC02-06CH11357.

## License

This project is distributed under the BSD-3-Clause license. See [LICENSE](LICENSE) for
details.
