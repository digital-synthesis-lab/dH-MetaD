# Implementing information entropy change as the general-purpose collective variable for enhanced sampling

We are using information entropy as the general-purpose collective variable (CV) for metadynamics enhanced sampling and we have validated this methods across multiple organic and inorganic systems. It can quantify the "novelty" of the sampled configurations based on chosen reference configurations and push the system away from sampling the reference structures.

Preprint of this work is avaliable: [preprint](https://doi.org/10.48550/arXiv.2604.05239)

# Dependencies

[PySAGES](https://github.com/SSAGESLabs/PySAGES.git)
[QUESTS](https://github.com/dskoda/quests.git)
[Openff](https://github.com/openforcefield/openff-toolkit.git)
ASE
RDKit
OpenMM
Jax

# PySages setup

Please follow the instruction in the PySAGES homepage [PySAGES](https://github.com/SSAGESLabs/PySAGES.git) for OpenMM and LAMMPS plugin compilation. Few codes have been adjusted for this work to track the simulation box under NPT ensemble. The adjusted repo can be found here [PySAGES-adjusted](git@github.com:XiangruiLi-Ray/PySAGES.git).

# Examples

### 1. Alanine Dipeptide

### 2. Alanine Tetrapeptide

### 3. Copper liquid-solid nucleation

### 4. Graphite solid-solid nucleation

### 5. Silicon liquid-solid nucleation
