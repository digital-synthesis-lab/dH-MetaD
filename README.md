# Implementing information entropy change as the general-purpose collective variable for enhanced sampling

We are using information entropy as the general-purpose collective variable (CV) for metadynamics enhanced sampling and we have validated this methods across multiple organic and inorganic systems. Information entropy change (dH) can quantify the "novelty" or "surprise" of the sampled configurations based on chosen reference configurations and push the system away from sampling the reference structures.

It allows blind exploration of unknown metastable states in the high-dimensional configurational space and model-free.
Preprint of this work is avaliable: [preprint](https://doi.org/10.48550/arXiv.2604.05239)

# Installation

You can install all dependencies from this reposistory:
```
git clone https://github.com/XiangruiLi-Ray/dH-MetaD.git
cd dH-Metad
pip install .
```

Some repo links are included here:
[PySAGES](https://github.com/SSAGESLabs/PySAGES.git)
[QUESTS](https://github.com/dskoda/quests.git)
[Openff](https://github.com/openforcefield/openff-toolkit.git)

# PySages and simulation backends setup

Please follow the instruction in the PySAGES homepage [PySAGES](https://github.com/SSAGESLabs/PySAGES.git) for OpenMM and LAMMPS plugin compilation. Few codes have been adjusted for this work to track the simulation box under NPT ensemble. The adjusted repo can be found here [PySAGES-adjusted](git@github.com:XiangruiLi-Ray/PySAGES.git).

# Examples

This repo have all files needed to directly implement dH-metadynamics across the following examples.

1. Alanine Dipeptide (ala2)

2. Alanine Tetrapeptide (ala4)

3. Copper liquid-solid nucleation (copper)

4. Graphite solid-solid nucleation (graphite)

5. Silicon liquid-solid nucleation (silicon)
