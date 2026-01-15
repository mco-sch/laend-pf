# laend-pf
laend-pf is an enhancement of the Life Cycle Assessment based ENergy Decision support tool [LAEND](https://github.com/inecmod/LAEND), which is based on the open energy modeling framework [oemof.solph](https://github.com/oemof/oemof-solph).
Both, LAEND and laend-pf enable a coupled energy system analysis with environmentally oriented sustainability assessment and optimization. However, their main differences are:
- laend-pf uses oemof's multi-period investment mode, enabling multi-period perfect foresight optimization in contrast to LAEND's myopic optimization.
- laend-pf relies on [brightway2](https://github.com/brightway-lca/brightway2) for corresponding LCA calculations, while LAEND relies on [openLCA](https://www.openlca.org/)

# Getting Started
**1. Clone the repository**\
Clone the repository and go to the directory it has been cloned to.
```
git clone https://github.com/mco-sch/laend-pf
cd laend-pf
```

**2. Create and activate a virtual environment** (recommended: Python 3.10)\
Other Python versions might work but were not tested. To set up and use virtual environments, please refer to the documentation of your python distribution (e.g. Anaconda, Micromamba). For instance, for Anaconda it would look like
```
conda create -n <myenv> python=3.10
conda activate <myenv>
```

**3. Install required packages** (don't forget to activate the virtual environment)
```
conda activate <myenv>
(myenv) pip install -r requirements.txt
```

**4. Install a solver** (e.g. CBC, GLPK, or Gurobi)\
For details see [here](https://github.com/oemof/oemof-solph?tab=readme-ov-file#installing-a-solver)
Make sure the path to the executable is added to your system's PATH variable

**5. Test the Installation**\
Test the installation and the installed solver by running the installation test in your venv:\
`(myenv) oemof_installation_test`

# Documentation & Usage
For documentation, see the LAEND documentation on [GitHub](https://github.com/inecmod/LAEND) or on the website of the [Institute for Industrial Ecology](https://www.hs-pforzheim.de/forschung/institute/inec/sonstiges/laend).

## LCA files
Before using, have a look at the existing LCIA files in `/in/LCA` and check if they are still up to date and valid for your use case.
If necessary, new LCIA files can be added to the directory, or by using brightway2 individual LCA models can be imported automatically (for configuration of automated import, adjust parameters in `config_pf_auto.py`)

## Potential Future Improvements
- modularization an rearrangement of .py-modules
- combination of myopic and multi-period perfect foresight optimization approaches within same tool
- reintroduction of functions that were neglected during the transformation from LAEND to laend-pf
