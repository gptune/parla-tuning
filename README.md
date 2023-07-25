# Code package for PARLA autotuning with GPTune

## Overview

This repository provides a package of Python codes and scripts for run and
reproduce our autotuning experiments on randomized least-squares solvers.

Randomzied least-squares algorithms contain tunable parameters that need to be
optimized for a given system to obtain a satisfactory performance in terms of
both runtime and accuracy. In our experiments, we first examine performance of
several different randomized least-squares solvers based on a grid search.
We then compare quality of different autotuning methodologies for optimizing
performance of randomized least-squares.

We show that a Bayesian optimization-based autotuning based on a GP surrogate
model (using GPTune) can be an effective way to tune randomized least-squares
algorithms (and potentially for other randomized algorithms too). The presented
Bayesian optimization-based approach outperforms primitive tuning methods such
as random search and grid search. In addition, the approach can leverage transfer
learning that can use performance results from different input matrices to tune
a new input matrix much faster. Also, the surrogate modeling enables sensivitiy
analysis that can estimate the sensitivity score of each tuning parameter.

## Setup

For experiments, you need to have a Python environment (we recommend a Python
version higher than 3.8) and install PARLA and GPTune. PARLA stands for Python
Algorithms for Randomized Linear Algebra and provides the algorithm
implementations for the randomized least-squares solvers considered in our
experiments. GPTune is a GP-based Bayesian optimization autotuner designed for
high-performance computing applications but can also be used as a general
purpose black-box autotuner. We use GPTune for our surrogate modeling and tuning
framework.

*(1) Setup PARLA*
```
$ git clone https://github.com/BallisticLA/parla -b newtonSketchLstsq
$ git checkout 74b0d20461a3d70780af83b3df430c4d9d748d15
install PARLA based on its README file
```

*(2) Setup GPTune*
```
$ git clone https://github.com/gptune/GPTune
$ git checkout 6286557bdfdf98bbe5e5d2fb0977ab578ac28378
install GPTune based on its guideline (README and UserGuide).
```

## Run experiments

There are multiple experiments we perform. First, please make sure to generate
the input test matrices used in our experiments.
```
$ cd input/
$ python test_matrix_generator_mvt.py
```

*(1) Grid evaluation of the PARLA*
```
$ cd grid_search
$ bash run_grid_search.sh
```
Note that this experiment will take a very long time. Once can minizmie the
experiment size in grid_search/grid_search.py by adjusting its grid search
loop iterations.

*(2) Run autotuning methodologies*

- Generate random initial samples
```
$ cd tuning
$ bash prepare_pilot_samples.sh
```

- Run multiple different autotuning methodologies
```
$ cd tuning
$ bash run_tuning_experiments.sh
```

*(3) Run transfer leraning autotuning*

- Prepare source
```
$ cd tuning_tla
$ bash run_source_task.sh
```

- Run transfer learning
```
$ cd tuning_tla
$ bash run_tla_mab.sh
$ bash run_tla_mab_various_sources.sh
$ bash run_tla_mab_various_options.sh
$ bash run_tla_mab_original.sh
```

## Analyzing experimental results

- Generate plots for understanding performance of randomized least-squares
solvers using grid search.
For the generated plots, please see *plots* directory.
```
$ python analysis_grid_search.py
```

- Generate plots to compare tuning quality of different autotuning methodologies
including transfer learning. Tuning quality is based on the tuned performance
depending on the number of evaluations.
For the generated plots, please see *plots* directory.
```
$ analysis_tuning.py
```

- Generate plots to see how different types of input matrices would affect
tuning quality of transfer learning.
For the generated plots, please see *plots* directory.
```
$ analysis_tuning_source_matrices.py
```

- Generate plots to see how different options for transfer learning would
affect the tuning quality.
For the generated plots, please see *plots* directory.
```
$ analysis_tuning_TLA_options.py
```

- Compute Sobol's sensitivity analysis based on some results from autotuning.
For the generated results, please see *sobol_analysis* directory.
```
$ python sobol_analysis.py
```

# Contributors

Younghyun Cho, UC Berkeley
Jim Demmel, UC Berkeley
Michal Derezinski, The University of Michigan
Haoyun Li, UC Berkeley
Hengrui Luo, Lawrence Berkeley National Laboratory
Michael Mahoney, UC Berkeley & Lawrence Berkeley National Laboratory & International Computer Science Institute
Riley Murray, UC Berkeley & Lawrence Berkeley National Laboratory & International Computer Science Institute

# Questions?

For questions about this package, please contact
Younghyun Cho <younghyun@berkeley.edu>

