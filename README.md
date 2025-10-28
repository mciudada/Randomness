# [Code to accompany "Certifying Randomness or its Lack Thereof in General Network Scenarios"](https://arxiv.org/abs/2510.20993)

## Maria Ciudad Alañón, Daniel Centeno, Andrew Watford and Elie Wolfe

This repository contains the codes used to obtain all the results in "Certifying Randomness or its Lack Thereof in Network Scenarios". Maria Ciudad Alañón, Daniel Centeno, Andrew Watford and Elie Wolfe.

All the code is written in Python. Part of the codes use the inflation library (and all its requirements) to solve inflation problems. Others used the solver Gurobi to solve linear and bilinear problems.

The files used for the results of the paper are:

- [helpers.py](helpers.py): this file contains helper functions used in other files.
- [probabilities.py](probabilities.py): this file contains all the probabilities used in the rest of the files.
- [randomness_bilocality.py](randomness_bilocality.py): this file uses the inflation package to prove randomness in the bilocality scenario.
- [randomness_triangle.py](randomness_triangle.py): this file uses the inflation package to prove randomness in the triangle scenario.
- [triangle_two_classical_sources.py](triangle_two_classical_sources.py): this file uses Gurobi to find a model with two classical sources in the triangle given the correlation and the cardinality of one of those sources.

The rest of the files have been used internally but are not used for the results shown in the paper.
