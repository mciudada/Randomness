import numpy as np
from inflation import InflationProblem, InflationLP
from qutip import *
from probabilities import prob_postquantum

# Defining the scenario
triangle_Eve = InflationProblem(dag={"rho_AB": ["A", "B", "E"],
                                    "rho_BC": ["B", "C", "E"], 
                                    "rho_AC": ["A", "C", "E"]},
                                    classical_sources=None,
                                    outcomes_per_party=(2, 2, 2, 2),
                                    settings_per_party=(1, 1, 1, 1),
                                    inflation_level_per_source=(2, 2, 3), 
                                    order=["A", "B", "C", "E"])
InfLP = InflationLP(triangle_Eve, include_all_outcomes=False, verbose=2)
for m in InfLP.knowable_atoms:
    print("kn", m)

values = {m.name: m.compute_marginal(prob_postquantum(0, np.sqrt(2)-1, 0)) 
          for m in InfLP.knowable_atoms if 'E' not in m.name}
print(values)
InfLP.set_objective({'1': 1, 'pAE(00|00)': 2, 'pE(0|0)': -1, 'pA(0|0)': -1}, direction='max') # guessing prob<=0.966273233777505 for level (2, 2, 3)

InfLP.set_values(values, use_lpi_constraints=True)
InfLP.solve(solve_dual=False)
print(InfLP.objective_value)
for m in InfLP.knowable_atoms: 
    print(f"{m.name}->{InfLP.solution_object['x'][m.name]}")

