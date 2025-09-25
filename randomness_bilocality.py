import numpy as np
from inflation import InflationProblem, InflationLP
from qutip import *
from probabilities import prob_ent, prob_Fritz


# Defining the scenario
chain_Eve = InflationProblem(dag={"rho_AB": ["A", "B", "E"],
                                    "rho_BC": ["B", "C", "E"]},
                                    classical_sources=None,
                                    outcomes_per_party=(2, 2, 2, 2),
                                    # outcomes_per_party=(2, 2, 2, 4),
                                    settings_per_party=(2, 1, 2, 1),
                                    inflation_level_per_source=(2, 2), 
                                    order=["A", "B", "C", "E"])
InfLP = InflationLP(chain_Eve, include_all_outcomes=False, verbose=2)
for m in InfLP.knowable_atoms:
    print("kn", m)

# For entanglement-swapping
p_ent = np.zeros((2, 2, 2, 2, 1, 2))
p_ent[:,:,:,:,0,:] = prob_ent()
values = {m.name: m.compute_marginal(p_ent) 
          for m in InfLP.knowable_atoms if 'E' not in m.name}
print(values)
InfLP.set_objective({'1': 1, 'pAE(00|10)': 2, 'pE(0|0)': -1, 'pA(0|1)': -1}, direction='max') # If guessing Alice
# InfLP.set_objective({'1': 1, 'pCE(00|00)': 2, 'pE(0|0)': -1, 'pC(0|0)': -1}, direction='max') # If guessing Charlie
# InfLP.set_objective({'1': 1, 'pBE(00|00)': 2, 'pE(0|0)': -1, 'pB(0|0)': -1}, direction='max') # If guessing Bob

# # Fritz-style correlation
# p_Fritz = np.zeros((2, 2, 2, 2, 1, 2))
# p_Fritz[:,:,:,:,0,:] = prob_Fritz()
# values = {m.name: m.compute_marginal(p_Fritz) 
#           for m in InfLP.knowable_atoms if 'E' not in m.name}
# print(values)
# InfLP.set_objective({'1': 1, 'pAE(00|10)': 2, 'pE(0|0)': -1, 'pA(0|1)': -1}, direction='max') # If guessing Alice
# InfLP.set_objective({'1': 1, 'pBE(00|00)': 2, 'pE(0|0)': -1, 'pB(0|0)': -1}, direction='max') # If guessing Bob
# InfLP.set_objective({'1': 1, 'pCE(00|00)': 2, 'pE(0|0)': -1, 'pC(0|0)': -1}, direction='max') # If guessing Charlie

InfLP.set_values(values, use_lpi_constraints=True)
InfLP.solve(solve_dual=False)
print(InfLP.objective_value)
for m in InfLP.knowable_atoms: 
    print(f"{m.name}->{InfLP.solution_object['x'][m.name]}")
    
