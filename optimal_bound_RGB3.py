import gurobipy as gp
from gurobipy import GRB
import numpy as np
from helpers import customize_model_for_nonlinear_SAT, check_feasibility
from triangle_two_classical_sources import impose_two_classical_sources


def optimal_nonclassical_bound_RGB3(cardL: int, print_model=True)-> str:
    with (gp.Env(empty=True) as env):
        env.start()
        with gp.Model("qcp", env=env) as m:
            customize_model_for_nonlinear_SAT(m)

            # Define quantum parameters
            lambda_0 = m.addVar(lb=0, ub=1, name="lambda_0")
            lambda_1 = m.addVar(lb=0, ub=1, name="lambda_1")
            u = m.addVar(lb=0, ub=1, name="u")
            v = m.addVar(lb=0, ub=1, name="v")
            m.addConstr(u ** 2 + v ** 2 == 1, name="Normalization parameters 1")
            m.addConstr(lambda_0 ** 2 + lambda_1 ** 2 == 1, name="Normalization parameters 2")

            # Define RGB3 distribution in terms of parameters
            p_ABC = m.addMVar((3, 3, 3), lb=0, name="p_ABC")
            for (i, j, k) in np.ndindex(3, 3, 3):
                if (i, j, k) == (0, 0, 1) or (i, j, k) == (0, 1, 0) or (i, j, k) == (1, 0, 0):
                    m.addConstr(p_ABC[i, j, k] == (lambda_1 ** 4) * (lambda_0 ** 2) * (u ** 2) + (lambda_0 ** 4) * (
                                lambda_1 ** 2) * (v ** 2))
                elif (i, j, k) == (0, 0, 2) or (i, j, k) == (0, 2, 0) or (i, j, k) == (2, 0, 0):
                    m.addConstr(p_ABC[i, j, k] == (lambda_1 ** 4) * (lambda_0 ** 2) * (v ** 2) + (lambda_0 ** 4) * (
                                lambda_1 ** 2) * (u ** 2))
                elif (i, j, k) == (1, 1, 2) or (i, j, k) == (1, 2, 1) or (i, j, k) == (2, 1, 1):
                    m.addConstr(p_ABC[i, j, k] == (lambda_1 ** 3 * u ** 2 * v - lambda_0 ** 3 * v ** 2 * u) ** 2)
                elif (i, j, k) == (1, 2, 2) or (i, j, k) == (2, 2, 1) or (i, j, k) == (2, 1, 2):
                    m.addConstr(p_ABC[i, j, k] == (lambda_1 ** 3 * u * v ** 2 + lambda_0 ** 3 * v * u ** 2) ** 2)
                elif (i, j, k) == (1, 1, 1):
                    m.addConstr(p_ABC[i, j, k] == (lambda_1 ** 3 * u ** 3 + lambda_0 ** 3 * v ** 3) ** 2)
                elif (i, j, k) == (2, 2, 2):
                    m.addConstr(p_ABC[i, j, k] == (lambda_1 ** 3 * v ** 3 - lambda_0 ** 3 * u ** 3) ** 2)
                else:
                    m.addConstr(p_ABC[i, j, k] == 0)

            # Define objective: a polynomial function, which, when negative, certifies nonclassicality.
            ineq = m.addVar(lb = -10, ub=10, name="ineq")
            m.addConstr(ineq ==3*((lambda_1**3)*(u**2)*v-(lambda_0**3)*u*(v**2))**2 - 3*(u**2)*((lambda_1**6)+(lambda_0**6)) + 2*((lambda_0**6)+((lambda_1**3)*(u**3)+(lambda_0**3)*(v**3))**2) + (lambda_1**6) + ((lambda_1**3)*(v**3)-(lambda_0**3)*(u**3))**2, name="objective")
            m.setObjective(ineq, GRB.MINIMIZE)

            # Impose causal compatibility with triangle consisting of 2 classical sources, one having cardinality=cardL
            Q_ABCCX, Q_X, Q_CC = impose_two_classical_sources(m, p_ABC, cardL)
            status_message, variable_values = check_feasibility(m, print_model=print_model)

            m.dispose()
        env.dispose()
    return status_message

# Evidently, some nonclassicality is consistent with 2 classical sources, one having cardinality=2.
print(optimal_nonclassical_bound_RGB3(2))
