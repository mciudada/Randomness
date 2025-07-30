import gurobipy as gp
from gurobipy import GRB
import numpy as np
from helpers import customize_model_for_nonlinear_SAT, check_feasibility
from triangle_two_classical_sources import impose_two_classical_sources


def optimal_nonclassical_bound_RGB3(cardL: int, print_model=True, stop_if_violation=10.0)-> str:
    with (gp.Env(empty=True) as env):
        env.start()
        with gp.Model("qcp", env=env) as m:
            customize_model_for_nonlinear_SAT(m)

            # Define quantum parameters
            lambda_0 = m.addVar(lb=0, ub=1, name="lambda_0")
            lambda_1 = m.addVar(lb=0, ub=1, name="lambda_1")
            u = m.addVar(lb=0, ub=1, name="u")
            v = m.addVar(lb=0, ub=1, name="v")

            # To accelerate computation, we define squares, cubes, and fourth powers
            lambda_0__2 = m.addVar(lb=0, ub=1, name="lambda_0^2")
            m.addConstr(lambda_0__2 == lambda_0 ** 2)
            lambda_0__3 = m.addVar(lb=0, ub=1, name="lambda_0^3")
            m.addConstr(lambda_0__3 == lambda_0 * lambda_0__2)
            lambda_0__4 = m.addVar(lb=0, ub=1, name="lambda_0^4")
            m.addConstr(lambda_0__4 == lambda_0__2**2)
            lambda_0__6 = m.addVar(lb=0, ub=1, name="lambda_0^6")
            m.addConstr(lambda_0__6 == lambda_0__2 * lambda_0__4)

            lambda_1__2 = m.addVar(lb=0, ub=1, name="lambda_1^2")
            m.addConstr(lambda_1__2 == lambda_1 ** 2)
            lambda_1__3 = m.addVar(lb=0, ub=1, name="lambda_1^3")
            m.addConstr(lambda_1__3 == lambda_1 * lambda_1__2)
            lambda_1__4 = m.addVar(lb=0, ub=1, name="lambda_1^4")
            m.addConstr(lambda_1__4 == lambda_1__2 ** 2)
            lambda_1__6 = m.addVar(lb=0, ub=1, name="lambda_1^6")
            m.addConstr(lambda_1__6 == lambda_1__2 * lambda_1__4)

            u__2 = m.addVar(lb=0, ub=1, name="u^2")
            m.addConstr(u__2 == u ** 2)
            u__3 = m.addVar(lb=0, ub=1, name="u^3")
            m.addConstr(u__3 == u * u__2)
            v__2 = m.addVar(lb=0, ub=1, name="v^2")
            m.addConstr(v__2 == v ** 2)
            v__3 = m.addVar(lb=0, ub=1, name="v^3")
            m.addConstr(v__3 == v * u__2)

            m.addConstr(u__2 + v__2 == 1, name="Normalization parameters 1")
            m.addConstr(lambda_0__2 + lambda_1__2 == 1, name="Normalization parameters 2")

            # Define RGB3 distribution in terms of parameters
            p_ABC = m.addMVar((3, 3, 3), lb=0, name="p_ABC")
            for (i, j, k) in np.ndindex(3, 3, 3):
                if (i, j, k) == (0, 0, 1) or (i, j, k) == (0, 1, 0) or (i, j, k) == (1, 0, 0):
                    m.addConstr(p_ABC[i, j, k] == (lambda_1__4) * (lambda_0__2) * (u__2) + (lambda_0__4) * (
                                lambda_1__2) * (v__2))
                elif (i, j, k) == (0, 0, 2) or (i, j, k) == (0, 2, 0) or (i, j, k) == (2, 0, 0):
                    m.addConstr(p_ABC[i, j, k] == (lambda_1__4) * (lambda_0__2) * (v__2) + (lambda_0__4) * (
                                lambda_1__2) * (u__2))
                elif (i, j, k) == (1, 1, 2) or (i, j, k) == (1, 2, 1) or (i, j, k) == (2, 1, 1):
                    m.addConstr(p_ABC[i, j, k] == (lambda_1__3 * u__2 * v - lambda_0__3 * v__2 * u) ** 2)
                elif (i, j, k) == (1, 2, 2) or (i, j, k) == (2, 2, 1) or (i, j, k) == (2, 1, 2):
                    m.addConstr(p_ABC[i, j, k] == (lambda_1__3 * u * v__2 + lambda_0__3 * v * u__2) ** 2)
                elif (i, j, k) == (1, 1, 1):
                    m.addConstr(p_ABC[i, j, k] == (lambda_1__3 * u__3 + lambda_0__3 * v__3) ** 2)
                elif (i, j, k) == (2, 2, 2):
                    m.addConstr(p_ABC[i, j, k] == (lambda_1__3 * v__3 - lambda_0__3 * u__3) ** 2)
                else:
                    m.addConstr(p_ABC[i, j, k] == 0)

            # Define objective: a polynomial function, which, when negative, certifies nonclassicality.
            ineq = m.addVar(lb = -10, ub=10, name="ineq")
            m.addConstr(ineq ==3*((lambda_1__3)*(u__2)*v-(lambda_0__3)*u*(v__2))**2 - 3*(u__2)*((lambda_1__6)+(lambda_0__6)) + 2*((lambda_0__6)+((lambda_1__3)*(u__3)+(lambda_0__3)*(v__3))**2) + (lambda_1__6) + ((lambda_1__3)*(v__3)-(lambda_0__3)*(u__3))**2, name="objective")
            m.setObjective(ineq, GRB.MINIMIZE)
            m.setParam('BestObjStop', -stop_if_violation)

            # Impose causal compatibility with triangle consisting of 2 classical sources, one having cardinality=cardL
            Q_ABCCX, Q_X, Q_CC = impose_two_classical_sources(m, p_ABC, cardL)
            status_message, variable_values = check_feasibility(m, print_model=print_model)

            m.dispose()
        env.dispose()
    return status_message

# Evidently, some nonclassicality is consistent with 2 classical sources, one having cardinality=2.
print(optimal_nonclassical_bound_RGB3(2, stop_if_violation=0.2))
