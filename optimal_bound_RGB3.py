import gurobipy as gp
from gurobipy import GRB
from itertools import repeat
import numpy as np
from qutip import *
from probabilities import prob_RGB3

time_limit = GRB.INFINITY
tol = 1e-5
return_dist = True
print_model = True

def optimal_nonclassical_bound_RGB3(cardL: int)-> bool:
    (cardA, cardB, cardC) = (3, 3, 3)
    rL = tuple(range(cardL))

    with (gp.Env(empty=True) as env):
        env.start()
        with gp.Model("qcp", env=env) as m:
            m.params.NonConvex = 2  # Using quadratic equality constraints.
            m.setParam('FeasibilityTol', tol)
            m.setParam('OptimalityTol', 0.01)
            m.setParam('TimeLimit', time_limit)
            m.setParam('Presolve', 0)
            m.setParam('PreSparsify', 1)
            m.setParam('PreQLinearize', 1)
            m.setParam('PreDepRow', 1)
            m.setParam('Symmetry', 2)
            m.setParam('Heuristics', 1.0)
            m.setParam('RINS', 0)
            m.setParam('MIPFocus', 1)
            m.setParam('MinRelNodes', 0)
            m.setParam('ZeroObjNodes', 0)
            m.setParam('ImproveStartGap', 0)
            m.setParam('PartitionPlace', 31)

            # Define fully unpacked probabilities
            unpacked_C_cardinalities = tuple(repeat(cardC, times=cardL))
            unpacked_all_cardinalities = (cardA,) + (cardB,) + unpacked_C_cardinalities + (cardL,) 

            # Define RGB3 in terms of its variables 
            lambda_0 = m.addVar(lb=0, ub=1, name="lambda_0")
            lambda_1 = m.addVar(lb=0, ub=1, name="lambda_1")
            u = m.addVar(lb=0, ub=1, name="u")
            v = m.addVar(lb=0, ub=1, name="v")
            m.addConstr(u**2 + v**2 == 1, name="Normalization parameters 1")
            m.addConstr(lambda_0**2 + lambda_1**2 == 1, name="Normalization parameters 2")
            p_ABC = m.addMVar((3, 3, 3), lb=0, name="p_ABC")
            for (i, j, k) in np.ndindex(3, 3, 3):
                if (i,j,k)==(0,0,1) or (i,j,k)==(0,1,0) or (i,j,k)==(1,0,0):
                    m.addConstr(p_ABC[i,j,k]==(lambda_1**4) * (lambda_0**2) * (u**2) + (lambda_0**4) * (lambda_1**2) * (v**2))
                elif (i,j,k)==(0,0,2) or (i,j,k)==(0,2,0) or (i,j,k)==(2,0,0):
                    m.addConstr(p_ABC[i,j,k]==(lambda_1**4) * (lambda_0**2) * (v**2) + (lambda_0**4) * (lambda_1**2) * (u**2))
                elif (i,j,k)==(1,1,2) or (i,j,k)==(1,2,1) or (i,j,k)==(2,1,1):
                    m.addConstr(p_ABC[i,j,k]==(lambda_1**3 * u**2 * v - lambda_0**3 * v**2 * u)**2)
                elif (i,j,k)==(1,2,2) or (i,j,k)==(2,2,1) or (i,j,k)==(2,1,2):
                    m.addConstr(p_ABC[i,j,k]==(lambda_1**3 * u * v**2 + lambda_0**3 * v * u**2)**2)
                elif (i,j,k)==(1,1,1):
                    m.addConstr(p_ABC[i,j,k]==(lambda_1**3 * u**3 +lambda_0**3 * v**3)**2)   
                elif (i,j,k)==(2,2,2):
                    m.addConstr(p_ABC[i,j,k]==(lambda_1**3 * v**3 - lambda_0**3 * u**3)**2)
                else:
                    m.addConstr(p_ABC[i,j,k]==0)
            
            # # # # Introduce core MVars, Q_ABCC and Q_L, from which everything else is derived.
            # # # Q_ABCCX = m.addMVar(unpacked_all_cardinalities, lb=0, name="Q_ABCCX")
            # # # for x in np.ndindex(cardL):
            # # #     m.addConstr(Q_ABCCX[...,x].sum() <= 1, name="Unpacked Normalization")
            # # # Q_L = m.addMVar((cardL,), lb=0, name="Q_L")
            # # # m.addConstr(Q_L.sum() == 1, name="Normalization of hidden L")

            # # # # NS constraint
            # # # Q_BCCX = m.addMVar((cardB,)+unpacked_C_cardinalities+(cardL,), lb=0, name="Q_BCCX")
            # # # m.addConstr(Q_BCCX == Q_ABCCX.sum(axis=0), name="Q_BCCX from Q_ABCCX")
            # # # for x in tuple(range(cardL-1)):
            # # #     m.addConstr(Q_BCCX[...,x] == Q_BCCX[...,x+1], name="NS constraint")
            
            # # # # Introduce Q_ACC, Q_CC, Q_AX to capture nonlinear constraint
            # # # Q_ACCX = m.addMVar((cardA,) + unpacked_C_cardinalities + (cardL,), name="Q_ACCZ")
            # # # m.addConstr(Q_ACCX == Q_ABCCX.sum(axis=1), name="Q_ACCX from Q_ABCCX")
            # # # Q_AX = m.addMVar((cardA,cardL), name="Q_AX")
            # # # Q_CC = m.addMVar(unpacked_C_cardinalities, name="Q_CC")
            # # # axes_of_CC = tuple(range(1,cardL+1))
            # # # m.addConstr(Q_AX == Q_ACCX.sum(axis=axes_of_CC), name="Q_AX from Q_ACCX")
            # # # axes_of_A = 0
            # # # m.addConstr(Q_CC == Q_ACCX.sum(axis=axes_of_A)[...,0], name="Q_CC from Q_ACCX")
            
            # # # # Nonlinear constraint: Q_ACCX = Q_AX * Q_CC
            # # # Q_AX_reshaped_for_multiplication = Q_AX.reshape((cardA,) + tuple(repeat(1, times=(cardL))) + (cardL,))
            # # # Q_CC_reshaped_for_multiplication = Q_CC.reshape((1,) + Q_CC.shape + (1,))
            
            # # # m.addConstr(Q_ACCX == Q_AX_reshaped_for_multiplication * Q_CC_reshaped_for_multiplication, name="Q_ACCX = Q_AX * Q_CC")

            # # # # Linear constraint: P_ABC relates to Q_LABCC 
            # # # # P_ABC = m.addMVar((cardA, cardB, cardC), lb=0, name="P_ABC")
            # # # temp = gp.MQuadExpr.zeros(p_ABC.shape)
            # # # default_axes_to_trace_out = set(range(cardL + 3))
            # # # default_axes_to_trace_out.discard(cardL+2) # removing axis for X
            # # # default_axes_to_trace_out.discard(0) # removing axis for A
            # # # default_axes_to_trace_out.discard(1) # removing axis for B
            # # # for l in rL:
            # # #     axes_to_trace_out = tuple(sorted(default_axes_to_trace_out.difference({l+2})))
            # # #     temp += Q_L[l] * Q_ABCCX[...,l].sum(axis=axes_to_trace_out)
            # # # m.addConstr(p_ABC == temp, name="P_ABC from Q_LAABCC")

            ineq = m.addVar(lb = -1e20, name="ineq")
            m.addConstr(ineq ==3*((lambda_1**3)*(u**2)*v-(lambda_0**3)*u*(v**2))**2 - 3*(u**2)*((lambda_1**6)+(lambda_0**6)) + 2*((lambda_0**6)+((lambda_1**3)*(u**3)+(lambda_0**3)*(v**3))**2) + (lambda_1**6) + ((lambda_1**3)*(v**3)-(lambda_0**3)*(u**3))**2, name="objective")

            m.setObjective(ineq, GRB.MINIMIZE)
            m.Params.NonConvex = 2

            m.update()
            current_status = m.getAttr("Status")
            gurobi_status_codes = [
                'LOADED',
                'OPTIMAL',
                'INFEASIBLE',
                'INF_OR_UNBD',
                'UNBOUNDED',
                'CUTOFF',
                'ITERATION_LIMIT',
                'NODE_LIMIT',
                'TIME_LIMIT',
                'SOLUTION_LIMIT',
                'INTERRUPTED',
                'NUMERIC',
                'SUBOPTIMAL',
                'INPROGRESS',
                'USER_OBJ_LIMIT',
                'WORK_LIMIT']
            if current_status == 1:
                m.optimize()
                current_status = m.getAttr("Status")
                current_status_string = gurobi_status_codes[current_status-1]
                print(f"Status: {current_status_string}")

            if m.getAttr("SolCount"):
                record_to_preserve = dict()
                for var in m.getVars():
                    record_to_preserve[var.VarName] = var.X
                    if print_model:
                        print(var.VarName, " := ", var.X)

            m.dispose()
        env.dispose()
    return current_status_string

print(optimal_nonclassical_bound_RGB3(2))