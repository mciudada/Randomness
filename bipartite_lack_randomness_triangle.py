import gurobipy as gp
from gurobipy import GRB
from itertools import repeat
import numpy as np
from qutip import *

time_limit = GRB.INFINITY
tol = 1e-5
return_dist = True
print_model = True

def check_bipartite_lack_randomness(p_ABC: np.ndarray, cardL: int)-> bool:
    (cardA, cardB, cardC) = p_ABC.shape
    rL = tuple(range(cardL))

    with (gp.Env(empty=True) as env):
        env.start()
        with gp.Model("qcp", env=env) as m:
            m.params.NonConvex = 2  
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
            unpacked_all_cardinalities = (cardA,) + (cardB,) + unpacked_C_cardinalities + (cardL,) #this corresponds to ABC0C1...X

            # Introduce core MVars, Q_ABCC and Q_L, from which everything else is derived.
            Q_ABCCX = m.addMVar(unpacked_all_cardinalities, lb=0, name="Q_ABCCX")
            for x in np.ndindex(cardL):
                m.addConstr(Q_ABCCX[...,x].sum() <= 1, name="Unpacked Normalization")
            Q_L = m.addMVar((cardL,), lb=0, name="Q_L")
            m.addConstr(Q_L.sum() == 1, name="Normalization of hidden L")

            # NS constraint
            Q_BCCX =  Q_ABCCX.sum(axis=0)
            m.addConstr(Q_BCCX[...,0] == Q_BCCX[...,1:], name="NS constraint")
            
            # Introduce Q_ACCX, Q_CC, Q_AX to capture nonlinear constraint
            Q_ACCX = Q_ABCCX.sum(axis=1)
            # m.addConstr(Q_ACCX == Q_ABCCX.sum(axis=1), name="Q_ACCX from Q_ABCCX")
            Q_AX = m.addMVar((cardA,cardL), name="Q_AX")
            Q_CC = m.addMVar(unpacked_C_cardinalities, name="Q_CC")
            axes_of_CC = tuple(range(1,cardL+1))
            m.addConstr(Q_AX == Q_ACCX.sum(axis=axes_of_CC), name="Q_AX from Q_ACCX")
            axes_of_A = 0
            m.addConstr(Q_CC == Q_ACCX.sum(axis=axes_of_A)[...,0], name="Q_CC from Q_ACCX")
            
            # Nonlinear constraint: Q_ACCX = Q_AX * Q_CC
            Q_AX_reshaped_for_multiplication = Q_AX.reshape((cardA,) + tuple(repeat(1, times=(cardL))) + (cardL,))
            Q_CC_reshaped_for_multiplication = Q_CC.reshape((1,) + Q_CC.shape + (1,))
            
            m.addConstr(Q_ACCX == Q_AX_reshaped_for_multiplication * Q_CC_reshaped_for_multiplication, name="Q_ACCX = Q_AX * Q_CC")

            # Linear constraint: P_ABC relates to Q_LABCC 
            temp = gp.MQuadExpr.zeros(p_ABC.shape)
            default_axes_to_trace_out = set(range(cardL + 3))
            default_axes_to_trace_out.discard(cardL+2) #removing axis for X
            default_axes_to_trace_out.discard(0) #removing axis for A
            default_axes_to_trace_out.discard(1) #removing axis for B
            for l in rL:
                axes_to_trace_out = tuple(sorted(default_axes_to_trace_out.difference({l+2})))
                temp += Q_L[l] * Q_ABCCX[...,l].sum(axis=axes_to_trace_out)
            m.addConstr(p_ABC == temp, name="P_ABC from Q_LAABCC")
            
            cardY = cardC**cardL
            cardE = cardB
            cardS = cardY
            cardinalities_R = ((cardA,cardB,cardE,cardL,cardY,cardS,))
            R_ABEXYS = m.addMVar(cardinalities_R, lb=0, name="R_ABEXYS")
            for x, y, s in np.ndindex(cardL, cardY, cardY):
                m.addConstr(R_ABEXYS[...,x,y,s].sum() <= 1, name="Normalization") #Needed?
            
            # No-signaling constraints for R
            R_ABXYS = R_ABEXYS.sum(axis=2)
            m.addConstr(R_ABXYS[:,:,:,:,0] == R_ABXYS[:,:,:,:,1:], name="NS for S")
            R_AEXYS = R_ABEXYS.sum(axis=1)
            m.addConstr(R_AEXYS[:,:,:,0,:] == R_AEXYS[:,:,:,1:,:], name="NS for Y")
            R_BEXYS = R_ABEXYS.sum(axis=0)
            m.addConstr(R_BEXYS[:, :, 0, :, :] == R_BEXYS[:, :, 1:, :,:], name="NS for X")

            
            # Conditional of Q for compatibility constraint
            Q_ABcondXCC = R_ABXYS[:,:,:,:,0].reshape((cardA, cardB, cardL)+unpacked_C_cardinalities)
            Q_CC_reshaped_for_conditioning = Q_CC.reshape(Q_CC.shape + tuple(repeat(1, times=cardA)) + tuple(repeat(1, times=cardL)))
            m.addConstr(Q_ABCCX == Q_ABcondXCC*Q_CC_reshaped_for_conditioning)
            
            # need to finish this, one constraint is missing
            for (a,e,x,y) in np.ndindex(cardB, cardE, cardL, cardY):
                if not a==e:
                    m.addConstr(R_BEXYS[a,e,x,y,y] == 0, name="Perfect prediction")

            
            

            m.setObjective(0.0, GRB.MAXIMIZE)
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
