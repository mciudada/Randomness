import gurobipy as gp
from helpers import status_dict, customize_model_for_nonlinear_SAT, create_NS_distribution
from itertools import repeat
import numpy as np

time_limit = GRB.INFINITY
tol = 1e-5
return_dist = True
print_model = True


def check_total_lack_randomness(p_ABC: np.ndarray, cardX: int) -> bool:
    (cardA, cardB, cardC) = p_ABC.shape
    rL = tuple(range(cardX))

    with (gp.Env(empty=True) as env):
        env.start()
        with gp.Model("qcp", env=env) as m:
            customize_model_for_nonlinear_SAT(m)

            # Define fully unpacked probabilities
            unpacked_C_cardinalities = tuple(repeat(cardC, times=cardX))
            unpacked_all_cardinalities = (cardA, cardB) + unpacked_C_cardinalities + (cardX,)  # this corresponds to ABC0C1...X

            Q_ABCCX = create_NS_distribution(m,
                                             outcome_cardinalities=((cardA, cardB) + unpacked_C_cardinalities),
                                             setting_cardinalities={0: cardX},
                                             name="Q_ABCCX",
                                             impose_normalization=True)
            Q_X = create_NS_distribution(m,
                                             outcome_cardinalities=(cardX,),
                                             setting_cardinalities=dict(),
                                             name="Q_X",
                                             impose_normalization=True)
            Q_AX = create_NS_distribution(m,
                                          outcome_cardinalities=(cardA,),
                                          setting_cardinalities={0: cardX},
                                          name="Q_AX",
                                          impose_normalization=False)
            Q_CC = create_NS_distribution(m,
                                          outcome_cardinalities=unpacked_C_cardinalities,
                                          setting_cardinalities=dict(),
                                          name="Q_CC",
                                          impose_normalization=False)
            axes_of_CC = tuple(range(2, cardX + 2))
            m.addConstr(Q_AX == Q_ABCCX.sum(axis=axes_of_CC), name="Q_AX from Q_ABCCX")
            axes_of_A = (0, 1)
            m.addConstr(Q_CC == Q_ABCCX[..., 0].sum(axis=axes_of_A), name="Q_CC from Q_ABCCX")
            Q_ACCX = Q_ABCCX.sum(axis=1)
            Q_AX_reshaped_for_multiplication = Q_AX.reshape((cardA,) + tuple(repeat(1, times=cardX)) + (cardX,))
            Q_CC_reshaped_for_multiplication = Q_CC.reshape((1,) + Q_CC.shape + (1,))
            m.addConstr(Q_ACCX == Q_AX_reshaped_for_multiplication * Q_CC_reshaped_for_multiplication,
                        name="Q_ACCX = Q_AX * Q_CC")



            # Linear constraint: P_ABC relates to Q_LABCC
            temp = gp.MQuadExpr.zeros(p_ABC.shape)
            default_axes_to_trace_out = set(range(cardX + 3))
            default_axes_to_trace_out.discard(cardX + 2)  # removing axis for X
            default_axes_to_trace_out.discard(0)  # removing axis for A
            default_axes_to_trace_out.discard(1)  # removing axis for B
            for l in rL:
                axes_to_trace_out = tuple(sorted(default_axes_to_trace_out.difference({l + 2})))
                temp += Q_X[l] * Q_ABCCX[..., l].sum(axis=axes_to_trace_out)
            m.addConstr(p_ABC == temp, name="P_ABC from Q_LAABCC")

            cardY = cardC ** cardX
            cardEa = cardA
            cardSx = cardX
            cardEb = cardB
            cardSy = cardY
            cardE = cardEa * cardEb
            cardS = cardSx * cardSy

            R_ABEabXYSxy = create_NS_distribution(m,
                                             outcome_cardinalities=(cardA, cardB, cardE),
                                             setting_cardinalities={0: cardX, 1: cardY, 2: cardS},
                                             name="R_ABEabXYSxy",
                                             impose_normalization=True)


            # Conditional of Q for compatibility constraint
            R_ABYX = m.addMVar((cardA, cardB, cardY, cardX), lb=0, ub=1, name="R_ABYX")  # Weird order on purpose
            R_ABXY = R_ABEabXYSxy[...,0].sum(axis=2)
            for (x, y) in np.ndindex(cardX, cardY):
                m.addConstr(R_ABXY[:, :, x, y] == R_ABYX[:, :, y, x], name="R_ABYX transposed R_ABYX")
            Q_CC_reshaped_for_conditioning = Q_CC.reshape((1, 1, cardY, 1))
            Q_ABCCX_reshaped_for_conditioning = Q_ABCCX.reshape(R_ABYX.shape)
            m.addConstr(Q_ABCCX_reshaped_for_conditioning == R_ABYX * Q_CC_reshaped_for_conditioning)

            # need to finish this, one constraint is missing
            R_ABEabXYSxy_reshaped_for_perfect_prediction = R_ABEabXYSxy.reshape((cardA, cardB, cardEa, cardEb,
                                                                                 cardX, cardY, cardSx, cardSy))
            for (b, eb, y) in np.ndindex(cardB, cardEb, cardY):
                if not b == eb:
                    m.addConstr(R_ABEabXYSxy_reshaped_for_perfect_prediction[:, b, :, eb, :, y, :, y] == 0, name="Perfect prediction of Bob")
            for (a, ea, x) in np.ndindex(cardA, cardEa, cardX):
                if not a == ea:
                    m.addConstr(R_ABEabXYSxy_reshaped_for_perfect_prediction[a, :, ea, :, x, :, x, :] == 0, name="Perfect prediction of Alice")


            m.update()
            m.optimize()
            status_message = status_dict.get(m.status, f"Unknown status ({m.status})")
            print(f"Model status: {m.status} - {status_message}")
            if m.getAttr("SolCount"):
                record_to_preserve = dict()
                for var in m.getVars():
                    record_to_preserve[var.VarName] = var.X
                    if print_model:
                        print(var.VarName, " := ", var.X)

            m.dispose()
        env.dispose()
    return status_message


if __name__ == "__main__":
    from probabilities import prob_RGB3

    RGB3_specific = prob_RGB3(1 / np.sqrt(2), 0.95)
    print("Normalization check:", RGB3_specific.sum())

    print(check_total_lack_randomness(RGB3_specific, cardX=2))