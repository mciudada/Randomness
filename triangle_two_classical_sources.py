from typing_extensions import Unpack

import gurobipy as gp
import numpy as np
from helpers import customize_model_for_nonlinear_SAT, create_NS_distribution, check_feasibility


def impose_two_classical_sources(m: gp._model.Model,
                                 p_ABC: np.ndarray,
                                 cardX: int):
    (cardA, cardB, cardC) = p_ABC.shape
    # Define fully unpacked probabilities
    unpacked_C_cardinalities = tuple([cardC]*cardX)
    # Instantiate variables
    Q_ABCC_X = create_NS_distribution(m,
                                     outcome_cardinalities=((cardA, cardB) + unpacked_C_cardinalities),
                                     setting_cardinalities={0: cardX},
                                     name="Q_ABCC_X",
                                     impose_normalization=True, impose_nosignalling=True)
    Q_X = create_NS_distribution(m,
                                 outcome_cardinalities=(cardX,),
                                 setting_cardinalities=dict(),
                                 name="Q_X",
                                 impose_normalization=True, impose_nosignalling=False)
    Q_CC = create_NS_distribution(m,
                                  outcome_cardinalities=unpacked_C_cardinalities,
                                  setting_cardinalities=dict(),
                                  name="Q_CC",
                                  impose_normalization=False, impose_nosignalling=False)

    # # Optional constraints known to preserve feasibility for RGB3, to accelerate solution finding
    # m.addConstr(Q_CC[0, 0] == 0)
    # m.addConstr(Q_CC[0, 1] == 0)
    # m.addConstr(Q_CC[1, 1] == 0)
    # m.addConstr(Q_CC[1, 2] == 0)
    # m.addConstr(Q_CC[2, 0] == 0)
    # m.addConstr(Q_CC[2, 1] == 0)
    # m.addConstr(Q_CC[2, 2] == 0)

    # Extract marginal variables
    cardY = cardC ** cardX
    Q_ACC_X = Q_ABCC_X.reshape((cardA,cardB,cardY,cardX)).sum(axis=1)
    Q_A_X = Q_ACC_X.sum(axis=1)

    # Define one new marginal variable for factorization convenience
    m.addConstr(Q_CC.reshape(cardY) == Q_ACC_X[..., 0].sum(axis=0), name="Q_CC from Q_ACCX")

    Q_CC_reshaped_for_multiplication = Q_CC.reshape((1,cardY,1))
    Q_A_X_reshaped_for_multiplication = Q_A_X[np.s_[:], np.newaxis, np.s_[:]]
    m.addConstr(Q_ACC_X == Q_A_X_reshaped_for_multiplication * Q_CC_reshaped_for_multiplication, name=f"Q_ACCX = Q_AX * Q_CC")

    # Linear constraint: P_ABC relates to Q_ABCCX
    temp = gp.MQuadExpr.zeros(p_ABC.shape)
    axes_of_CC_within_ABCCX = tuple(range(2, cardX + 2))
    for x in range(cardX):
        axes_to_trace_out = axes_of_CC_within_ABCCX[:x] + axes_of_CC_within_ABCCX[x+1:]
        temp += Q_X[x] * Q_ABCC_X[..., x].sum(axis=axes_to_trace_out)
    m.addConstr(p_ABC == temp, name="P_ABC from Q_ABCCX")
    return Q_ABCC_X, Q_X, Q_CC


def check_singlepartite_lack_randomness(p_ABC: np.ndarray, cardX: int, print_model=False) -> str:
    with (gp.Env(empty=True) as env):
        env.start()
        with gp.Model("qcp", env=env) as m:
            customize_model_for_nonlinear_SAT(m)
            Q_ABCCX, Q_X, Q_CC = impose_two_classical_sources(m, p_ABC, cardX)
            # Perform actual optimization
            status_message, variable_values = check_feasibility(m, print_model=print_model)
            m.dispose()
        env.dispose()
    return status_message


if __name__ == "__main__":
    from probabilities import RGB3_specific
    print(check_singlepartite_lack_randomness(p_ABC=RGB3_specific,
                                               cardX=2,
                                               print_model=True)[0])
