import gurobipy as gp
from itertools import repeat
import numpy as np
from helpers import customize_model_for_nonlinear_SAT, create_NS_distribution, check_feasibility


def impose_two_classical_sources(m: gp._model.Model,
                                 p_ABC: np.ndarray,
                                 cardX: int):
    (cardA, cardB, cardC) = p_ABC.shape
    # Define fully unpacked probabilities
    unpacked_C_cardinalities = tuple(repeat(cardC, times=cardX))
    # Instantiate variables
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
                                  impose_normalization=False, impose_nosignalling=False)
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

    Q_ACCX = Q_ABCCX.sum(axis=1)
    axes_of_CC_within_ACCX = tuple(range(1, cardX + 1))
    axes_of_AX_within_ACCX = (0, cardX + 1)
    m.addConstr(Q_AX == Q_ACCX.sum(axis=axes_of_CC_within_ACCX), name="Q_AX from Q_ACCX")
    m.addConstr(Q_CC == Q_ACCX[..., 0].sum(axis=axes_of_AX_within_ACCX[:1]), name="Q_CC from Q_ACCX")

    AX_newshape = np.ones(Q_ACCX.ndim, dtype=int)
    AX_newshape[list(axes_of_AX_within_ACCX)] = Q_AX.shape
    AX_newshape = tuple(AX_newshape.tolist())
    CC_newshape = np.ones(Q_ACCX.ndim, dtype=int)
    CC_newshape[list(axes_of_CC_within_ACCX)] = Q_CC.shape
    CC_newshape = tuple(CC_newshape.tolist())

    Q_AX_reshaped_for_multiplication = Q_AX.reshape(AX_newshape)
    Q_CC_reshaped_for_multiplication = Q_CC.reshape(CC_newshape)
    m.addConstr(Q_ACCX == Q_AX_reshaped_for_multiplication * Q_CC_reshaped_for_multiplication,
                name="Q_ACCX = Q_AX * Q_CC")

    # Linear constraint: P_ABC relates to Q_LABCC
    axes_of_CC_within_ABCCX = tuple(range(2, cardX + 2))
    temp = gp.MQuadExpr.zeros(p_ABC.shape)
    for x in range(cardX):
        axes_to_trace_out = axes_of_CC_within_ABCCX[:x] + axes_of_CC_within_ABCCX[x + 1:]
        temp += Q_X[x] * Q_ABCCX[..., x].sum(axis=axes_to_trace_out)
    m.addConstr(p_ABC == temp, name="P_ABC from Q_LAABCC")
    return Q_ABCCX, Q_X, Q_AX, Q_CC


def check_singlepartite_lack_randomness(p_ABC: np.ndarray, cardX: int, print_model=False) -> str:
    with (gp.Env(empty=True) as env):
        env.start()
        with gp.Model("qcp", env=env) as m:
            customize_model_for_nonlinear_SAT(m)
            Q_ABCCX, Q_X, Q_AX, Q_CC = impose_two_classical_sources(m, p_ABC, cardX)
            # Perform actual optimization
            status_message = check_feasibility(m, print_model=print_model)
            m.dispose()
        env.dispose()
    return status_message


if __name__ == "__main__":
    from probabilities import prob_RGB3

    RGB3_specific = prob_RGB3(1 / np.sqrt(2), 0.95)
    print(check_singlepartite_lack_randomness(p_ABC=RGB3_specific,
                                               cardX=2,
                                               print_model=True))
