import gurobipy as gp
import numpy as np
from helpers import customize_model_for_nonlinear_SAT, create_randomless_distribution, check_feasibility
from triangle_two_classical_sources import impose_two_classical_sources
from collections import OrderedDict


def check_total_lack_randomness(p_ABC: np.ndarray, cardX: int, print_model=False) -> str:
    with (gp.Env(empty=True) as env):
        env.start()
        with gp.Model("qcp", env=env) as m:
            customize_model_for_nonlinear_SAT(m)

            (Q_ABCC_giv_X, Q_X, Q_CC) = impose_two_classical_sources(m, p_ABC, cardX)

            (cardA, cardB, cardC) = p_ABC.shape
            cardY = cardC ** cardX
            cardEa = cardA
            cardSx = cardX
            cardEb = cardB
            cardSy = cardY
            cardE = cardEa * cardEb
            cardS = cardSx * cardSy

            # Conditional of Q for compatibility constraint
            R_ABEab_giv_XYSxy = create_randomless_distribution(m,
                                                               outcome_cardinalities=(cardA, cardB),
                                                               setting_cardinalities={0: cardX, 1: cardY},
                                                               name="R_ABEabXYSxy",
                                                               who_predicted=(0, 1))
            R_AB_giv_XY = R_ABEab_giv_XYSxy[..., 0].sum(axis=2)
            Q_Y_reshaped_for_conditioning = Q_CC.reshape((1, 1, cardY))
            Q_ABY_giv_X_reshaped_for_conditioning = Q_ABCC_giv_X.reshape((cardA, cardB, cardY, cardX))
            for x in range(cardX):
                m.addConstr(Q_ABY_giv_X_reshaped_for_conditioning[:, :, :, x] == R_AB_giv_XY[:, :, x,
                                                                                 :] * Q_Y_reshaped_for_conditioning)


            # Perform actual optimization
            status_message = check_feasibility(m, print_model=print_model)

            m.dispose()
        env.dispose()
    return status_message


if __name__ == "__main__":
    from probabilities import RGB3_specific

    print(check_total_lack_randomness(p_ABC=RGB3_specific,
                                      cardX=2,
                                      print_model=True)[0])
