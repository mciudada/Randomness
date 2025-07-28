import gurobipy as gp
import numpy as np
from helpers import customize_model_for_nonlinear_SAT, create_randomless_distribution, check_feasibility
from triangle_two_classical_sources import impose_two_classical_sources
from collections import OrderedDict


def check_bipartite_lack_randomness(p_ABC: np.ndarray, cardX: int, print_model=False) -> str:
    with (gp.Env(empty=True) as env):
        env.start()
        with gp.Model("qcp", env=env) as m:
            customize_model_for_nonlinear_SAT(m)

            (Q_ABCC_X, Q_X, Q_CC) = impose_two_classical_sources(m, p_ABC, cardX)

            (cardA, cardB, cardC) = p_ABC.shape
            cardY = cardC ** cardX

            R_ABE_XYS = create_randomless_distribution(m,
                                                       outcome_cardinalities=(cardA, cardB),
                                                       setting_cardinalities={0: cardX, 1: cardY},
                                                       who_predicted=(0,),
                                                       name="R_ABE_XYS")
            R_AB_XY = R_ABE_XYS[..., 0].sum(axis=2)
            Q_Y_reshaped_for_conditioning = Q_CC.reshape((1, 1, cardY))
            Q_ABY_X_reshaped_for_conditioning = Q_ABCC_X.reshape((cardA, cardB, cardY, cardX))
            for x in range(cardX):
                m.addConstr(Q_ABY_X_reshaped_for_conditioning[:,:,:,x] == R_AB_XY[:,:,x,:] * Q_Y_reshaped_for_conditioning)


            # Perform actual optimization
            status_message, variable_values = check_feasibility(m, print_model=print_model)

            m.dispose()
        env.dispose()
    return status_message



if __name__ == "__main__":
    from probabilities import RGB3_specific
    print(check_bipartite_lack_randomness(p_ABC=RGB3_specific,
                                          cardX=2,
                                          print_model=True)[0])
