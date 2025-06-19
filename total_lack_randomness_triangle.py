import gurobipy as gp
import numpy as np
from helpers import customize_model_for_nonlinear_SAT, create_NS_distribution, check_feasibility
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
            R_ABEab_giv_XYSxy = create_NS_distribution(m,
                                                       outcome_cardinalities=(cardA, cardB, cardE),
                                                       setting_cardinalities=OrderedDict([(0, cardX), (1, cardY), (2, cardS)]),
                                                       name="R_ABEabXYSxy",
                                                       impose_normalization=True, impose_nosignalling=True)
            R_AB_giv_XY = R_ABEab_giv_XYSxy[..., 0].sum(axis=2)
            Q_Y_reshaped_for_conditioning = Q_CC.reshape((1, 1, cardY))
            Q_ABY_giv_X_reshaped_for_conditioning = Q_ABCC_giv_X.reshape((cardA, cardB, cardY, cardX))
            for x in range(cardX):
                m.addConstr(Q_ABY_giv_X_reshaped_for_conditioning[:,:,:,x] == R_AB_giv_XY[:,:,x,:] * Q_Y_reshaped_for_conditioning)


            # Impose Eve can perfectly predict both Alice and Bob
            R_ABEaEb_giv_XYSxSy = R_ABEab_giv_XYSxy.reshape((cardA, cardB, cardEa, cardEb,
                                                             cardX, cardY, cardSx, cardSy))
            for (b, eb, y) in np.ndindex(cardB, cardEb, cardY):
                if not b == eb:
                    m.addConstr(R_ABEaEb_giv_XYSxSy[:, b, :, eb, :, y, :, y] == 0,
                                name="Perfect prediction of Bob")
            for (a, ea, x) in np.ndindex(cardA, cardEa, cardX):
                if not a == ea:
                    m.addConstr(R_ABEaEb_giv_XYSxSy[a, :, ea, :, x, :, x, :] == 0,
                                name="Perfect prediction of Alice")

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
