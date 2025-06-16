import gurobipy as gp
import numpy as np
from helpers import customize_model_for_nonlinear_SAT, create_NS_distribution, check_feasibility
from triangle_two_classical_sources import impose_two_classical_sources


def check_total_lack_randomness(p_ABC: np.ndarray, cardX: int, print_model=False) -> str:
    with (gp.Env(empty=True) as env):
        env.start()
        with gp.Model("qcp", env=env) as m:
            customize_model_for_nonlinear_SAT(m)

            (Q_ABCCX, Q_X, Q_AX, Q_CC) = impose_two_classical_sources(m, p_ABC, cardX)

            (cardA, cardB, cardC) = p_ABC.shape
            cardY = cardC ** cardX
            cardEa = cardA
            cardSx = cardX
            cardEb = cardB
            cardSy = cardY
            cardE = cardEa * cardEb
            cardS = cardSx * cardSy

            # Conditional of Q for compatibility constraint
            R_ABYX_shape = (cardA, cardB, cardY, cardX)
            R_ABYX = m.addMVar(R_ABYX_shape, lb=0, ub=1, name="R_ABYX")  # Weird order on purpose
            Q_CC_reshaped_for_conditioning = Q_CC.reshape((1, 1, cardY, 1))
            Q_ABCCX_reshaped_for_conditioning = Q_ABCCX.reshape(R_ABYX.shape)
            m.addConstr(Q_ABCCX_reshaped_for_conditioning == R_ABYX * Q_CC_reshaped_for_conditioning)
            # Relate R_ABYX to RABEXYS
            R_ABEabXYSxy = create_NS_distribution(m,
                                                  outcome_cardinalities=(cardA, cardB, cardE),
                                                  setting_cardinalities={0: cardX, 1: cardY, 2: cardS},
                                                  name="R_ABEabXYSxy",
                                                  impose_normalization=True, impose_nosignalling=True)
            R_ABXY = R_ABEabXYSxy[..., 0].sum(axis=2)
            for (x, y) in np.ndindex(cardX, cardY):
                m.addConstr(R_ABXY[:, :, x, y] == R_ABYX[:, :, y, x], name="R_ABYX transposed R_ABYX")

            # Impose Eve can perfectly predict both Alice and Bob
            R_ABEabXYSxy_reshaped_for_perfect_prediction = R_ABEabXYSxy.reshape((cardA, cardB, cardEa, cardEb,
                                                                                 cardX, cardY, cardSx, cardSy))
            for (b, eb, y) in np.ndindex(cardB, cardEb, cardY):
                if not b == eb:
                    m.addConstr(R_ABEabXYSxy_reshaped_for_perfect_prediction[:, b, :, eb, :, y, :, y] == 0,
                                name="Perfect prediction of Bob")
            for (a, ea, x) in np.ndindex(cardA, cardEa, cardX):
                if not a == ea:
                    m.addConstr(R_ABEabXYSxy_reshaped_for_perfect_prediction[a, :, ea, :, x, :, x, :] == 0,
                                name="Perfect prediction of Alice")

            # Perform actual optimization
            status_message = check_feasibility(m, print_model=print_model)

            m.dispose()
        env.dispose()
    return status_message


if __name__ == "__main__":
    from probabilities import prob_RGB3

    RGB3_specific = prob_RGB3(1 / np.sqrt(2), 0.95)
    print(check_total_lack_randomness(p_ABC=RGB3_specific,
                                      cardX=2,
                                      print_model=True))
