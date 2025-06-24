import gurobipy as gp
import numpy as np
from helpers import customize_model_for_nonlinear_SAT, create_NS_distribution, check_feasibility
from collections import OrderedDict


def impose_BC_classical_and_Bob_predictable(m: gp._model.Model,
                                            p_ABC_XZ: np.ndarray):
    (cardA, cardB, cardC, cardX, cardZ) = p_ABC_XZ.shape
    # Define fully unpacked probabilities
    unpacked_A_cardinalities = tuple([cardA] * cardX)
    As_as_one_cardinality = int(np.prod(unpacked_A_cardinalities).tolist())
    # Instantiate variable
    Q_AABC_Z = create_NS_distribution(m,
                                      outcome_cardinalities=(unpacked_A_cardinalities + (cardB, cardC)),
                                      setting_cardinalities={cardX + 1: cardZ},
                                      name="Q_AABC_Z",
                                      impose_normalization=True, impose_nosignalling=True)

    Q_AAC_Z = Q_AABC_Z.reshape((As_as_one_cardinality, cardB, cardC, cardZ)).sum(axis=1)
    Q_C_Z = Q_AAC_Z.sum(axis=0)
    Q_AA = m.addMVar(shape=(As_as_one_cardinality,), lb=0, ub=1, name="Q_AA")
    m.addConstr(Q_AA[..., np.newaxis] == Q_AAC_Z.sum(axis=1),
                name=f"Q_AA from Q_AAC_Z")
    Q_AA_reshaped_for_multiplication = Q_AA.reshape((As_as_one_cardinality, 1, 1))
    Q_C_Z_reshaped_for_multiplication = Q_C_Z[np.newaxis, np.s_[:], np.s_[:]]
    m.addConstr(Q_AAC_Z == Q_C_Z_reshaped_for_multiplication * Q_AA_reshaped_for_multiplication,
                name=f"Q_AAC_Z = Q_AA * Q_C_Z")

    # Linear constraint: P_ABC_XZ relates to Q_AABC_Z
    axes_of_AA_within_AABC_Z = tuple(range(cardX))
    for x in range(cardX):
        axes_to_trace_out = axes_of_AA_within_AABC_Z[:x] + axes_of_AA_within_AABC_Z[x + 1:]
        m.addConstr(p_ABC_XZ[:, :, :, x, :] == Q_AABC_Z.sum(axis=axes_to_trace_out), name="P_ABC_XZ from Q_AABC_Z")

    # Next: impose Bob predicatable
    # Conditional of Q for compatibility constraint
    cardY = As_as_one_cardinality
    cardE = cardB
    cardS = cardY
    R_BCE_YZS = create_NS_distribution(m,
                                       outcome_cardinalities=(cardB, cardC, cardE),
                                       setting_cardinalities=OrderedDict([(0, cardY), (1, cardZ), (2, cardS)]),
                                       name="R_BCE_YZS",
                                       impose_normalization=True, impose_nosignalling=True)
    R_BC_YZ = R_BCE_YZS[..., 0].sum(axis=2)
    Q_YBC_Z_reshaped_for_conditioning = Q_AABC_Z.reshape((cardY, cardB, cardC, cardZ))
    for y in range(cardY):
        m.addConstr(Q_YBC_Z_reshaped_for_conditioning[y, :, :, :] == R_BC_YZ[:, :, y, :] * Q_AA[y])

    # Impose Eve can perfectly predict Bob
    for (b, e, y) in np.ndindex(cardB, cardE, cardY):
        if not b == e:
            m.addConstr(R_BCE_YZS[b, :, e, y, :, y] == 0,
                        name="Perfect prediction of Bob")
    # Impose Eve acts like symmetric extension when settings do not match.
    for (e, b, y, s) in np.ndindex(cardB, cardY, cardS):
        if b>e or y>s:
            m.addConstr(R_BCE_YZS[b, :, e, y, :, s] == R_BCE_YZS[e, :, b, s, :, y],
                        name="Eve as symmetric extension.")

    return Q_AABC_Z, Q_AA, R_BCE_YZS


def check_Bob_lack_randomness(p_ABC_XZ: np.ndarray, print_model=False) -> str:
    with (gp.Env(empty=True) as env):
        env.start()
        with gp.Model("qcp", env=env) as m:
            customize_model_for_nonlinear_SAT(m)
            Q_AABC_Z, Q_AA, R_BCE_YZS = impose_BC_classical_and_Bob_predictable(m, p_ABC_XZ)
            # Perform actual optimization
            status_message = check_feasibility(m, print_model=print_model)
            m.dispose()
        env.dispose()
    return status_message


if __name__ == "__main__":
    from probabilities import MNN_specific

    print(check_Bob_lack_randomness(p_ABC_XZ=MNN_specific,
                                    print_model=True)[0])
