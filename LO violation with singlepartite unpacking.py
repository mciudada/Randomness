import gurobipy as gp
from helpers import create_NS_distribution, check_feasibility

def parse_p_abc_xyz(s):
    """
    Parse a string like
    '000|000 001|000 002|110 ...'
    into a list of tuples:
    (a,b,c,x,y,z)
    where xyz come from the right side and abc from the left.
    """
    terms = s.split()  # split on whitespace
    result = []
    for term in terms:
        left, right = term.split('|')   # e.g. '000', '000'
        xyz = [int(c) for c in left]    # [0, 0, 0]
        abc = [int(c) for c in right]   # [0, 0, 0]
        nums = xyz + abc                # [0, 0, 0, 0, 0, 0]
        result.append(tuple(nums))
    return result

cardA=3
cardB=3
cardC=3
cardX=2
cardY=2
cardZ=2


unpacked_A_cardinalities = tuple([cardA]*cardX)
unpacked_B_cardinalities = tuple([cardB]*cardY)
unpacked_C_cardinalities = tuple([cardC]*cardZ)
print_model=True

with (gp.Env(empty=True) as env):
    env.start()
    with gp.Model("qcp", env=env) as m:
        P_ABC_XYZ= create_NS_distribution(m,
                               outcome_cardinalities=(cardA, cardB, cardC),
                               setting_cardinalities={0: cardX, 1: cardY, 2: cardZ},
                               name="P_ABC_XYZ",
                               impose_normalization=True, impose_nosignalling=True)

        Q_ABCC_XY=create_NS_distribution(m,
                               outcome_cardinalities=((cardA, cardB) + unpacked_C_cardinalities),
                               setting_cardinalities={0: cardX, 1: cardY},
                               name="Q_ABCC_XY",
                               impose_normalization=True, impose_nosignalling=True)
        Q_ABBC_XZ=create_NS_distribution(m,
                               outcome_cardinalities=((cardA,) + unpacked_B_cardinalities + (cardC,)),
                               setting_cardinalities={0: cardX, cardY: cardZ},
                               name="Q_ABBC_XZ",
                               impose_normalization=True, impose_nosignalling=True)
        Q_AABC_YZ=create_NS_distribution(m,
                               outcome_cardinalities=(unpacked_A_cardinalities + (cardB, cardC)),
                               setting_cardinalities={cardX-1: cardY, cardX: cardZ},
                               name="Q_AABC_YZ",
                               impose_normalization=True, impose_nosignalling=True)

        # axes_of_CC_within_ABCC_XY = tuple(range(2, cardZ + 2))
        # for z in range(cardZ):
        #     axes_to_trace_out = axes_of_CC_within_ABCC_XY[:z] + axes_of_CC_within_ABCC_XY[z+1:]
        #     m.addConstr(P_ABC_XYZ[...,:,:,z] == Q_ABCC_XY.sum(axis=axes_to_trace_out), name=f"P_ABC from Q_ABCC_XY for z={z}")

        # axes_of_BB_within_ABBC_XZ = tuple(range(1, cardY + 1))
        # for y in range(cardY):
        #     axes_to_trace_out = axes_of_BB_within_ABBC_XZ[:y] + axes_of_BB_within_ABBC_XZ[y+1:]
        #     m.addConstr(P_ABC_XYZ[...,:,y,:] == Q_ABBC_XZ.sum(axis=axes_to_trace_out), name=f"P_ABC from Q_ABBC_XZ for y={y}")
        #
        # axes_of_AA_within_AABC_YZ = tuple(range(0, cardX))
        # for x in range(cardX):
        #     axes_to_trace_out = axes_of_AA_within_AABC_YZ[:x] + axes_of_AA_within_AABC_YZ[x+1:]
        #     m.addConstr(P_ABC_XYZ[...,x,:,:] == Q_AABC_YZ.sum(axis=axes_to_trace_out), name=f"P_ABC from Q_AABC_YZ for x={x}")
        #
        # ineq = m.addVar(lb=0, ub=cardA*cardB*cardC*cardX*cardY*cardZ, name="ineq")
        ineq_string = "000|000 001|000 002|110 010|000 011|000 012|110 102|110 112|110 120|011 220|011 221|101 222|101"
        ineq_string = "000|001 001|001 002|111 010|001 011|001 110|010 120|010 121|100 122|100 210|010 220|010 221|100 222|100"
        ineq_string = "000|000 001|000 002|110 010|000 011|000 012|110 100|000 101|000 110|000 111|000 120|101 220|101 221|011 222|011"
        ineq_string = "000|000 001|000 002|110 010|000 011|000 012|110 100|000 101|000 102|110 110|000 111|000 112|110 220|011 221|011 222|101"
        ineq_vars = [P_ABC_XYZ[idxs] for idxs in parse_p_abc_xyz(ineq_string)]
        ineq = gp.quicksum(ineq_vars)
        # gp.GRB.INFINITY
        m.setObjective(ineq, gp.GRB.MAXIMIZE)

        # Perform actual optimization
        status_message, variable_values = check_feasibility(m, print_model=False)
        print("Objective value:", m.objVal)
        for var in ineq_vars:
            print(var.VarName, " := ", var.X)
        print("Objective value:", m.objVal)
        m.dispose()
    env.dispose()