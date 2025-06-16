import gurobipy as gp
import numpy as np

status_dict = {
    gp.GRB.OPTIMAL: "Optimal solution found",
    gp.GRB.UNBOUNDED: "Model is unbounded",
    gp.GRB.INFEASIBLE: "Model is infeasible",
    gp.GRB.INF_OR_UNBD: "Model is infeasible or unbounded",
    gp.GRB.INTERRUPTED: "Optimization was interrupted",
    gp.GRB.TIME_LIMIT: "Time limit reached",
    gp.GRB.SUBOPTIMAL: "Suboptimal solution found",
    gp.GRB.USER_OBJ_LIMIT: "User objective limit reached",
    gp.GRB.NUMERIC: "Numerical issues",
}

def customize_model_for_nonlinear_SAT(m: gp._model.Model):
    m.setParam('NonConvex', 2)
    m.setParam('FeasibilityTol', 1e-5) #Adjust as needed
    m.setParam('OptimalityTol', 0.01)
    m.setParam('TimeLimit', gp.GRB.INFINITY)
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
    m.setObjective(0.0, gp.GRB.MAXIMIZE)

def create_NS_distribution(m: gp._model.Model,
                           outcome_cardinalities: tuple[int,...],
                           setting_cardinalities: dict[int, int],
                           name: str = "P",
                           impose_normalization=False,
                           impose_nosignalling=True) -> gp._matrixapi.MVar:
    setting_values = tuple(setting_cardinalities.values())
    shape = tuple(outcome_cardinalities)+setting_values
    nof_outcomes = len(outcome_cardinalities)
    nof_settings = len(setting_values)
    nof_axes = nof_outcomes + nof_settings
    this_MVar = m.addMVar(shape=shape, lb=0, ub=1, name=name)
    if impose_normalization:
        m.addConstr( this_MVar.sum(axis=tuple(range(nof_outcomes))) == 1 )
    if nof_outcomes > 1 and nof_settings > 0:
        if impose_nosignalling:
            all_axes_template = [np.s_[:]] * (nof_axes - 1)
            for i, party in enumerate(setting_cardinalities.keys()):
                marginal = this_MVar.sum(axis=party)
                where_setting = (nof_outcomes-1)+i
                LHS = all_axes_template.copy()
                LHS[where_setting] = np.s_[:1]
                RHS = all_axes_template.copy()
                RHS[where_setting] = np.s_[1:]
                m.addConstr(marginal[tuple(LHS)] == marginal[tuple(RHS)])
    return this_MVar

# def shapes_for_factorization(where_dims1: tuple[tuple[int,...], tuple[int,...]],
#                              where_dims2: tuple[tuple[int,...], tuple[int,...]]) -> tuple[tuple[int,...], tuple[int,...]]:
#     indices1, dimensions1 = where_dims1
#     indices2, dimensions2 = where_dims2
#     combined_indices = tuple(sorted(indices1+indices2))
#     where1 = np.isin(combined_indices, indices1)
#     where2 = np.logical_not(where1)
#     template1 = np.ones(len(combined_indices), dtype=int)
#     template2 = template1.copy()
#     template1[where1] = dimensions1
#     template2[where2] = dimensions2
#     shape_1 = tuple(template1.tolist())
#     shape_2 = tuple(template2.tolist())
#     return shape_1, shape_2

def check_feasibility(m: gp._model.Model,
                      print_model=False) -> str:
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
    return status_message
