import gurobipy as gp
import numpy as np
from math import prod
from collections import OrderedDict

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
    m.setParam('FeasibilityTol', 1e-5)  # Adjust as needed
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


def create_distribution(m: gp._model.Model,
                        outcome_cardinalities: tuple[int, ...],
                        setting_cardinalities: dict[int, int],
                        name: str = "P",
                        impose_normalization=False) -> gp._matrixapi.MVar:
    setting_values = tuple(setting_cardinalities.values())
    shape = tuple(outcome_cardinalities) + setting_values
    nof_parties = len(outcome_cardinalities)
    this_MVar = m.addMVar(shape=shape, lb=0, ub=1, name=name)
    if impose_normalization:
        m.addConstr(this_MVar.sum(axis=tuple(range(nof_parties))) == 1)
    return this_MVar


def create_randomless_distribution(m: gp._model.Model,
                                   outcome_cardinalities: tuple[int, ...],
                                   setting_cardinalities: dict[int, int],
                                   name: str = "P",
                                   impose_normalization=False,
                                   import_no_signalling=True,
                                   who_predicted=tuple[int, ...]) -> gp._matrixapi.MVar:
    outcome_cardinalities = tuple(outcome_cardinalities)
    nof_parties = len(outcome_cardinalities)
    nof_parties_with_one_eve = nof_parties + 1
    nof_predictions = len(who_predicted)
    nof_parties_with_eves = nof_parties + nof_predictions
    eve_outcome_cardinalities = tuple(outcome_cardinalities[k] for k in iter(who_predicted))
    eve_setting_cardinalities = tuple(setting_cardinalities[k] for k in iter(who_predicted))
    eve_outcome_cardinality = prod(eve_outcome_cardinalities)
    eve_setting_cardinality = prod(eve_setting_cardinalities)
    outcome_cardinalities_with_one_eve = list(outcome_cardinalities)
    outcome_cardinalities_with_one_eve.append(eve_outcome_cardinality)
    outcome_cardinalities_with_eves = list(outcome_cardinalities)
    outcome_cardinalities_with_eves.extend(eve_outcome_cardinalities)
    setting_cardinalities_with_one_eve = OrderedDict(list(setting_cardinalities.items()))
    setting_cardinalities_with_eves = OrderedDict(list(setting_cardinalities.items()))
    setting_cardinalities_with_one_eve[nof_parties] = eve_setting_cardinality
    for which_eve, what_setting in enumerate(eve_setting_cardinalities):
        setting_cardinalities_with_eves[nof_parties + which_eve] = what_setting
    shape_with_eves = tuple(outcome_cardinalities_with_eves) + tuple(setting_cardinalities_with_eves.values())
    shape_with_one_eve = tuple(outcome_cardinalities_with_one_eve) + tuple(setting_cardinalities_with_one_eve.values())
    party_to_setting_index_with_eves = {party: (i + nof_parties_with_eves)
                                        for i, party in enumerate(setting_cardinalities_with_eves.keys())}

    nonzero_event_collections = set([])
    for event in np.ndindex(shape_with_eves):
        # First let's make sure the event has nonzero probability, and then if so, let's collect all the different versions of the event
        is_probability_zero = False
        for eve, party_being_predicted in enumerate(who_predicted):
            eve_as_party = nof_parties + eve
            where_party_setting = party_to_setting_index_with_eves[party_being_predicted]
            where_eve_setting = party_to_setting_index_with_eves[eve_as_party]
            party_outcome = event[party_being_predicted]
            eve_outcome = event[eve_as_party]
            party_setting = event[where_party_setting]
            eve_setting = event[where_eve_setting]
            if (
                    (party_setting == eve_setting)
                    and (not (party_outcome == eve_outcome))
            ):
                is_probability_zero = True
                break
        if is_probability_zero:
            continue
        # Now, let's try to find all events symmetric to this one
        variant_events = {event}
        for eve, party_being_predicted in enumerate(who_predicted):
            eve_as_party = nof_parties + eve
            where_party_setting = party_to_setting_index_with_eves[party_being_predicted]
            where_eve_setting = party_to_setting_index_with_eves[eve_as_party]
            new_variant_events = set([])
            for variant_event in variant_events:
                party_outcome = variant_event[party_being_predicted]
                eve_outcome = variant_event[eve_as_party]
                party_setting = variant_event[where_party_setting]
                eve_setting = variant_event[where_eve_setting]
                event_copy = list(variant_event)
                event_copy[party_being_predicted] = eve_outcome
                event_copy[eve_as_party] = party_outcome
                event_copy[where_party_setting] = eve_setting
                event_copy[where_eve_setting] = party_setting
                event_copy = tuple(event_copy)
                new_variant_events.add(event_copy)
            variant_events.update(new_variant_events)
        nonzero_event_collections.add(tuple(sorted(variant_events)))

    zero_var = m.addVar(lb=0.0, ub=0.0, name="0", vtype=gp.GRB.CONTINUOUS)
    template_MVar = np.full(fill_value=zero_var, shape=shape_with_eves, dtype=object)

    for event_collection in nonzero_event_collections:
        # print("Event collection", event_collection)
        initial_event = event_collection[0]
        new_var = m.addVar(lb=0, ub=1, name=f"{name}{initial_event}", vtype=gp.GRB.CONTINUOUS)
        for event in event_collection:
            template_MVar[event] = new_var
    template_MVar = template_MVar.reshape(shape_with_one_eve)
    this_MVar = gp.MVar.fromlist(template_MVar)
    if impose_normalization:
        m.addConstr(this_MVar.sum(axis=tuple(range(nof_parties_with_one_eve))) == 1)
    if import_no_signalling:
        constrain_NS(m=m,
                     this_MVar=this_MVar,
                     setting_cardinalities=setting_cardinalities_with_one_eve)
    return this_MVar


def constrain_NS(m: gp._model.Model,
                 this_MVar: gp._matrixapi.MVar,
                 setting_cardinalities: dict[int, int]):
    nof_settings = len(setting_cardinalities)
    nof_axes = this_MVar.ndim
    nof_parties = nof_axes - nof_settings
    all_axes_template = [np.s_[:]] * (nof_axes - 1)
    for i, party in enumerate(setting_cardinalities.keys()):
        marginal = this_MVar.sum(axis=party)
        where_setting = (nof_parties - 1) + i
        LHS = all_axes_template.copy()
        LHS[where_setting] = np.s_[:1]
        RHS = all_axes_template.copy()
        RHS[where_setting] = np.s_[1:]
        m.addConstr(marginal[tuple(LHS)] == marginal[tuple(RHS)])
    return this_MVar


def create_NS_distribution(m: gp._model.Model,
                           outcome_cardinalities: tuple[int, ...],
                           setting_cardinalities: dict[int, int],
                           name: str = "P",
                           impose_normalization=False,
                           impose_nosignalling=True) -> gp._matrixapi.MVar:
    this_MVar = create_distribution(m=m,
                                    outcome_cardinalities=outcome_cardinalities,
                                    setting_cardinalities=setting_cardinalities,
                                    name=name,
                                    impose_normalization=impose_normalization)
    if impose_nosignalling:
        constrain_NS(m=m,
                     this_MVar=this_MVar,
                     setting_cardinalities=setting_cardinalities)
    return this_MVar


def check_feasibility(m: gp._model.Model,
                      print_model=False) -> (str, dict):
    m.update()
    m.optimize()
    status_message = status_dict.get(m.status, f"Unknown status ({m.status})")
    print(f"Model status: {m.status} - {status_message}")
    record_to_preserve = dict()
    if m.getAttr("SolCount"):
        for var in m.getVars():
            record_to_preserve[var.VarName] = var.X
            if print_model and np.abs(var.X) > 1e-5:
                print(var.VarName, " := ", var.X)
    return status_message, record_to_preserve
