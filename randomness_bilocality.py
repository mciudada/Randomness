"""Certify randomness in the bilocality scenario against a non-signalling adversary.

Usage:
    python randomness_bilocality.py --party A --distribution ent
    python randomness_bilocality.py --party B --distribution fritz
"""

import argparse

import numpy as np
from inflation import InflationProblem, InflationLP

from probabilities import prob_ent, prob_Fritz

# Eve's guessing-probability objective, one per party she tries to guess.
OBJECTIVES = {
    "A": {"1": 1, "pAE(00|10)": 2, "pE(0|0)": -1, "pA(0|1)": -1},
    "B": {"1": 1, "pBE(00|00)": 2, "pE(0|0)": -1, "pB(0|0)": -1},
    "C": {"1": 1, "pCE(00|00)": 2, "pE(0|0)": -1, "pC(0|0)": -1},
}

DISTRIBUTIONS = {
    "ent": prob_ent,      # entanglement swapping
    "fritz": prob_Fritz,  # Fritz-style correlation
}


def build_problem(eve_outcomes=2, verbose=0):
    """The A-B-C chain with Eve sharing both sources."""
    chain_Eve = InflationProblem(
        dag={"rho_AB": ["A", "B", "E"],
             "rho_BC": ["B", "C", "E"]},
        classical_sources=None,
        outcomes_per_party=(2, 2, 2, eve_outcomes),
        settings_per_party=(2, 1, 2, 1),
        inflation_level_per_source=(2, 2),
        order=["A", "B", "C", "E"],
    )
    return InflationLP(chain_Eve, include_all_outcomes=False, verbose=verbose)


def certify(party, distribution, eve_outcomes=2, verbose=0):
    """Maximise Eve's guessing probability for `party`. Below 1 certifies randomness."""
    inf_lp = build_problem(eve_outcomes=eve_outcomes, verbose=verbose)

    p = np.zeros((2, 2, 2, 2, 1, 2))
    p[:, :, :, :, 0, :] = DISTRIBUTIONS[distribution]()

    values = {m.name: m.compute_marginal(p)
              for m in inf_lp.knowable_atoms if "E" not in m.name}

    inf_lp.set_objective(OBJECTIVES[party], direction="max")
    inf_lp.set_values(values, use_lpi_constraints=True)
    inf_lp.solve(solve_dual=False)
    return inf_lp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--party", choices=sorted(OBJECTIVES), default="A",
                        help="which party Eve tries to guess")
    parser.add_argument("--distribution", choices=sorted(DISTRIBUTIONS), default="ent")
    parser.add_argument("--eve-outcomes", type=int, default=2,
                        help="cardinality of Eve's outcome (the paper also uses 4)")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    inf_lp = certify(args.party, args.distribution,
                     eve_outcomes=args.eve_outcomes, verbose=args.verbose)

    print(f"Guessing probability bound for {args.party} "
          f"({args.distribution}): {inf_lp.objective_value}")

    if args.verbose:
        for m in inf_lp.knowable_atoms:
            print(f"{m.name} -> {inf_lp.solution_object['x'][m.name]}")
    
