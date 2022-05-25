import numpy as np
from felix.src.candidate_creator.candidate import Candidate
from felix.src.algorithms.run_foeir import solve_lp_with_DTC


def test_individual_fairness_constraint():
    ranking = [
        Candidate(
            qualification=1,
            originalQualification=1,
            protectedAttributes=None,
            query=1,
            features=None,
            outlier_feature=0,
        ),
        Candidate(
            qualification=1,
            originalQualification=0.8,
            protectedAttributes=None,
            query=1,
            features=None,
            outlier_feature=0,
        ),
    ]
    a = solve_lp_with_DTC(
        candidates=ranking,
        top_k=len(ranking),
        outlier_objective=False,
        individual_fairness=True,
    )[0]
    assert np.allclose(a, [[0.745501], [1 - 0.745501], [1 - 0.745501], [0.745501]])
    return True
