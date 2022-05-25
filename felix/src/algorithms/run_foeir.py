"""
Parts of this code refer to https://github.com/MilkaLichtblau/BA_Laura
"""

import numpy as np
from cvxopt import spmatrix, matrix, sparse, solvers
from felix.src.algorithms.stochastic_policy import (
    StochasticPolicy,
    is_doubly_stochastic_matrix,
)
from felix.src.measures.outlier_metrics import determine_outlier_vector

solvers.options["show_progress"] = False


def runFOEIR(
    candidates,
    outlier_objective=False,
    top_k=None,
    decomposition_method="vanilla_BvN",
    individual_fairness=True,
    number_of_resamples=10,
    mrp_matrix=None,
):

    """
    Start the calculation of the ranking for fairness of exposure under the
    Disparate Treatment (DT) fairness constraint.

    Returns the stochastic ranking policy and a boolean signaling whether the
    calculation was successful.
    """

    n = len(candidates)
    if not top_k:
        top_k = len(candidates)

    if mrp_matrix is None:
        x, isRanked = solve_lp_with_DTC(
            candidates,
            top_k=top_k,
            outlier_objective=outlier_objective,
            individual_fairness=individual_fairness,
        )
        if individual_fairness:
            print("We use individual fairness")
        if isRanked == True:

            x = np.reshape(x, (n, top_k))
            x = np.asarray(x, dtype="float64")
            if not is_doubly_stochastic_matrix(x):
                print("MRP is not doubly stochastic")

            stochastic_policy = StochasticPolicy.from_mrp(
                x,
                candidates=candidates,
                decomposition_method=decomposition_method,
                number_of_resamples=number_of_resamples,
            )

            return stochastic_policy, isRanked

    else:
        stochastic_policy = StochasticPolicy.from_mrp(
            mrp_matrix,
            candidates=candidates,
            decomposition_method=decomposition_method,
            number_of_resamples=number_of_resamples,
        )
        return stochastic_policy, True
    return None, False


def solve_lp_with_DTC(
    candidates,
    top_k=None,
    outlier_window_size=10,
    outlier_objective=False,
    individual_fairness=True,
):
    """
    Solve the linear program with the disparate treatment constraint.

    Returns doubly stochastic matrix as numpy array
    """

    n = len(candidates)
    if top_k is None:
        top_k = n

    print("Start building LP with DTC.")
    learned_scores = []
    unprotected_aggregated_scores = 0
    protected_aggregated_scores = 0
    proCount = 0
    unproCount = 0
    pro_indices = []
    unpro_indicex = []

    for candidate in candidates:
        learned_scores.append(candidate.learnedScores)

    # initialize position-based exposure v
    position_bias = np.arange(1, (top_k + 1), 1)
    position_bias = 1 / np.log2(1 + position_bias)
    position_bias = np.reshape(position_bias, (1, top_k))

    learned_scores = np.asarray(learned_scores)

    I = []
    J = []
    I2 = []
    J2 = []

    # set up indices for column and row constraints
    for j in range(n * top_k):
        J.append(j)

    for i in range(n):
        for j in range(top_k):
            J2.append(j * n + i)
            I.append(i)
            I2.append(j)

    # aggregate the scores of the protected and unprotected groups resp.
    for i in range(n):

        if candidates[i].isProtected == True:

            proCount += 1
            pro_indices.append(i)
            protected_aggregated_scores += learned_scores[i]
        else:
            unproCount += 1
            unpro_indicex.append(i)
            unprotected_aggregated_scores += learned_scores[i]

    learned_scores = np.reshape(learned_scores, (n, 1))
    # uv contains the product of position bias at each position with the merit at
    # each item (flattened and negated).
    uv = learned_scores.dot(position_bias)

    if not individual_fairness:
        # check if there are protected items
        if proCount == 0:
            print(
                "Cannot create a marginal rank probability matrix P for "
                " because there are no protected items in the data set."
            )
            return 0, False
        # check if there are unprotected items
        if unproCount == 0:
            print(
                "Cannot create a marginal rank probability matrix P for "
                " because there are no unprotected items in the data set."
            )
            return 0, False

        initf = np.zeros((n, 1))

        initf[pro_indices] = unprotected_aggregated_scores
        initf[unpro_indicex] = -protected_aggregated_scores

        f1 = initf.dot(position_bias)

        f1 = f1.flatten()
        f1 = np.reshape(f1, (1, n * top_k))

        # we define f and f_value to be used for the fairness constraint
        f = matrix(f1)
        f_value = 0

    # Individual fairness
    else:
        # For individual fairness we compare the ratio of expected exposure at each item
        individual_fairness_constraint = []
        for i in range(len(uv)):
            for j in range(i + 1, len(uv)):
                c = np.zeros(uv.shape)
                c[i] = uv[j]
                c[j] = -uv[i]
                c = c.flatten()
                individual_fairness_constraint.append(c)

        individual_fairness_constraint = matrix(
            np.array(individual_fairness_constraint)
        )
        individual_fairness_value = matrix(
            np.zeros(individual_fairness_constraint.size[0])
        )
        # we define f and f_value to be used for the fairness constraint
        f = individual_fairness_constraint
        f_value = individual_fairness_value

    # Set up objectives

    if outlier_objective:
        # we add the outlier objective, using the outlierness vector calculated based
        # on all available candidate items.
        outlier_vector, outlierness_vector = determine_outlier_vector(
            [c.outlier_feature for c in candidates], alpha=2.5
        )
        outlierness_vector = [
            i if outlier_vector[id] == 1 else 0
            for id, i in enumerate(outlierness_vector)
        ]
        # we are only concerned with removing outliers from the top-k of the rankings
        h = np.array([1] * outlier_window_size + [0] * (top_k - outlier_window_size))
        h = np.reshape(h, (1, top_k))
        o = np.array(outlierness_vector)
        o = np.reshape(o, (n, 1))

        oh = o.dot(h)

        ohuv = uv - oh

        ohuv = ohuv.flatten()
        ohuv = np.negative(ohuv)

    uv = uv.flatten()
    # negate objective function to convert maximization problem to minimization problem
    uv = np.negative(uv)

    # set up constraints x <= 1
    A = spmatrix(1.0, range(n * top_k), range(n * top_k))
    # set up constraints x >= 0
    A1 = spmatrix(-1.0, range(n * top_k), range(n * top_k))
    # set up constraints that sum(rows) <= 1
    M1 = spmatrix(1.0, I, J)
    # set up constraints sum(columns) <= 1
    M2 = spmatrix(1.0, I2, J)

    alpha = 0.99999  # we tolerate an error of 1e-5
    # set up constraints sum(columns)>alpha
    M3 = spmatrix(-1.0, I2, J)

    # values for x<=1
    a = matrix(1.0, (n * top_k, 1))
    # values for x >= 0
    a1 = matrix(0.0, (n * top_k, 1))
    # values for sums columns <= 1
    h1 = matrix(1.0, (n, 1))
    # values for sums rows <= 1
    h2 = matrix(1.0, (top_k, 1))
    # values for sums columns > alpha
    h3 = matrix(-alpha, (top_k, 1))

    # construct objective function
    if outlier_objective:
        c = matrix(ohuv)
    else:
        c = matrix(uv)

    G = sparse([M1, M2, M3, A, A1, f])
    h = matrix([h2, h1, h3, a, a1, f_value])

    print("Start solving LP with DTC.")
    try:
        sol = solvers.lp(c, G, h)
    except Exception:
        print(
            "Cannot create a marginal rank probability matrix P because "
            "linear program can not be solved."
        )
        return 0, False
    print("Finished solving LP with DTC.")
    if sol["x"] is None:
        print(
            "Cannot create a marginal rank probability matrix P because "
            "linear program can not be solved."
        )
        return 0, False
    return np.array(sol["x"]), True
