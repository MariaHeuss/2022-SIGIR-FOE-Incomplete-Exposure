import pytest
from felix.src.algorithms.stochastic_policy import StochasticPolicy
from felix.src.candidate_creator.candidate import get_test_candidate
from felix.src.algorithms.plackett_luce.rankers import (
    to_permutation_matrix,
)
from felix.src.measures.run_metrics import runMetrics
from felix.src.measures.outlier_metrics import *


def test_fairness_metric():
    candidate_a = get_test_candidate(1, 0)
    candidate_b = get_test_candidate(1, 0)
    candidate_c = get_test_candidate(0, 0)
    mat1 = to_permutation_matrix([0, 2], 3)
    coef1 = 0.8
    mat2 = to_permutation_matrix([1, 0], 3)
    coef2 = 0.2
    coefficient_matrix_tuples = [(coef1, mat1), (coef2, mat2)]
    stochastic_policy = StochasticPolicy(
        query_number=1,
        candidates=[candidate_a, candidate_b, candidate_c],
        coefficient_matrix_tuples=coefficient_matrix_tuples,
    )
    metrics = runMetrics(stochastic_policy)

    available_exposure = 1 + 1 / np.log2(3)
    target_exposure = [0.5 * available_exposure, 0.5 * available_exposure, 0]
    actual_exposure = [
        coef1 * 1 + coef2 * 1 / np.log2(3),
        coef2 * 1,
        coef1 * 1 / np.log2(3),
    ]
    EEL = sum(
        (target_exposure[i] - actual_exposure[i]) ** 2
        for i in range(len(target_exposure))
    )
    assert metrics["EEL"] == EEL


def test_ndcg_and_outlier_metrics():
    candidates = [get_test_candidate(0, 0) for _ in range(8)]
    candidates.append(get_test_candidate(1, 0))
    candidates.append(get_test_candidate(1, 1000))
    mat1 = to_permutation_matrix([0, 1, 9, 3, 4, 5, 6, 7], 10)
    mat2 = to_permutation_matrix([8, 1, 2, 3, 4, 5, 6, 7], 10)
    coef1 = 0.8
    coef2 = 0.2

    coefficient_matrix_tuples = [(coef1, mat1), (coef2, mat2)]
    stochastic_policy = StochasticPolicy(
        query_number=1,
        candidates=candidates,
        coefficient_matrix_tuples=coefficient_matrix_tuples,
    )
    metrics = runMetrics(stochastic_policy)

    outlier_count = 0.8
    optimal_dcg = 1 + 1 / np.log2(3)
    expected_dcg = 0.8 * 1 / 2 + 0.2 * 1
    ndcg5 = expected_dcg / optimal_dcg

    assert metrics["outlier_count"] == outlier_count
    assert metrics["ndcg5"] == pytest.approx(ndcg5, 0.0001)


def test_expected_number_of_outliers():
    policy = [
        (
            0.4,
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ],
        ),
        (
            0.6,
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
            ],
        ),
    ]
    scores = [1, 1, 1, 1, 1, 100000]
    assert expected_number_of_outliers(policy, scores, alpha=1.5) == 0.6


def test_probability_displayed_outlier_matrix():
    policy = [
        (0.4, [[1, 0], [0, 1], [0, 0]]),
        (0.35, [[0, 1], [0, 0], [1, 0]]),
        (0.25, [[0, 1], [1, 0], [0, 0]]),
    ]
    scores = [1, 2, 2]
    assert probability_displayed_outlier_matrix(policy, scores, alpha=2.0) == 0


def test_probability_displayed_outlier_matrix_2():
    policy = [
        (
            0.4,
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ],
        ),
        (
            0.6,
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
            ],
        ),
    ]
    scores = [1, 1, 1, 1, 1, 100000]
    assert probability_displayed_outlier_matrix(policy, scores, alpha=1.5) == 0.6


def run_metrics_tests():
    test_fairness_metric()
    test_ndcg_and_outlier_metrics()
    test_expected_number_of_outliers()
    test_probability_displayed_outlier_matrix()
    test_probability_displayed_outlier_matrix_2()
