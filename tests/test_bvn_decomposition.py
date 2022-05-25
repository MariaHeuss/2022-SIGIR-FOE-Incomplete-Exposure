from felix.src.code_bvn_decomposition.bvn_decomposition import (
    resampling_condition,
    birkhoff_von_neumann_decomposition,
)
from felix.src.code_bvn_decomposition.random_stochastic_matrix import (
    add_single_column_with_rest_probabilities,
)
import numpy as np


def test_resampling_condition():
    # Each row encodes one item at different positions.
    matrix_true = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    matrix_false = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    scores = [1.0, 2.0, 3.0]
    quality_measure = np.mean
    threshold = 2
    top_k = 2
    assert (
        resampling_condition(matrix_true, scores, quality_measure, threshold, top_k)
        == True
    )
    assert (
        resampling_condition(matrix_false, scores, quality_measure, threshold, top_k)
        == False
    )


def simple_test_bvn_with_test_resampling_condition():
    # Create a list of 20 identical candidates all with same relevance score and same learned score.
    D = np.array(
        [
            [0.2, 0.4, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.4],
            [0.2, 0.2, 0.4, 0.2],
            [0.4, 0.2, 0.2, 0.2],
        ]
    )
    scores = [1, 1, 0, 0]

    def test_quality_measure(scores):
        return sum(scores) % 2

    quality_measure = test_quality_measure
    top_k = 2
    resampling_threshold = 1
    a = birkhoff_von_neumann_decomposition(
        D,
        resampling_threshold=resampling_threshold,
        top_k=top_k,
        quality_measure=quality_measure,
        scores=scores,
        re_search_outliers=4,
    )


def test_add_single_column_with_rest_probabilities():
    mat = [[1 / 4, 1 / 2], [1 / 4, 0], [1 / 4, 1 / 3], [1 / 4, 1 / 6]]
    rest_added = add_single_column_with_rest_probabilities(mat)
    assert np.allclose(
        [
            [1 / 4, 1 / 2, 1 / 4],
            [1 / 4, 0, 3 / 4],
            [1 / 4, 1 / 3, 5 / 12],
            [1 / 4, 1 / 6, 7 / 12],
        ],
        rest_added,
    )


def run_bvn_tests():
    test_resampling_condition()
    simple_test_bvn_with_test_resampling_condition()
    test_add_single_column_with_rest_probabilities()
