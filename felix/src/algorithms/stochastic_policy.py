import numpy as np
from functools import partial

from felix.src.code_bvn_decomposition.bvn_decomposition import (
    birkhoff_von_neumann_decomposition,
)
from felix.src.measures.outlier_metrics import (
    count_outliers_zscore,
    measure_outlierness,
)


def is_doubly_stochastic_matrix(matrix):
    k = len(matrix[0])
    n = len(matrix)
    doubly_stochastic = True
    if not np.allclose(
        [sum([row[j] for row in matrix]) for j in range(k)], np.ones(k), atol=1e-04
    ):
        print("Columns do not sum to 1")
        doubly_stochastic = False
    if not np.all(
        np.less_equal([sum([row[j] for row in matrix.T]) for j in range(n)], np.ones(n))
    ):
        print("Sum of rows are smaller than 1")
        doubly_stochastic = False
    if np.any(matrix < -1e-5):
        print("Values smaller than 0")
        doubly_stochastic = False
    if np.any(matrix > 1):
        print("Values bigger than 1")
        doubly_stochastic = False
    return doubly_stochastic


def is_permutation_matrix(matrix):
    k = len(matrix[0])
    permutation = is_doubly_stochastic_matrix(matrix)
    if not (
        np.allclose(np.min(matrix, axis=0), np.zeros(k), atol=1e-05)
        and np.allclose(np.max(matrix, axis=0), np.ones(k), atol=1e-05)
    ):
        print("Not a permutation matrix")
        permutation = False
    return permutation


class StochasticPolicy(object):
    def __init__(self, query_number, candidates, coefficient_matrix_tuples: list):
        self.query_number = query_number
        self.coefficients_matrix_tuples = coefficient_matrix_tuples
        self.candidates = candidates

    @classmethod
    def from_mrp(
        cls,
        marginal_rank_probability_matrix,
        candidates,
        decomposition_method="vanilla_BvN",
        query_number=None,
        atol=1e-5,
        top_k=10,
        number_of_resamples=10,
        outlier_threshold=2.5,
    ):
        if decomposition_method == "vanilla_BvN":
            decomposition = birkhoff_von_neumann_decomposition(
                marginal_rank_probability_matrix
            )
        elif decomposition_method == "outlier_resample":
            scores = [candidate.outlier_feature for candidate in candidates]
            decomposition = birkhoff_von_neumann_decomposition(
                marginal_rank_probability_matrix,
                re_search_outliers=number_of_resamples,
                scores=scores,
                quality_measure=partial(count_outliers_zscore, alpha=outlier_threshold),
                top_k=top_k,
            )
        else:
            raise NotImplementedError

        # Remove extremely matrices with extremely small coefficients to not have rounding errors
        # blow up decomposition size.
        decomposition = [(c, m) for c, m in decomposition if c > atol]

        return cls(
            query_number=query_number,
            candidates=candidates,
            coefficient_matrix_tuples=decomposition,
        )

    def compute_mrp(self):
        mrp = np.zeros(self.coefficients_matrix_tuples[0][1].shape)
        for coeff, pmatrix in self.coefficients_matrix_tuples:
            mrp += coeff * pmatrix
        return mrp

    def is_valid_policy(self):
        is_valid = True
        if not np.isclose(sum(c for c, _ in self.coefficients_matrix_tuples), 1):
            print(
                "Coefficients do not sum to 1: ",
                sum(c for c, _ in self.coefficients_matrix_tuples),
            )
            is_valid = False

        for matrix in [m for _, m in self.coefficients_matrix_tuples]:
            if not is_permutation_matrix(matrix):
                print("Contains matrices that are not permutation matrices")
                is_valid = False
        return is_valid

    def expected_outlierness(self, k=None, outlier_threshold=2.5):
        if k is None:
            k = len(self.coefficients_matrix_tuples[0][1][0])
        expected_outlireness = 0
        scores = [candidate.outlier_feature for candidate in self.candidates]
        for coefficient, matrix in self.coefficients_matrix_tuples:
            top_k_scores = np.matmul(scores, matrix)[:k]
            expected_outlireness += coefficient * measure_outlierness(
                top_k_scores, alpha=outlier_threshold
            )
        return expected_outlireness

    def expected_number_of_outliers(self, k=None, outlier_threshold=2.5):
        if k is None:
            k = len(self.coefficients_matrix_tuples[0][1][0])
        expected_outlireness = 0
        scores = [candidate.outlier_feature for candidate in self.candidates]
        for coefficient, matrix in self.coefficients_matrix_tuples:
            top_k_scores = np.matmul(scores, matrix)[:k]
            expected_outlireness += coefficient * count_outliers_zscore(
                top_k_scores, alpha=outlier_threshold
            )
        return expected_outlireness

    def probability_of_displaying_an_outlier_ranking(
        self, k=None, outlier_threshold=2.5
    ):
        if k is None:
            k = len(self.coefficients_matrix_tuples[0][1][0])
        prob = 0
        scores = [candidate.outlier_feature for candidate in self.candidates]
        for coefficient, matrix in self.coefficients_matrix_tuples:
            top_k_scores = np.matmul(scores, matrix)[:k]
            if count_outliers_zscore(top_k_scores, alpha=outlier_threshold) >= 1:
                prob += coefficient
        return prob
