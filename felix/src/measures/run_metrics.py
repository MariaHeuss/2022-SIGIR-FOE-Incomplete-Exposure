import felix.src.measures.ranking_metrics as ranking_metrics
import felix.src.measures.fairness_metrics as fairness_metrics
import numpy as np


def get_relevance_labels(candidates):
    relevance_labels = []
    for candidate in candidates:
        relevance_labels.append(candidate.originalQualification)
    return relevance_labels


def weighted_average_ndcg(k, stochastic_policy, relevance_labels):
    total_ndcg = 0
    for coefficient, permutation in stochastic_policy:
        ranking_labels = np.matmul(permutation.T, relevance_labels)
        ndcg = ranking_metrics.ndcg_at_k(
            k, ranking_labels, total_labels=relevance_labels
        )
        total_ndcg += coefficient * ndcg
    return total_ndcg


def get_protected_attribute_matrix(candidates):
    group = np.zeros((2, len(candidates)))
    for i, candidate in enumerate(candidates):
        if candidate.isProtected:
            group[1, i] = 1
        else:
            group[0, i] = 1
    return group


def runMetrics(stochastic_policy, individual_fairness=True, outlier_top_k=10):

    coefficients_matrix_tuples = stochastic_policy.coefficients_matrix_tuples

    # If we use individual fairness as constraint we get an identity matrix as group matrix.
    if individual_fairness:
        group_matrix = np.identity(len(stochastic_policy.candidates))

    # In case we want to use group fairness we use the attributes that are given in the data
    else:
        group_matrix = get_protected_attribute_matrix(
            candidates=stochastic_policy.candidates
        )

    exposure_per_group = fairness_metrics.total_exposure_per_group(
        coefficients_matrix_tuples, group_matrix
    )

    relevance_labels = np.array(get_relevance_labels(stochastic_policy.candidates))

    merit = fairness_metrics.get_metrit_per_group(relevance_labels, group_matrix)

    eel = fairness_metrics.EEL(
        merit_per_group=merit, exposure_per_group=exposure_per_group
    )

    ndcg1 = weighted_average_ndcg(1, coefficients_matrix_tuples, relevance_labels)
    ndcg5 = weighted_average_ndcg(5, coefficients_matrix_tuples, relevance_labels)
    ndcg10 = weighted_average_ndcg(10, coefficients_matrix_tuples, relevance_labels)

    outlierness_omit = stochastic_policy.expected_outlierness(
        k=outlier_top_k, outlier_threshold=2.5
    )

    outlier_count = stochastic_policy.expected_number_of_outliers(
        k=outlier_top_k, outlier_threshold=2.5
    )

    prob_outlier = stochastic_policy.probability_of_displaying_an_outlier_ranking(
        k=outlier_top_k, outlier_threshold=2.5
    )

    results = {
        "EEL": eel,
        "ndcg1": ndcg1,
        "ndcg5": ndcg5,
        "ndcg10": ndcg10,
        "outlierness_omit": outlierness_omit,
        "outlier_count": outlier_count,
        "outlier_probability": prob_outlier,
    }

    return results
