import numpy as np


def count_outliers_zscore(list, alpha=2.5):
    std = np.std(list)
    mean = np.mean(list)
    count = 0
    for i in list:
        if abs(i - mean) > alpha * std:
            count += 1
    return count


def determine_outlier_vector(list, alpha=2.5):
    std = np.std(list)
    mean = np.mean(list)
    outlierness_vector = [
        (i - mean) / std if abs(i - mean) > alpha * std else 0 for i in list
    ]
    outlier_vector = [1 if abs(i - mean) > alpha * std else 0 for i in list]
    return outlier_vector, outlierness_vector


def measure_outlierness(list, alpha=2.5):
    std = np.std(list)
    if std == 0:
        return 0
    mean = np.mean(list)
    list = [(abs((s - mean)) / std) for s in list]
    outlier_vector, outlierness_vector = determine_outlier_vector(list, alpha=alpha)
    outlierness_vector = [
        o if outlier_vector[i] != 0 else 0 for i, o in enumerate(outlierness_vector)
    ]
    return 0 if not outlierness_vector else np.mean(outlierness_vector)


def probability_displayed_outlier_matrix(policy, observable_scores, alpha=2.5):
    """Takes as an input a list of pairs of permutation matrix and probability coefficient
    and a list of observable scores and returns the probability that a matrix with an
    outlier is being displayed."""
    prob = 0
    for coefficient, matrix in policy:
        # get a list with all items in the top k ranking
        top_k = [i for i in range(len(matrix)) if sum(matrix[i]) == 1]
        # get score of all items in the list
        scores = [observable_scores[i] for i in top_k]
        # calculate whether the ranking has an outlier
        if count_outliers_zscore(scores, alpha=alpha) > 0:
            # add the probability that this ranking is displayed to the total
            # probability of getting an outlier list.
            prob += coefficient
    return prob


def expected_number_of_outliers(policy, observable_scores, alpha=2.5):
    """Takes as an input a list of pairs of permutation matrix and probability scalar and returns
    the expected number of outliers in a sampled ranking."""
    E = 0
    for coefficient, matrix in policy:
        # get a list with all items in the top k ranking
        top_k = [i for i in range(len(matrix)) if sum(matrix[i]) == 1]
        # get score of all items in the list
        scores = [observable_scores[i] for i in top_k]
        # add to weighted sum if scores contain an outlier
        E += coefficient * count_outliers_zscore(scores, alpha=alpha)
    return E
