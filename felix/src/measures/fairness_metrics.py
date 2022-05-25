import numpy as np


def get_exposure_per_rank(num_positions):
    v = np.arange(1, (num_positions + 1), 1)
    v = 1 / np.log2(1 + v)
    return v


def get_exposure_per_item(permutation_matrix):
    exposure_per_rank = get_exposure_per_rank(len(permutation_matrix[0]))
    return np.matmul(permutation_matrix, exposure_per_rank)


def get_exposure_per_group(permutation_matrix, group_per_candidate):
    exposure_per_item = get_exposure_per_item(permutation_matrix)
    return np.matmul(group_per_candidate, exposure_per_item)


def total_exposure_per_group(coefficient_matrix_tuples, group_per_candidate):
    exposure_per_group = np.zeros((len(group_per_candidate),))
    for coefficient, permutation in coefficient_matrix_tuples:
        exposure_per_group += coefficient * get_exposure_per_group(
            permutation, group_per_candidate
        )
    return exposure_per_group


def get_metrit_per_group(relevance_labels, group_per_candidate):
    return np.matmul(group_per_candidate, relevance_labels)


def EEL(merit_per_group, exposure_per_group):
    s_exp = exposure_per_group
    avail_exp = sum(exposure_per_group)
    total_merit = sum(merit_per_group)
    tgt_exp = merit_per_group * (avail_exp / total_merit)
    delta = s_exp - tgt_exp

    ee_loss = np.dot(delta, delta)
    return ee_loss
