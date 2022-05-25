import numpy as np


def dcg_at_k(labels_ranked, k):
    if k > 0:
        k = min(labels_ranked.shape[0], k)
    else:
        k = labels_ranked.shape[0]
    denom = 1.0 / np.log2(np.arange(k) + 2.0)
    nom = 2 ** labels_ranked - 1.0
    dcg = np.sum(nom[:k] * denom)
    return dcg


def ndcg_at_k(k, relevance_labels, total_labels=None):

    """
    Calculate NDCG

    @param k: rank of last item we consider for this metric
    @param relevance_labels: true relevance labels in order that we want to determine the ndcg of
    @param total_labels: if not None, defines the labels used for determining the ideal ranking

    return NDCG
    """
    if total_labels is None:
        ideal_labels = np.array(sorted(relevance_labels, reverse=True))
    else:
        ideal_labels = np.array(
            sorted(total_labels, reverse=True)[: len(relevance_labels)]
        )
    relevance_labels = np.array(relevance_labels)
    if dcg_at_k(relevance_labels, k) == 0:
        return 0
    else:
        ndcg = dcg_at_k(relevance_labels, k) / dcg_at_k(ideal_labels, k)
    return ndcg
