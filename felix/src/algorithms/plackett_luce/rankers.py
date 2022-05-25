from felix.src.algorithms.stochastic_policy import StochasticPolicy
from felix.src.algorithms.plackett_luce.gumbel_sampling import gumbel_sample_rankings
import numpy as np

""" 
Here we define the Plackett-Luce (PL) and Deterministic ranker. To make those rankers
compatible with the evaluation, for the PL ranker we sample a high number n of    
rankings and use them with equal probability 1/n to build a stochastic ranking policy. 
For the deterministic ranker we use the one deterministic ranking that we get by 
sorting the items with respect to the predicted scores and give this ranking a probability
of 1 in the stochastic ranking policy.
"""


def to_permutation_matrix(permutation, num_docs):
    """Converts a permutation into a permutation matrix."""
    top_k = len(permutation)
    P = np.zeros((num_docs, top_k))
    matches = zip(permutation, np.arange(0, top_k, 1))
    # This is a cleverer way of doing
    #
    #     for (u, v) in matches.items():
    #         P[u, v] = 1
    #
    targets = tuple(zip(*matches))
    P[targets] = 1
    return P


class PLRanker(object):
    def __init__(self, candidate_list, num_samplings=1000, uniform_scores=False):
        self.candidate_list = candidate_list
        self.predicted_scores = np.array(
            [candidate.learnedScores for candidate in self.candidate_list]
        )
        self.num_samplings = num_samplings
        # if uniform_scores is true, to get a uniform ranking policy that samples
        # item at each position with the same probability, we set a high epsilon value
        if uniform_scores:
            self.epsilon = 10000
        else:
            self.epsilon = 0

    def get_stochastic_policy(
        self,
        top_k=None,
        query=None,
    ):
        # We need to normalize the scores and take their logarithm to input into the gumbel sampling.
        scores = self.predicted_scores
        min_score = np.min(scores)
        scores = scores - min_score
        # If epsilon is big, it will overshadow the scores to get an uniform policy
        scores = scores + self.epsilon * np.max(scores)
        scores = scores / np.max(scores)
        log_scores = np.log(scores)
        # Use the gumbel_sampling as a fast approximation of PL sampling
        rankings = gumbel_sample_rankings(
            log_scores=log_scores, n_samples=self.num_samplings
        )[0]
        coefficients_matrix_tuples = []
        for ranking in rankings:
            # We are only interested in the first k items that were sampled by the PL sampler
            ranking = ranking[:top_k]
            permutation_mat = to_permutation_matrix(ranking, num_docs=len(scores))
            coefficients_matrix_tuples.append((1 / self.num_samplings, permutation_mat))
        policy = StochasticPolicy(
            query_number=query,
            candidates=self.candidate_list,
            coefficient_matrix_tuples=coefficients_matrix_tuples,
        )
        return policy


class DeterministicRanker:
    def __init__(self, candidate_list, oracle=False, descending=True):
        self.candidate_list = candidate_list
        self.predicted_scores = np.array(
            [candidate.learnedScores for candidate in self.candidate_list]
        )
        self.oracle = oracle
        self.descending = descending

    def get_stochastic_policy(
        self,
        top_k=None,
        query=None,
    ):
        ranking = zip(range(len(self.candidate_list)), self.candidate_list)
        if self.oracle:
            ranking = sorted(
                ranking,
                key=lambda x: x[1].originalQualification,
                reverse=self.descending,
            )
        else:
            ranking = sorted(
                ranking, key=lambda x: x[1].learnedScores, reverse=self.descending
            )
        ranking = [s[0] for s in ranking]
        ranking = ranking[:top_k]
        permutation_mat = to_permutation_matrix(
            ranking, num_docs=len(self.candidate_list)
        )
        coefficients_matrix_tuples = [(1, permutation_mat)]
        policy = StochasticPolicy(
            query_number=query,
            candidates=self.candidate_list,
            coefficient_matrix_tuples=coefficients_matrix_tuples,
        )
        return policy
