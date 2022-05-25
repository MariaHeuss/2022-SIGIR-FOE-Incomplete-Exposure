"""

Big parts of the code refer to https://github.com/jfinkels/birkhoff/blob/master/birkhoff.py,
with Copyright 2015 by Jeffrey Finkelstein under the GNU Affero General Public License.
See `/license/LICENSE_birkhoff' for full license.

Adjustments to the code were made to account for non squared matrices that are stochastic in
the columns and sum to values smaller or equal than 1 in the rows. Moreover, the iterative
aggregation and re-decomposition that allows for reduction of outliers in the final policy
is included in this file. For reference see #TODO cite arxiv paper

Provides a function for computing the Birkhoff--von Neumann decomposition of
a certain matrices into a convex combination of permutation matrices.

"""


import itertools
from networkx import from_numpy_matrix
from felix.src.code_bvn_decomposition.matching import (
    hopcroft_karp_matching_multiplicity,
)
from felix.src.code_bvn_decomposition.random_stochastic_matrix import *
from functools import reduce
import numpy as np

#: Any number smaller than this will be rounded down to 0 when computing the
#: difference between NumPy arrays of floats.
TOLERANCE = np.finfo(np.float).eps * 10.0


def to_pattern_matrix(D):
    """Returns the Boolean matrix in the same shape as `D` with ones exactly
    where there are nonzero entries in `D`.
    `D` must be a NumPy array.
    """
    result = np.zeros_like(D)
    # This is a cleverer way of doing
    #
    #     for (u, v) in zip(*(D.nonzero())):
    #         result[u, v] = 1
    #
    result[D.nonzero()] = 1
    return result


def zeros(m, n):
    """Convenience function for ``numpy.zeros((m, n))``."""
    return np.zeros((m, n))


def hstack(left, right):
    """Convenience function for ``numpy.hstack((left, right))``."""
    return np.hstack((left, right))


def vstack(top, bottom):
    """Convenience function for ``numpy.vstack((top, bottom))``."""
    return np.vstack((top, bottom))


def four_blocks(topleft, topright, bottomleft, bottomright):
    """Convenience function that creates a block matrix with the specified
    blocks.
    Each argument must be a NumPy matrix. The two top matrices must have the
    same number of rows, as must the two bottom matrices. The two left matrices
    must have the same number of columns, as must the two right matrices.
    """
    return vstack(hstack(topleft, topright), hstack(bottomleft, bottomright))


def to_bipartite_matrix(A):
    """Returns the adjacency matrix of a bipartite graph whose biadjacency
    matrix is `A`.
    `A` must be a NumPy array.
    If `A` has **m** rows and **n** columns, then the returned matrix has **m +
    n** rows and columns.
    """
    m, n = A.shape
    return four_blocks(zeros(m, m), A, A.T, zeros(n, n))


def to_permutation_matrix(matches, m, n):
    """Converts a permutation into a permutation matrix.
    `matches` is a dictionary whose keys are vertices and whose values are
    partners. For each vertex ``u`` and ``v``, entry (``u``, ``v``) in the
    returned matrix will be a ``1`` if and only if ``matches[u] == v``.
    Pre-condition: `matches` must be a permutation on an initial subset of the
    natural numbers.
    Returns a permutation matrix as a square NumPy array.
    """
    P = np.zeros((m, n))
    # This is a cleverer way of doing
    #
    #     for (u, v) in matches.items():
    #         P[u, v] = 1
    #
    targets = tuple(zip(*(matches.items())))
    P[targets] = 1
    return P


def resampling_condition(matrix, scores, quality_measure, threshold=1, top_k=10):
    scores = np.matmul(scores, matrix)[:top_k]
    return quality_measure(scores) >= threshold


def birkhoff_von_neumann_decomposition(
    D,
    re_search_outliers=0,
    quality_measure=None,
    scores=None,
    top_k=None,
    resampling_threshold=1,
):
    m, n = D.shape
    right_nodes_multiplicity = None
    if m < n:
        raise ValueError(
            "The input matrix should have at least as many rows as columns. Else input transposed matrix."
        )
    elif m > n:
        # For a non-squared matrix add a column with multiplicity m - n.
        # We will use an adjusted version of the Hopcroft-Karb algorithm
        # with multiplicity to find a matching that fits this multiplicity
        # perfectly.
        right_nodes_multiplicity = {column: 1 for column in range(m, m + n)}
        D = add_single_column_with_rest_probabilities(D)
        right_nodes_multiplicity[m + n] = m - n

        indices = list(itertools.product(range(m), range(n + 1)))
    else:
        indices = list(itertools.product(range(m), range(n)))

    # These two lists store the coefficients and matrices that we iteratively
    # split off with the BvN decomposition
    coefficients = []
    permutations = []

    # Create a copy of D so that we don't modify it directly. Cast the
    # entries of the matrix to floating point numbers, regardless of
    # whether they were integers.
    S = D.astype("float")
    outlier_permutations = []
    outlier_coefficients = []
    while not np.allclose(S, 0):
        # Create an undirected graph whose adjacency matrix contains a 1
        # exactly where the matrix S has a nonzero entry.
        W = to_pattern_matrix(S)

        # Construct the bipartite graph whose left and right vertices both
        # represent the vertex set of the pattern graph (whose adjacency matrix
        # is ``W``).
        X = to_bipartite_matrix(W)

        # Convert the matrix of a bipartite graph into a NetworkX graph object.
        G = from_numpy_matrix(X)

        # Compute a perfect matching for this graph. The dictionary `M` has one
        # entry for each matched vertex (in both the left and the right vertex
        # sets), and the corresponding value is its partner.
        #
        # The bipartite maximum matching algorithm requires specifying
        # the left set of nodes in the bipartite graph. By construction,
        # the left set of nodes is {0, ..., n - 1} and the right set is
        # {n, ..., 2n - 1}; see `to_bipartite_matrix()`.
        left_nodes = range(m)

        M = hopcroft_karp_matching_multiplicity(
            G,
            left_nodes,
            right_nodes_multiplicity=right_nodes_multiplicity,
        )

        # However, since we have both a left vertex set and a right vertex set,
        # each representing the original vertex set of the pattern graph
        # (``W``), we need to convert any vertex greater than ``n`` to its
        # original vertex number. To do this,
        #
        #   - ignore any keys greater than ``n``, since they are already
        #     covered by earlier key/value pairs,
        #   - ensure that all values are less than ``n``.
        #

        M = {u: v - m for u, v in M.items() if u < m}

        # Convert that perfect matching to a permutation matrix.
        P = to_permutation_matrix(M, m, m)
        P = P[:, : n + 1]

        # Get the smallest entry of S corresponding to the 1 entries in the
        # permutation matrix.
        q = min(S[i, j] for (i, j) in indices if P[i, j] == 1)

        # If the re_search_outliers argument is passed as True we group the
        # permutation matrices into two sets, dependent on whether they contain
        # an outlier.
        if re_search_outliers > 0 and resampling_condition(
            matrix=P,
            scores=scores,
            quality_measure=quality_measure,
            top_k=top_k,
            threshold=resampling_threshold,
        ):
            outlier_permutations.append(P[:, :n])
            outlier_coefficients.append(q)
        else:
            coefficients.append(q)
            permutations.append(P[:, :n])
        # Subtract P scaled by q. After this subtraction, S has a zero entry
        # where the value q used to live.
        S -= q * P
        # PRECISION ISSUE: There seems to be a problem with floating point
        # precision here, so we need to round down to 0 any entry that is very
        # small.
        S[np.abs(S) < TOLERANCE] = 0.0

    # If re_search_outliers is passed as True we aggregate the rankings with
    # outliers to a new matrix that we input into another iteration of the BvN
    # decomposition algorithm.
    if re_search_outliers > 0 and outlier_permutations:
        # We aggregate the outlier matrices.
        outliers = [q * P for q, P in zip(outlier_coefficients, outlier_permutations)]
        outliers_sum = reduce(lambda x, y: x + y, outliers)
        # Since our algorithm does not actually sample randomly but in order
        # we permute the rows of the matrix to get a different decomposition
        # in the next iteration
        permut = (random_permutation_matrix(m), random_permutation_matrix(n))
        scores_perm = permut[0].dot(scores)
        outliers_sum = permut[0].dot(outliers_sum[:, :n])
        # call the bvn algorithm on this matrix
        rest_decomposition = birkhoff_von_neumann_decomposition(
            outliers_sum,
            quality_measure=quality_measure,
            scores=scores_perm,
            re_search_outliers=re_search_outliers - 1,
            top_k=top_k,
        )
        # Permute all matrices in the decomposition back
        rest_decomposition = [
            (c, permut[0].T.dot(mat)) for c, mat in list(rest_decomposition)
        ]
        decomp = list(zip(coefficients, permutations)) + rest_decomposition
        # We are only interested in the first n columns of the permutation matrices
        decomp = [(c, mat[:, :n]) for c, mat in list(decomp)]
        return decomp
    decomp = list(zip(coefficients, permutations)) + list(
        zip(outlier_coefficients, outlier_permutations)
    )
    decomp = [(c, mat[:, :n]) for c, mat in list(decomp)]
    return decomp
