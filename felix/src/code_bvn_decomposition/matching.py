"""

Modified version of the implementation of the Hopcroft--Karp algorithm in the
networkx package (https://networkx.org) version 2.6.2. See `/license/LICENSE_Networkx`
for full license.

Adjustments to the code were made to account for graphs that are biparate
if you account for certain nodes with higher multiplicity. Nodes with
multiplicity m can be thought of as m copies of the same node. Adding nodes
with multiplicity instead of copies of the same node increases efficiency of
the algorithm.

"""


import collections
import itertools
from networkx.algorithms.bipartite import sets as bipartite_sets

INFINITY = float("inf")


def hopcroft_karp_matching_multiplicity(
    G,
    top_nodes=None,
    right_nodes_multiplicity=None,
):
    """Returns the maximum cardinality matching of the bipartite graph `G`.

    A matching is a set of edges that do not share any nodes. In this variant
    of the Hopcroft Karb matching algorithm, we allow certain nodes to have a
    higher multiplicity, which allows them to match with more than one node.
    Maximum cardinality matching is a matching with the most edges possible. It
    is not always unique. Finding a matching in a bipartite graph can be
    treated as a networkx flow problem.

    Parameters
    ----------
    G : NetworkX graph

      Undirected bipartite graph

    top_nodes : container of nodes

      Container with all nodes in one bipartite node set. If not supplied
      it will be computed. But if more than one solution exists an exception
      will be raised.

    right_nodes_multiplicity: dict

      Dictionary with right-node-id as keys and multiplicity counter as values.

    Returns
    -------
    matches : dictionary

      The matching is returned as a dictionary, `matches`, such that
      ``matches[v] == w`` if node `v` is matched to node `w`. Unmatched
      nodes do not occur as a key in `matches`.

    Raises
    ------
    AmbiguousSolution
      Raised if the input bipartite graph is disconnected and no container
      with all nodes in one bipartite set is provided. When determining
      the nodes in each bipartite set more than one valid solution is
      possible if the input graph is disconnected.

    Notes
    -----
    This function is implemented with the `Hopcroft--Karp matching algorithm
    <https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm>`_ for
    bipartite graphs.

    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------
    maximum_matching
    hopcroft_karp_matching
    eppstein_matching

    References
    ----------
    .. [1] John E. Hopcroft and Richard M. Karp. "An n^{5 / 2} Algorithm for
       Maximum Matchings in Bipartite Graphs" In: **SIAM Journal of Computing**
       2.4 (1973), pp. 225--231. <https://doi.org/10.1137/0202019>.

    """
    # First we define some auxiliary search functions.
    #
    # If you are a human reading these auxiliary search functions, the "global"
    # variables `leftmatches`, `rightmatches`, `distances`, etc. are defined
    # below the functions, so that they are initialized close to the initial
    # invocation of the search functions.
    def breadth_first_search():
        for v in left:
            if leftmatches[v] is None:
                distances[v] = 0
                queue.append(v)
            else:
                distances[v] = INFINITY
        distances[None] = INFINITY
        while queue:
            v = queue.popleft()
            if distances[v] < distances[None]:
                for u in G[v]:
                    if len(rightmatches[u]) < right_nodes_multiplicity.get(u, 0):
                        distances[None] = distances[v] + 1
                    else:
                        for rightmatch in rightmatches[u]:
                            if distances[rightmatch] is INFINITY:
                                distances[rightmatch] = distances[v] + 1
                                queue.append(rightmatch)
        return distances[None] is not INFINITY

    def depth_first_search(v):
        if v is not None:
            if type(v) == int:
                v = [v]
            for v_ in v:
                for u in G[v_]:
                    # If u not matched with full multiplicity_count we have a new path.
                    if not rightmatches[u] or len(
                        rightmatches[u]
                    ) < right_nodes_multiplicity.get(u, 0):
                        # We need to remove the previous connection of v_
                        if leftmatches[v_]:
                            rightmatches[leftmatches[v_]].remove(v_)
                        leftmatches[v_] = u
                        rightmatches[u].append(v_)
                        return True
                    # If u matched with full multiplicity_count we need to look further for a path
                    else:
                        for r in rightmatches[u]:
                            if distances[r] == distances[v_] + 1:
                                if len(rightmatches[u]) < right_nodes_multiplicity.get(
                                    u, 0
                                ):
                                    rightmatches[u].append(v_)
                                    leftmatches[v_] = u
                                    return True
                                elif depth_first_search(r):
                                    if leftmatches[v_]:
                                        rightmatches[leftmatches[v_]].remove(v_)
                                    rightmatches[u].append(v_)
                                    leftmatches[v_] = u
                                    return True
                distances[v_] = INFINITY
            return False
        return True

    # Initialize the "global" variables that maintain state during the search.
    left, right = bipartite_sets(G, top_nodes)
    leftmatches = {v: None for v in left}
    rightmatches = {v: [] for v in right}
    distances = {}
    queue = collections.deque()

    if right_nodes_multiplicity is None:
        right_nodes_multiplicity = {r: 1 for r in right}

    while breadth_first_search():
        for v in left:
            if leftmatches[v] is None:
                if depth_first_search(v):
                    pass

    # Strip the entries matched to `None`.
    leftmatches = {k: v for k, v in leftmatches.items() if v is not None}
    rightmatches = {k: v for k, v in rightmatches.items() if v is not None}

    # At this point, the left matches and the right matches are inverses of one
    # another. In other words,
    #
    #     leftmatches == {v, k for k, v in rightmatches.items()}
    #
    # Finally, we combine both the left matches and right matches.
    return dict(itertools.chain(leftmatches.items(), rightmatches.items()))
