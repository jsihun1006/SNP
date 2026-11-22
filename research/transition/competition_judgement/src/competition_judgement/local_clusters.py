"""Frame-local bipartite proximity components; no identities or temporal links."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MixedComponent:
    radius_m: float
    cluster_id: int
    nodes: tuple[int, ...]  # Ephemeral indices into ONE graph, never source slots.
    family_id_5m: int
    same_node_set_across_radii: bool


@dataclass(frozen=True)
class MixedGraph:
    xy_m: np.ndarray
    team: np.ndarray
    observed: np.ndarray
    distance_m: np.ndarray
    components: dict[float, tuple[MixedComponent, ...]]


def connected_mixed_sets(adjacency):
    """All components with an actual edge; caller supplies cross-team edges only."""
    adjacency = np.asarray(adjacency, dtype=bool)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1] or not np.array_equal(adjacency, adjacency.T):
        raise ValueError("Expected symmetric square adjacency")
    if np.diag(adjacency).any():
        raise ValueError("Mixed components exclude self loops")
    seen = np.zeros(len(adjacency), bool)
    result = []
    for seed in np.flatnonzero(adjacency.any(axis=1)):
        if seen[seed]:
            continue
        stack, nodes = [int(seed)], []
        seen[seed] = True
        while stack:
            node = stack.pop()
            nodes.append(node)
            for neighbor in np.flatnonzero(adjacency[node]):
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(int(neighbor))
        result.append(tuple(sorted(nodes)))
    return result


def build_mixed_graph(xy_m, team):
    """One float64 distance matrix for both exact inclusive 3m/5m graphs.

    Bounds/duplicate flags are handled upstream by the existing QA policy.
    Nonfinite rows are absent nodes; finite same-team duplicates are retained.
    Canonical spatial ordering makes IDs stable under input permutation and H/A
    exchange within this frame. These IDs never imply cross-frame identity.
    """
    xy = np.asarray(xy_m, dtype=np.float64)
    team = np.asarray(team)
    if xy.ndim != 2 or xy.shape[1] != 2 or team.shape != (len(xy),) or not np.isin(team, [0, 1]).all():
        raise ValueError("Invalid player coordinate/team arrays")
    keep = np.isfinite(xy).all(axis=1)
    xy, team = xy[keep].copy(), team[keep].astype(np.int8, copy=True)
    order = np.lexsort((team, xy[:, 1], xy[:, 0]))
    xy, team = xy[order], team[order]
    distance = np.linalg.norm(xy[:, None, :]-xy[None, :, :], axis=-1)
    cross = team[:, None] != team[None, :]
    sets = {r: connected_mixed_sets(cross & (distance <= r)) for r in (3., 5.)}
    # Coordinate signatures ignore team names and contain multiplicity.
    def signature(nodes):
        return tuple(sorted(tuple(xy[i]) for i in nodes))
    sets = {r: sorted(values, key=signature) for r, values in sets.items()}
    parents = {node: index for index, nodes in enumerate(sets[5.]) for node in nodes}
    components = {}
    for radius in (3., 5.):
        values = []
        for i, nodes in enumerate(sets[radius]):
            family = parents[nodes[0]]
            assert all(parents[node] == family for node in nodes)
            same = any(nodes == other for other in sets[5. if radius == 3. else 3.])
            values.append(MixedComponent(radius, i, nodes, family, same))
        components[radius] = tuple(values)
    observed = np.ones(len(xy), dtype=bool)
    for array in (xy, team, observed, distance):
        array.setflags(write=False)
    return MixedGraph(xy, team, observed, distance, components)
