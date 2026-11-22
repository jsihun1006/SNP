"""Spatial descriptors, kept distinct from v1 candidate state or ball evidence."""
import numpy as np

from mtah_relational.data_quality import assess_observations
from .quality import POLICY
from .local_clusters import build_mixed_graph

SUMMARY_NAMES = ("n_local_clusters", "max_local_cluster_size", "n_small_mixed_clusters",
                 "n_compact_mixed_clusters", "max_local_cross_density", "max_local_contact_strength",
                 "total_unique_players_in_mixed_clusters", "largest_cluster_radius_m",
                 "max_cluster_diameter_m", "n_cross_edges", "has_strong_compact_relation")


def component_features(graph, component, config):
    nodes = np.asarray(component.nodes)
    xy, team = graph.xy_m[nodes], graph.team[nodes]
    distances = graph.distance_m[np.ix_(nodes, nodes)]
    h, a = np.flatnonzero(team == 0), np.flatnonzero(team == 1)
    cross = distances[np.ix_(h, a)]
    edge = cross <= component.radius_m
    edge_dist = cross[edge]
    n, nh, na, ne = len(nodes), len(h), len(a), int(edge.sum())
    center = xy.mean(axis=0)
    extent = np.ptp(xy, axis=0)
    radius = float(np.linalg.norm(xy-center, axis=1).mean())
    diameter = float(distances.max())
    cross_density = ne/(nh*na)
    balance = 2*min(nh, na)/n
    contact = balance*cross_density*float(np.exp(-edge_dist/config["proximity_decay_m"]).mean())
    degrees = np.r_[edge.sum(axis=1), edge.sum(axis=0)]
    compact = (diameter <= config["compact_max_diameter_over_radius"]*component.radius_m and
               radius <= config["compact_max_mean_radius_over_radius"]*component.radius_m)
    return dict(radius_m=component.radius_m, cluster_id=component.cluster_id,
                family_id_5m=component.family_id_5m,
                same_node_set_across_radii=component.same_node_set_across_radii,
                n_players=n, n_home=nh, n_away=na, n_cross_edges=ne,
                n_unique_contact_players=int((degrees > 0).sum()),
                cluster_centroid_x=float(center[0]), cluster_centroid_y=float(center[1]),
                cluster_radius_m=radius, cluster_bbox_width_m=float(extent[1]),
                cluster_bbox_length_m=float(extent[0]), cluster_bbox_area_m2=float(extent.prod()),
                cluster_max_pairwise_distance_m=diameter, min_cross_distance_m=float(cross.min()),
                mean_cross_edge_distance_m=float(edge_dist.mean()),
                cross_pair_count_3m=int((cross <= 3.).sum()), cross_pair_count_5m=int((cross <= 5.).sum()),
                cross_pair_density=cross_density, home_fraction=nh/n, away_fraction=na/n,
                team_balance=balance, component_size=n, edge_density=2*ne/(n*(n-1)),
                max_degree=int(degrees.max()), mean_degree=2*ne/n,
                contact_strength=contact, compact_mixed_cluster=bool(compact),
                small_mixed_cluster=bool(n <= config["small_max_players"]))


def summarize_components(rows, available, config):
    if not available:
        return {name: np.nan for name in SUMMARY_NAMES}
    n = len(rows)
    # Empty-but-observed graph: count zero is meaningful; extrema undefined.
    largest = sorted(rows, key=lambda row: (-row["n_players"],row["cluster_id"]))[0] if n else None
    return dict(n_local_clusters=n, max_local_cluster_size=max((r["n_players"] for r in rows),default=0),
                n_small_mixed_clusters=sum(r["small_mixed_cluster"] for r in rows),
                n_compact_mixed_clusters=sum(r["compact_mixed_cluster"] for r in rows),
                max_local_cross_density=max((r["cross_pair_density"] for r in rows),default=np.nan),
                max_local_contact_strength=max((r["contact_strength"] for r in rows),default=np.nan),
                total_unique_players_in_mixed_clusters=sum(r["n_players"] for r in rows),
                largest_cluster_radius_m=largest["cluster_radius_m"] if largest else np.nan,
                max_cluster_diameter_m=max((r["cluster_max_pairwise_distance_m"] for r in rows),default=np.nan),
                n_cross_edges=sum(r["n_cross_edges"] for r in rows),
                has_strong_compact_relation=any(r["compact_mixed_cluster"] and
                                               r["contact_strength"] >= config["descriptive_strong_contact_min"]
                                               for r in rows))


def extract_local_frame(frame, pitch, config):
    """Single frame only; input/output has no persistent member IDs or ball field."""
    xy, q = assess_observations(frame, pitch, POLICY)
    graph = build_mixed_graph(xy,frame.team_codes)
    nh, na = int((graph.team == 0).sum()), int((graph.team == 1).sum())
    available = bool(nh and na)
    tier = "strict_valid" if available and q["quality_valid"] else "low" if available else "insufficient"
    summary = dict(frame=frame.frame,period=frame.period,n_home=nh,n_away=na,
                   local_metric_available=available,strict_quality_valid=q["quality_valid"],quality_tier=tier,
                   **{key:q[key] for key in ("missing_slots","partial_missing_slots","invalid_slots",
                      "outside_pitch_slots","excluded_bounds_slots","duplicate_suspect_pairs")})
    all_rows = []
    for radius in (3., 5.):
        rows = [dict(frame=frame.frame,period=frame.period,quality_tier=tier,
                     strict_quality_valid=q["quality_valid"],**component_features(graph,c,config))
                for c in graph.components[radius]]
        values = summarize_components(rows,available,config)
        summary.update({f"{name}_{int(radius)}m":value for name,value in values.items()})
        summary[f"local_cluster_extrema_available_{int(radius)}m"] = bool(rows) and available
        if radius == config["primary_radius_m"]:
            summary.update(values)
        all_rows.extend(rows)
    # Counts at distinct radii are views, not separate events. Five-metre family
    # count is available as the unique maximal spatial grouping, not an episode.
    return summary, all_rows, graph
