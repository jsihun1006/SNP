"""Auxiliary 8x6 counts. Global opponent distances are never zone-gated."""
import numpy as np


def zone_metrics(xy_m, teams, pitch, columns=8, rows=6):
    xy, teams = np.asarray(xy_m), np.asarray(teams)
    if columns <= 0 or rows <= 0:
        raise ValueError("Invalid zone grid")
    n_zones = columns*rows
    inside = np.isfinite(xy).all(axis=1) & (xy >= 0).all(axis=1) & (xy <= pitch).all(axis=1)
    p, t = xy[inside], teams[inside]
    # Exact touchline points belong to the last cell; outside points are not clipped in.
    gx = np.minimum((p[:, 0]/pitch[0]*columns).astype(int), columns-1)
    gy = np.minimum((p[:, 1]/pitch[1]*rows).astype(int), rows-1)
    zone = gy*columns + gx
    home = np.bincount(zone[t == 0], minlength=n_zones)
    away = np.bincount(zone[t == 1], minlength=n_zones)
    total = home+away
    mix = np.divide(2*np.minimum(home, away), total, out=np.full(n_zones, np.nan), where=total > 0)
    d = np.linalg.norm(p[t == 0, None] - p[None, t == 1], axis=-1)
    hz, az = zone[t == 0], zone[t == 1]
    same_zone = hz[:, None] == az[None, :]
    pairs = {}
    for radius in (3, 5, 10):
        hi, _ = np.nonzero(same_zone & (d <= radius))
        pairs[radius] = np.bincount(hz[hi], minlength=n_zones)
    area = pitch[0]*pitch[1]/n_zones
    return [dict(zone=int(z), zone_x=int(z % columns), zone_y=int(z // columns),
                 n_home=int(home[z]), n_away=int(away[z]), n_players=int(total[z]),
                 mixed_index=float(mix[z]), density_per_m2=float(total[z]/area),
                 **{f"within_zone_pairs_{r}m": int(pairs[r][z]) for r in (3, 5, 10)})
            for z in range(n_zones)]
