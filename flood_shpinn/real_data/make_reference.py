#!/usr/bin/env python3
"""Independent 'known flood-prone area' reference for the Fort Worth DEM crop used by flood_shpinn.

HAND = Height Above Nearest Drainage (Rennó et al. 2008; Nobre et al. 2011): elevation of each DEM cell above
the channel cell it drains to (D8). Low-HAND land is the standard terrain proxy for regulatory floodplains.

THIS IS A TERRAIN-DERIVED PROXY, NOT OBSERVED FLOODING. It does not use the flood model, but it is not FEMA/USGS data.
Swap in observed data with `prep_observed_fema.py` (needs internet access to hazards.fema.gov) and pass --observed.

Outputs (next to this script's cwd):  reference.csv  row,col,hand_frac_lt5,hand_frac_lt10,hand_min_m,ref_flood
"""
import heapq, json, sys, numpy as np, rasterio

DEM = sys.argv[1] if len(sys.argv) > 1 else "dem.tif"
R0, R1, C0, C1, BLK = 0, 126, 181, 367, 6            # identical crop/aggregation to prep_real_data.py
H_FLOOD = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0   # HAND threshold (m) defining 'flood-prone'
CHAN_KM2 = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0  # contributing area that defines a channel

ds = rasterio.open(DEM); z0 = ds.read(1).astype(float)
nr0, nc0 = z0.shape
lat = ds.bounds.top - (np.arange(nr0) + 0.5) * ds.res[1]
dy = ds.res[1] * 111_320.0; dx = ds.res[0] * 111_320.0 * np.cos(np.radians(lat))
cell_area_km2 = (dx * dy)[:, None] * np.ones((1, nc0)) / 1e6

# --- priority-flood pit filling (Barnes et al. 2014), epsilon to keep a gradient across flats
z = z0.copy(); filled = np.zeros_like(z, bool); pq = []
for r in range(nr0):
    for c in range(nc0):
        if r in (0, nr0 - 1) or c in (0, nc0 - 1):
            heapq.heappush(pq, (z[r, c], r, c)); filled[r, c] = True
N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
while pq:
    e, r, c = heapq.heappop(pq)
    for dr, dc in N8:
        rr, cc = r + dr, c + dc
        if 0 <= rr < nr0 and 0 <= cc < nc0 and not filled[rr, cc]:
            filled[rr, cc] = True
            z[rr, cc] = max(z[rr, cc], e + 1e-4)
            heapq.heappush(pq, (z[rr, cc], rr, cc))

# --- D8 flow direction (steepest descent), accumulation (area-weighted), channel mask
recv = -np.ones((nr0, nc0, 2), int)
for r in range(nr0):
    for c in range(nc0):
        best, bd = 0.0, None
        for dr, dc in N8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < nr0 and 0 <= cc < nc0:
                s = (z[r, c] - z[rr, cc]) / np.hypot(dr, dc)
                if s > best: best, bd = s, (rr, cc)
        if bd: recv[r, c] = bd
order = np.dstack(np.unravel_index(np.argsort(-z, axis=None), z.shape))[0]
acc = cell_area_km2.copy()
for r, c in order:
    rr, cc = recv[r, c]
    if rr >= 0: acc[rr, cc] += acc[r, c]
chan = acc >= CHAN_KM2

# --- HAND: walk downstream to first channel cell, memoised in high->low order reversed
hand = np.full((nr0, nc0), np.nan); base = np.full((nr0, nc0), np.nan)
for r, c in order[::-1]:                          # low -> high so receivers are done first
    if chan[r, c]: base[r, c] = z[r, c]
    else:
        rr, cc = recv[r, c]
        base[r, c] = base[rr, cc] if rr >= 0 else z[r, c]
hand = z0 - base
hand = np.where(np.isnan(hand), 0.0, np.maximum(hand, 0.0))

h = hand[R0:R1, C0:C1]
nr, nc = h.shape[0] // BLK, h.shape[1] // BLK
hb = h[:nr * BLK, :nc * BLK].reshape(nr, BLK, nc, BLK)
f5 = (hb < 5).mean(axis=(1, 3)); f10 = (hb < 10).mean(axis=(1, 3)); hmin = hb.min(axis=(1, 3))
fl = (hb < H_FLOOD).mean(axis=(1, 3)) >= 0.5
with open("reference.csv", "w") as f:
    f.write(f"# TERRAIN-DERIVED PROXY (HAND<{H_FLOOD} m, channel>={CHAN_KM2} km2) NOT observed flooding. ref_flood = >=50% of native cells in block\n")
    f.write("row,col,hand_frac_lt5,hand_frac_lt10,hand_min_m,ref_flood\n")
    for r in range(nr):
        for c in range(nc): f.write(f"{r},{c},{f5[r,c]:.3f},{f10[r,c]:.3f},{hmin[r,c]:.2f},{int(fl[r,c])}\n")
print(json.dumps(dict(grid=[nr, nc], ref_flood_cells=int(fl.sum()), frac=float(fl.mean()), H=H_FLOOD, chan_km2=CHAN_KM2,
                      channel_cells_in_crop=int(chan[R0:R1, C0:C1].sum()), hand_p50=float(np.median(h)), hand_p90=float(np.percentile(h, 90)))))
