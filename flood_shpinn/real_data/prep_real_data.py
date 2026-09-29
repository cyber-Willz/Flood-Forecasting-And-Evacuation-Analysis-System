#!/usr/bin/env python3
"""Fetch a REAL DEM live from GitHub and convert it to flood_shpinn inputs.

Source (real, fetched at run time):
  https://raw.githubusercontent.com/mdbartos/pysheds/master/data/dem.tif
  3-arcsec DEM, EPSG:4326, Fort Worth / Trinity River, Texas (lon -97.485..-97.179, lat 32.522..32.822).

REAL   : elevation (aggregated by block mean).
DERIVED: outlets  = lowest cells on the downstream (east) boundary column;
         imperv   = proxy from local slope (flat land -> more built-up);
         pop      = proxy: flat, low-lying terrace cells above the river bed (settlement-like), NOT census data;
         shelters = 3 flat high-ground cells on the west/south flank near populated area.
These proxies exist only because the DEM carries no land-cover/population layers.
"""
import sys, urllib.request, numpy as np, rasterio, json, hashlib
URL = "https://raw.githubusercontent.com/mdbartos/pysheds/master/data/dem.tif"
R0, R1, C0, C1, BLK = 0, 126, 181, 367, 6      # crop (rows, cols) of native grid + block size
urllib.request.urlretrieve(URL, "dem.tif")
sha = hashlib.sha256(open("dem.tif", "rb").read()).hexdigest()
ds = rasterio.open("dem.tif"); a = ds.read(1).astype(float)[R0:R1, C0:C1]
nr, nc = a.shape[0] // BLK, a.shape[1] // BLK
z = a[:nr*BLK, :nc*BLK].reshape(nr, BLK, nc, BLK).mean(axis=(1, 3))
lat0 = ds.bounds.top - (R0 + R1) / 2 * ds.res[1]
dy = ds.res[1] * 111_320.0; dx = ds.res[0] * 111_320.0 * np.cos(np.radians(lat0))
cell_m = float(np.sqrt(dx * dy) * BLK)
gy, gx = np.gradient(z)
slope = np.hypot(gx, gy) / cell_m                       # m/m
sl_n = np.clip(slope / np.percentile(slope, 90), 0, 1)
imperv = np.clip(0.75 - 0.5 * sl_n, 0.2, 0.75)          # proxy
zmin = z.min(); rel = z - zmin
terrace = (rel > 6) & (rel < 45) & (sl_n < 0.55)        # flat land a few m..45 m above the river bed
pop = np.where(terrace, 150.0 * (1 - sl_n), 5.0) * (0.6 + 0.4 * np.cos(np.arange(nr)[:, None] * 0.7) ** 2)
pop = np.round(pop, 1)
role = np.full((nr, nc), "", dtype=object)
east = z[:, -1]; out_rows = np.argsort(east)[:2]; 
for r in out_rows: role[r, nc - 1] = "outlet"
# shelters: high, flat, populated-adjacent cells on west column band
cand = [(z[r, c], r, c) for r in range(nr) for c in range(0, 4) if sl_n[r, c] < 0.5 and role[r, c] == ""]
cand.sort(reverse=True); picked = []
for _, r, c in cand:
    if all(abs(r - pr) + abs(c - pc) > 6 for pr, pc in picked): picked.append((r, c))
    if len(picked) == 3: break
for r, c in picked: role[r, c] = "shelter"
with open("cells.csv", "w") as f:
    f.write("# REAL elevation (pysheds dem.tif, Fort Worth TX); imperv/pop/shelter = terrain-derived proxies\n")
    f.write("row,col,elev_m,pop,imperv,role\n")
    for r in range(nr):
        for c in range(nc): f.write(f"{r},{c},{z[r,c]:.3f},{pop[r,c]:.1f},{imperv[r,c]:.2f},{role[r,c]}\n")
# same stress-test hyetograph as the addon's default synthetic demo (NOT an observed storm)
hours = np.arange(0, 6.01, 0.25); peak, t0, tp, t1 = 130.0, 0.5, 2.5, 5.0
q = np.where((hours <= t0) | (hours >= t1), 0, np.where(hours <= tp, peak * (hours - t0) / (tp - t0), peak * (t1 - hours) / (t1 - tp)))
with open("rain.csv", "w") as f:
    f.write("hour,mm_per_h\n"); [f.write(f"{h:.2f},{v:.2f}\n") for h, v in zip(hours, q)]
meta = dict(source=URL, sha256=sha, native_shape=list(ds.shape), crop=[R0, R1, C0, C1], block=BLK, grid=[nr, nc], cell_m=round(cell_m, 1),
            elev_min=float(z.min()), elev_max=float(z.max()), pop_total=float(pop.sum()), outlets=[(int(r), nc - 1) for r in out_rows], shelters=[(int(r), int(c)) for r, c in picked],
            bounds=dict(left=ds.bounds.left + C0 * ds.res[0], right=ds.bounds.left + C1 * ds.res[0], top=ds.bounds.top - R0 * ds.res[1], bottom=ds.bounds.top - R1 * ds.res[1]))
json.dump(meta, open("meta.json", "w"), indent=1); print(json.dumps(meta, indent=1))
