#!/usr/bin/env python3
"""Rasterise FEMA NFHL Special Flood Hazard Areas (zones A/AE/AH/AO/AR/A99/V/VE) onto the flood_shpinn grid.

*** NOT RUN IN THE ENVIRONMENT THAT PRODUCED v0.3: hazards.fema.gov is blocked from that sandbox. ***
Run it where you have internet, then pass the result:  flood_demo ... --reference observed.csv --calibrate

Needs: pip install rasterio shapely requests.   Uses the public NFHL ArcGIS REST service, layer 28 (S_FLD_HAZ_AR).
The grid (bounds, block size, shape) is read from real_data_inputs/meta.json written by prep_real_data.py, so it
lines up cell-for-cell with cells.csv.  For other US sites, generate cells.csv from that site's DEM the same way and
point --meta at its meta.json.
"""
import json, sys, requests, numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import shape

META = sys.argv[1] if len(sys.argv) > 1 else "../real_data_inputs/meta.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "observed.csv"
SFHA = {"A", "AE", "AH", "AO", "AR", "A99", "V", "VE"}
URL = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"

m = json.load(open(META)); b = m["bounds"]; nr, nc = m["grid"]
params = dict(f="geojson", where="1=1", outFields="FLD_ZONE,ZONE_SUBTY,SFHA_TF", returnGeometry="true", outSR=4326,
              geometry=f'{b["left"]},{b["bottom"]},{b["right"]},{b["top"]}', geometryType="esriGeometryEnvelope",
              inSR=4326, spatialRel="esriSpatialRelIntersects")
feats = requests.get(URL, params=params, timeout=120).json().get("features", [])
polys = [(shape(f["geometry"]), 1) for f in feats
         if f["properties"].get("FLD_ZONE") in SFHA or f["properties"].get("SFHA_TF") == "T"]
if not polys: sys.exit("no SFHA polygons returned for this extent (check the service URL / extent)")
SS = 8                                                   # 8x8 sub-cells per grid cell -> fractional coverage
tr = from_bounds(b["left"], b["bottom"], b["right"], b["top"], nc * SS, nr * SS)
ras = rasterize(polys, out_shape=(nr * SS, nc * SS), transform=tr, fill=0, dtype="uint8")
frac = ras.reshape(nr, SS, nc, SS).mean(axis=(1, 3))
with open(OUT, "w") as f:
    f.write("# FEMA NFHL SFHA coverage per cell; flooded = >=50% of the cell inside an SFHA polygon\n")
    f.write("row,col,sfha_frac,flooded\n")
    for r in range(nr):
        for c in range(nc): f.write(f"{r},{c},{frac[r,c]:.3f},{int(frac[r,c] >= 0.5)}\n")
print(f"wrote {OUT}: {int((frac >= 0.5).sum())} of {nr*nc} cells flagged")
