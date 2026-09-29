import csv, numpy as np, rasterio
from pyproj import Transformer
r=rasterio.open('dem/2025_Jul_TX_Flood_DEM.tif')
rows=list(csv.DictReader(open('../flood_shpinn/real_data/observed/2025_Jul_TX_Flood.csv',newline='')))
tr=Transformer.from_crs(4326,5070,always_xy=True)
out=[]
for x in rows:
    try: e=float(x['elev_ft'])*0.3048
    except: continue
    lat,lon=float(x['latitude_dd']),float(x['longitude_dd'])
    X,Y=tr.transform(lon,lat)
    out.append((x['hwm_id'],lat,lon,X,Y,e,x['hwmQualityName'],x['hwmTypeName'],x['countyName'],x['hwm_environment'],x['stillwater']))
print(len(out),'with elev of',len(rows))
arr=np.array([(o[3],o[4]) for o in out])
dem=r.read(1)
vals=[]
for X,Y in arr:
    try:
        rr,cc=r.index(X,Y)
        vals.append(dem[rr,cc] if 0<=rr<dem.shape[0] and 0<=cc<dem.shape[1] else np.nan)
    except Exception: vals.append(np.nan)
vals=np.array(vals); e=np.array([o[5] for o in out])
ok=vals>-1e5
print('in DEM',ok.sum())
d=e[ok]-vals[ok]
print('WSE-DEM m: min %.2f p10 %.2f med %.2f p90 %.2f max %.2f'%(d.min(),*np.percentile(d,[10,50,90]),d.max()))
print('HWM below DEM count',(d<0).sum(), ' < -1m',(d<-1).sum())
import collections
print(collections.Counter(o[7] for o in out), collections.Counter(o[9] for o in out), collections.Counter(o[10] for o in out))
np.save('../windows/hwm.npy',np.array([(o[0],o[1],o[2],o[3],o[4],o[5]) for o in out],dtype=float))
# cluster by X,Y grid of 5km
for i,(o,v) in enumerate(zip(out,vals)):
    pass
import sys
xs=arr[:,0];ys=arr[:,1]
print('X range',xs.min(),xs.max(),'Y',ys.min(),ys.max())
# 4km bins
from collections import Counter
c=Counter((int(x//4000),int(y//4000)) for x,y in arr)
for k,v in c.most_common(12): print(k,v)
