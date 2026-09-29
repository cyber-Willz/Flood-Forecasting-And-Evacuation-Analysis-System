import csv, json, numpy as np, rasterio, os
from rasterio.windows import Window
from scipy.spatial import cKDTree
r=rasterio.open('dem/2025_Jul_TX_Flood_DEM.tif')
T=r.transform; x0,y0=T.c,T.f; res=10.0
hw=np.load('../windows/hwm.npy')  # id,lat,lon,X,Y,wse_m
BL=10; N=40  # 100 m cells, 40x40 = 4 km
WIN=BL*N     # native px per window side
# candidate windows on a 1 km lattice; score by mark count
cands=[]
for X0 in range(int(hw[:,3].min())-2000,int(hw[:,3].max()),1000):
    for Y1 in range(int(hw[:,4].min()),int(hw[:,4].max())+2000,1000):
        m=(hw[:,3]>=X0)&(hw[:,3]<X0+4000)&(hw[:,4]<=Y1)&(hw[:,4]>Y1-4000)
        cands.append((m.sum(),X0,Y1))
cands.sort(reverse=True)
chosen=[]
for n,X0,Y1 in cands:
    if n<8: break
    if all(abs(X0-c[1])>=4000 or abs(Y1-c[2])>=4000 for c in chosen): chosen.append((n,X0,Y1))
    if len(chosen)==4: break
print('windows',chosen)
os.makedirs('../windows',exist_ok=True)
# rain: documented approximation of the 4 Jul 2025 storm (hours since 00:00 CDT)
rain=[(0,0),(1.0,0),(1.01,25),(3.0,25),(3.01,55),(6.0,55),(6.01,25),(8.0,25),(8.01,0),(12,0)]
with open('../windows/rain.csv','w') as f:
    f.write('hour,mm_per_h\n')
    for h,m in rain: f.write(f'{h:.2f},{m:.2f}\n')
print('storm total mm',sum(0.5*(rain[i][1]+rain[i+1][1])*(rain[i+1][0]-rain[i][0]) for i in range(len(rain)-1)))
meta=[]
for wi,(n,X0,Y1) in enumerate(chosen):
    col0=int(round((X0-x0)/res)); row0=int(round((y0-Y1)/res))
    a=r.read(1,window=Window(col0,row0,WIN,WIN)).astype(float)
    if a.shape!=(WIN,WIN) or (a<-1e5).mean()>0.01: print('skip',wi,a.shape,(a<-1e5).mean()); continue
    a[a<-1e5]=np.nan
    e=np.nanmean(a.reshape(N,BL,N,BL),axis=(1,3))
    e=np.where(np.isnan(e),np.nanmean(e),e)
    # cell centres
    cx=X0+(np.arange(N)+0.5)*100; cy=Y1-(np.arange(N)+0.5)*100
    # marks in window
    m=(hw[:,3]>=X0)&(hw[:,3]<X0+4000)&(hw[:,4]<=Y1)&(hw[:,4]>Y1-4000)
    mk=hw[m]
    mr=((Y1-mk[:,4])//100).astype(int).clip(0,N-1); mc=((mk[:,3]-X0)//100).astype(int).clip(0,N-1)
    # roles: outlet = 2 lowest boundary cells; shelters = 2 highest boundary cells
    bd=[(i,j) for i in range(N) for j in range(N) if i in(0,N-1) or j in(0,N-1)]
    bd.sort(key=lambda p:e[p]); outl=set(bd[:2]); shel=set(bd[-2:])
    with open(f'../windows/cells_w{wi}.csv','w') as f:
        f.write(f'# REAL 3DEP-derived DEM, 100 m block means, window {wi}; pop/imperv are constant PLACEHOLDERS\nrow,col,elev_m,pop,imperv,role\n')
        for i in range(N):
            for j in range(N):
                role='outlet' if (i,j) in outl else ('shelter' if (i,j) in shel else '')
                f.write(f'{i},{j},{e[i,j]:.3f},10.0,0.20,{role}\n')
    # IDW WSE reference at 10 m
    px=X0+(np.arange(WIN)+0.5)*10; py=Y1-(np.arange(WIN)+0.5)*10
    PX,PY=np.meshgrid(px,py)
    tree=cKDTree(mk[:,3:5]); k=min(4,len(mk))
    d,ix=tree.query(np.c_[PX.ravel(),PY.ravel()],k=k)
    d=np.maximum(d,1.0); w=1/d**2
    wse=(w*mk[ix,5]).sum(1)/w.sum(1)
    near=d[:,0]<=800
    inund=(a.ravel()<wse)&near&~np.isnan(a.ravel())
    inund=inund.reshape(WIN,WIN)
    frac=inund.reshape(N,BL,N,BL).mean(axis=(1,3))
    dcell=cKDTree(mk[:,3:5]).query(np.c_[np.repeat(cx[None,:],N,0).ravel(),np.repeat(cy[:,None],N,1).ravel()])[0].reshape(N,N)
    np.savez(f'../windows/ref_w{wi}.npz',frac=frac,dcell=dcell,elev=e,mr=mr,mc=mc,wse=mk[:,5],ids=mk[:,0])
    meta.append(dict(w=wi,X0=X0,Y1=Y1,marks=int(m.sum()),cells_in_scope=int((dcell<=800).sum()),ref_wet_cells=int(((frac>=0.5)&(dcell<=800)).sum()),elev_min=float(e.min()),elev_max=float(e.max())))
    print(meta[-1])
json.dump(meta,open('../windows/meta.json','w'),indent=1)
