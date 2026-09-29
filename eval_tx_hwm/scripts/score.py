import csv, json, sys, numpy as np
from scipy.ndimage import minimum_filter, uniform_filter
from scipy.stats import rankdata
tag=sys.argv[1]; col=sys.argv[2] if len(sys.argv)>2 else 'peak_depth_truth_m'
def auc(s,l):
    l=np.asarray(l,bool); 
    if l.all() or (~l).all(): return float('nan')
    r=rankdata(s); n1=l.sum(); n0=(~l).sum()
    return (r[l].sum()-n1*(n1+1)/2)/(n1*n0)
rows=[]
res={}
for w in range(4):
    z=np.load(f'../windows/ref_w{w}.npz'); N=40
    d=list(csv.DictReader(open(f'../results/{tag}_w{w}/flood_evac_results_v2_seed1.csv')))
    pk=np.array([float(x[col]) for x in d]).reshape(N,N)
    elev=z['elev']; frac=z['frac']; dc=z['dcell']
    cells=list(csv.DictReader(l for l in open(f'../windows/cells_w{w}.csv') if not l.startswith('#')))
    outl=np.array([x['role']=='outlet' for x in cells]).reshape(N,N)
    scope=(dc<=800)&~outl
    lab=(frac>=0.5)
    hand=elev-minimum_filter(elev,size=9)
    relief=elev-uniform_filter(elev,size=9)
    sc={'physics_peak_depth':pk,'low_elevation':-elev,'HAND9':-hand,'local_relief':-relief}
    A={k:auc(v[scope],lab[scope]) for k,v in sc.items()}
    # mark-based (non-circular): percentile rank of score at mark cells within scope
    mr,mc=z['mr'],z['mc']; wse=z['wse']
    pr={}
    for k,v in sc.items():
        vals=v[scope]; rk=rankdata(vals)/len(vals)
        idx=np.where(scope.ravel())[0]; pos={i:j for j,i in enumerate(idx)}
        got=[rk[pos[r*N+c]] for r,c in zip(mr,mc) if (r*N+c) in pos]
        pr[k]=float(np.mean(got)) if got else float('nan')
    # WSE error at marks
    e_phys=[(elev[r,c]+pk[r,c])-s for r,c,s in zip(mr,mc,wse)]
    e_zero=[elev[r,c]-s for r,c,s in zip(mr,mc,wse)]
    hit=float(np.mean([pk[r,c]>0.10 for r,c in zip(mr,mc)]))
    res[w]=dict(cells_scope=int(scope.sum()),ref_wet=int((lab&scope).sum()),prev=float((lab&scope).sum()/scope.sum()),AUC=A,mark_rank=pr,
        marks=len(mr),wse_bias_phys=float(np.mean(e_phys)),wse_rmse_phys=float(np.sqrt(np.mean(np.square(e_phys)))),wse_rmse_zero_depth=float(np.sqrt(np.mean(np.square(e_zero)))),
        peak_depth_max=float(pk.max()),peak_depth_p99=float(np.percentile(pk,99)),mark_hit_gt0p1m=hit,wet_cells_gt0p1=int((pk>0.10).sum()))
    print(w,json.dumps(res[w],indent=None))
json.dump(res,open(f'../results/score_{tag}.json','w'),indent=1)
