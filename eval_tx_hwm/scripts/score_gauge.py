import csv, json, numpy as np
FT=0.3048
def cells(w): return list(csv.DictReader(l for l in open(f'../windows/cells_w{w}.csv') if not l.startswith('#')))
def load(tag,w):
    d=list(csv.DictReader(open(f'../results/{tag}_w{w}/flood_evac_results_v2_seed1.csv')))
    return np.array([float(x['peak_depth_truth_m']) for x in d]).reshape(40,40)
# gauges: (window, X,Y, X0,Y1, 2026 crest WSE m, 2025 nearest-HWM WSE m)
G={'Kerrville 08166200':dict(w=1,X=-304642,Y=780357,X0=-305396,Y1=780667,
     s26=(1601.14+30.06)*FT, s25=499.08, datum=1601.14*FT),
   'Center Point 08166250':dict(w=4,X=-299741,Y=772923,X0=-301740,Y1=774920,
     s26=(1529.54+37.94)*FT, s25=478.38, datum=1529.54*FT)}
out={}
for n,g in G.items():
    r=(g['Y1']-g['Y'])//100; c=(g['X']-g['X0'])//100
    cs=cells(g['w']); cell=cs[r*40+c]; elev=float(cell['elev_m'])
    print(f"\n{n}: window {g['w']} cell ({r},{c}) role='{cell['role']}' block-mean ground {elev:.1f} m")
    t25=('phys','cal') if g['w']==1 else ('e25phys','e25cal')
    dat=g['datum']
    m25=m26=None
    for ev,tagp,tagc,obs in [('2026 (gauge crest, provisional)','e26phys','e26cal',g['s26']),('2025 (nearest HWM)',t25[0],t25[1],g['s25'])]:
        try: pp=load(tagp,g['w'])[r,c]; pc=load(tagc,g['w'])[r,c]
        except Exception as ex: print('  ',ev,'missing',ex); continue
        od=obs-elev
        print(f"  {ev}: observed WSE {obs:.2f} m = {od:+.2f} m above block-mean ground | model depth v0.2 {pp:.3f} m, calibrated {pc:.3f} m | WSE error v0.2 {elev+pp-obs:+.2f}, calibrated {elev+pc-obs:+.2f}, zero-depth {elev-obs:+.2f}")
        st=obs-dat
        print(f"     stage above gauge zero (river depth) {st:.2f} m ({st/FT:.1f} ft) vs model {pp:.2f} m (v0.2) / {pc:.2f} m (calibrated): model reaches {100*pp/st:.1f}% / {100*pc/st:.1f}% of observed stage")
        out[f'{n}|{ev}']=dict(obs=obs,ground=elev,stage=st,v02=float(pp),cal=float(pc))
        if ev.startswith('2026'): m26=(st,pp,pc)
        else: m25=(st,pp,pc)
    if m25 and m26:
        print(f"  2026/2025 ratio: observed stage {m26[0]/m25[0]:.2f} | model v0.2 {m26[1]/m25[1]:.2f} | calibrated {m26[2]/m25[2]:.2f}  (rain ratio 479/265 = {479/265:.2f})")
    # domain-wide context
    for tag in ['e26phys','e26cal']:
        try:
            pk=load(tag,g['w']); print(f"  {tag} window peak depth max {pk.max():.2f} m, p99 {np.percentile(pk,99):.2f} m, cells >0.15 m: {(pk>0.15).sum()}")
        except Exception: pass
json.dump(out,open('../results/score_gauges.json','w'),indent=1)
