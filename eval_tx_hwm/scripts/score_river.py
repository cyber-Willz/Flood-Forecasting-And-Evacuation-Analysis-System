"""Score v0.4 river_eval output on the same windows as score.py, plus water-surface-elevation (WSE) checks.

usage: python3 score_river.py <results_dir> <tag> [--json out.json] [--quiet]

Metrics per window (0,1,2,3 have the interpolated-extent reference; 4 = Center Point has marks only):
  AUC / mark rank      : identical definitions to score.py (peak depth as the ranking score)
  CSI / POD / FAR      : peak depth > 0.10 m vs reference wet (frac >= 0.5, within 800 m of a mark)
  WSE error at marks   : (a) old-style  elev + peak depth - observed  (the score.py definition; comparable with v0.2/v0.3)
                         (b) surface    bed_near + stage - observed    (the model's own water surface)
  oracle floor         : a single uniform stage chosen to fit that window's marks (the best any uniform-stage model can do)
"""
import csv, json, sys, os
import numpy as np
from scipy.ndimage import minimum_filter, uniform_filter
from scipy.stats import rankdata

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FT = 0.3048
HW = np.load(f'{ROOT}/windows/hwm.npy')  # id,lat,lon,X,Y,wse_m
ORIGIN = {0: (-322396, 784667), 1: (-305396, 780667), 2: (-314396, 785667), 3: (-309396, 784667), 4: (-301740, 774920)}
# gauges (EPSG:5070 X,Y; datum NAVD88 m). Datums are from the earlier addendum; cross-checked against nearest HWM below.
GAUGE = {'Kerrville 08166200': dict(w=1, X=-304642, Y=780357, datum=1601.14 * FT),
         'Center Point 08166250': dict(w=4, X=-299741, Y=772923, datum=1529.54 * FT)}


def auc(s, l):
    l = np.asarray(l, bool)
    if l.all() or (~l).all():
        return float('nan')
    r = rankdata(s)
    n1, n0 = l.sum(), (~l).sum()
    return float((r[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def marks(w):
    X0, Y1 = ORIGIN[w]
    m = (HW[:, 3] >= X0) & (HW[:, 3] < X0 + 4000) & (HW[:, 4] <= Y1) & (HW[:, 4] > Y1 - 4000)
    mk = HW[m]
    return ((Y1 - mk[:, 4]) // 100).astype(int).clip(0, 39), ((mk[:, 3] - X0) // 100).astype(int).clip(0, 39), mk[:, 5]


def load_reach(rdir, w):
    rc = list(csv.DictReader(open(f'{rdir}/reach_w{w}.csv')))
    g = lambda k: np.array([float(x[k]) for x in rc]).reshape(40, 40)
    return dict(elev=g('elev_m'), thal=g('thalweg').astype(bool), bed_near=g('bed_near_m'), hand=g('hand_m'))


def load_rating(rdir, w):
    rt = list(csv.DictReader(open(f'{rdir}/rating_w{w}.csv')))
    return np.array([[float(x[k]) for k in ('stage_m', 'area_m2', 'top_width_m', 'q_m3s')] for x in rt])


def stage_from_q(rt, q):
    return float(np.interp(q, rt[:, 3], rt[:, 0]))


def q_from_stage(rt, h):
    return float(np.interp(h, rt[:, 0], rt[:, 3]))


def score(rdir, tag, wins=(0, 1, 2, 3, 4), verbose=True, depth_col='peak_depth_truth_m'):
    summ = {s['id']: s for s in json.load(open(f'{rdir}/{tag}_summary.json'))}
    out = {}
    for w in wins:
        if w not in summ:
            continue
        R = load_reach(rdir, w)
        d = np.array([float(x[depth_col]) for x in csv.DictReader(open(f'{rdir}/{tag}_w{w}/flood_evac_results_v2_seed1.csv'))]).reshape(40, 40)
        mr, mc, wse_obs = marks(w)
        h = summ[w]['stage_peak_m']
        elev = R['elev']
        e_old = elev[mr, mc] + d[mr, mc] - wse_obs
        wf = f'{rdir}/{tag}_w{w}_wse.csv'
        if os.path.exists(wf):  # v0.5: slope-following water surface
            W = np.array([float(x['wse_m']) for x in csv.DictReader(open(wf))]).reshape(40, 40)
            e_surf = W[mr, mc] - wse_obs
        else:  # v0.4: uniform stage above the local thalweg bed
            e_surf = R['bed_near'][mr, mc] + h - wse_obs
        e_zero = elev[mr, mc] - wse_obs
        h_or = wse_obs - R['bed_near'][mr, mc]
        rec = dict(marks=len(mr), q_peak=summ[w]['q_peak_m3s'], t_peak_h=summ[w]['t_peak_h'], stage_m=h,
                   wse_bias_old=float(e_old.mean()), wse_rmse_old=float(np.sqrt((e_old ** 2).mean())),
                   wse_bias_surface=float(e_surf.mean()), wse_rmse_surface=float(np.sqrt((e_surf ** 2).mean())),
                   wse_rmse_zero=float(np.sqrt((e_zero ** 2).mean())),
                   oracle_stage_m=float(h_or.mean()), wse_rmse_oracle_uniform=float(h_or.std()),
                   mark_hit_gt0p1=float(np.mean(d[mr, mc] > 0.10)))
        if w in (0, 1, 2, 3):
            z = np.load(f'{ROOT}/windows/ref_w{w}.npz')
            frac, dc = z['frac'], z['dcell']
            outl = np.array([x['role'] == 'outlet' for x in csv.DictReader(l for l in open(f'{ROOT}/windows/cells_w{w}.csv') if not l.startswith('#'))]).reshape(40, 40)
            scope = (dc <= 800) & ~outl
            lab = frac >= 0.5
            hand9 = elev - minimum_filter(elev, size=9)
            sc = {'model': d, 'low_elevation': -elev, 'HAND9': -hand9}
            rec['AUC'] = {k: auc(v[scope], lab[scope]) for k, v in sc.items()}
            idx = np.where(scope.ravel())[0]
            pos = {i: j for j, i in enumerate(idx)}
            rk = {}
            for k, v in sc.items():
                r_ = rankdata(v[scope]) / scope.sum()
                got = [r_[pos[a * 40 + b]] for a, b in zip(mr, mc) if (a * 40 + b) in pos]
                rk[k] = float(np.mean(got)) if got else float('nan')
            rec['mark_rank'] = rk
            pred = d > 0.10
            tp = int((pred & lab & scope).sum()); fp = int((pred & ~lab & scope).sum()); fn = int((~pred & lab & scope).sum())
            rec.update(csi=tp / max(tp + fp + fn, 1), pod=tp / max(tp + fn, 1), far=fp / max(tp + fp, 1), ref_wet=int((lab & scope).sum()), pred_wet=int((pred & scope).sum()))
        out[w] = rec
        if verbose:
            a = rec.get('AUC', {})
            print(f"w{w}: Q={rec['q_peak']:.0f} m3/s stage {h:.2f} m (oracle {rec['oracle_stage_m']:.2f}) | WSE err old-style bias {rec['wse_bias_old']:+.2f} rmse {rec['wse_rmse_old']:.2f} | surface bias {rec['wse_bias_surface']:+.2f} rmse {rec['wse_rmse_surface']:.2f} | zero-depth rmse {rec['wse_rmse_zero']:.2f} | uniform-oracle floor {rec['wse_rmse_oracle_uniform']:.2f}"
                  + (f" | AUC model {a['model']:.3f} lowelev {a['low_elevation']:.3f} | CSI {rec['csi']:.2f} POD {rec['pod']:.2f} FAR {rec['far']:.2f} | mark-rank {rec['mark_rank']['model']:.2f}" if a else ''))
    return out


def gauge_wse(rdir, tag):
    """Predicted peak WSE at the two gauges vs the observed 2025 crest."""
    summ = {s['id']: s for s in json.load(open(f'{rdir}/{tag}_summary.json'))}
    res = {}
    for name, g in GAUGE.items():
        if g['w'] not in summ:
            continue
        R = load_reach(rdir, g['w'])
        X0, Y1 = ORIGIN[g['w']]
        r, c = (Y1 - g['Y']) // 100, (g['X'] - X0) // 100
        wf = f"{rdir}/{tag}_w{g['w']}_wse.csv"
        wp = (float(np.array([float(x['wse_m']) for x in csv.DictReader(open(wf))]).reshape(40, 40)[r, c]) if os.path.exists(wf)
              else float(R['bed_near'][r, c] + summ[g['w']]['stage_peak_m']))
        res[name] = dict(cell=(int(r), int(c)), bed_near=float(R['bed_near'][r, c]), ground=float(R['elev'][r, c]), wse_pred=wp)
    return res


if __name__ == '__main__':
    rdir, tag = sys.argv[1], sys.argv[2]
    res = score(rdir, tag)
    print('gauges:', json.dumps(gauge_wse(rdir, tag)))
    if '--json' in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)
