"""v0.5 parameter study for the 1D router (river_eval --router d1d).

usage: python3 scripts/tune_v05.py            # writes results_v05/tune_grid.json and prints the summary

Objective terms (all independent of the 'AUC' circularity):
  E_wse  pooled water-surface RMSE (m) at the 2025 marks, model surface vs observed HWM (score_river 'surface')
  E_q    |ln(Q_peak(w1)/3794)|   reported Kerrville peak "more than 134,000 cfs" (a lower bound)
  E_lag  |lag(Hunt input peak -> w1 peak) - 1.6| h   reported Hunt->Kerrville crest lag
J = E_wse + 2*E_q + E_lag.   The weights are a judgement call; the full table is written so any other weighting can be applied.
Split-sample: parameters are re-chosen on the WSE marks of w0,w2,w3 and scored on w1,w4, and the reverse.
"""
import itertools, json, os, subprocess, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import score_river as sr
EXE = os.path.abspath(os.path.join(ROOT, '..', 'target', 'release', 'examples', 'river_eval'))
Q_KERR, LAG_OBS, Q_IN_T = 3794.0, 1.6, 4.5
OUT = f'{ROOT}/results_v05/grid'

def run(tag, **kw):
    d = f'{OUT}/{tag}'
    args = [EXE, '--chain', f'{ROOT}/windows/chain_2025.csv', '--out', d, '--tag', 'g']
    for k, v in kw.items():
        args += [f'--{k.replace("_", "-")}', str(v)]
    subprocess.run(args, check=True, capture_output=True)
    res = sr.score(d, 'g', verbose=False)
    summ = {s['id']: s for s in json.load(open(f'{d}/g_summary.json'))}
    return res, summ

def pooled(res, wins):
    m = np.array([res[w]['marks'] for w in wins]); r = np.array([res[w]['wse_rmse_surface'] for w in wins])
    return float(np.sqrt((m * r ** 2).sum() / m.sum()))

def evaluate(tag, **kw):
    res, summ = run(tag, **kw)
    wins = (0, 1, 2, 3, 4)
    e_w = pooled(res, wins)
    q1, t1 = summ[1]['q_peak_m3s'], summ[1]['t_peak_h']
    e_q, e_l = abs(math.log(q1 / Q_KERR)), abs((t1 - Q_IN_T) - LAG_OBS)
    return dict(params=kw, wse_pooled=e_w, wse_by_win={w: res[w]['wse_rmse_surface'] for w in wins}, wse_bias={w: res[w]['wse_bias_surface'] for w in wins},
                marks={w: res[w]['marks'] for w in wins}, q_w1=q1, attn=q1 / 8892.0, lag_h=t1 - Q_IN_T, E_q=e_q, E_lag=e_l, J=e_w + 2 * e_q + e_l)

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    grid = dict(n_ch=[0.035, 0.05, 0.07], n_fp=[0.07, 0.10, 0.15, 0.20], hbf=[2.0, 3.0, 4.5, 6.0], carve=[0.0, 1.5, 3.0])
    rows = []
    for i, vals in enumerate(itertools.product(*grid.values())):
        kw = dict(zip(grid.keys(), vals))
        rows.append(evaluate(f'r{i}', **kw))
    json.dump(rows, open(f'{ROOT}/results_v05/tune_grid.json', 'w'), indent=1)
    best = min(rows, key=lambda r: r['J'])
    print('cases', len(rows)); print('best J', json.dumps(best))
    # split-sample on WSE only (Q/lag terms do not use marks)
    def wsel(r, wins):
        m = np.array([r['marks'][str(w)] if str(w) in r['marks'] else r['marks'][w] for w in wins]); e = np.array([r['wse_by_win'][w] for w in wins])
        return float(np.sqrt((m * e ** 2).sum() / m.sum()))
    for fit, test in (((0, 2, 3), (1, 4)), ((1, 4), (0, 2, 3))):
        b = min(rows, key=lambda r: wsel(r, fit) + 2 * r['E_q'] + r['E_lag'])
        print(f'fit {fit}: {b["params"]}  fit-RMSE {wsel(b, fit):.2f}  held-out {test} RMSE {wsel(b, test):.2f}')
