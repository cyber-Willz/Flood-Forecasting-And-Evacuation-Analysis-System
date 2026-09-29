"""Sensitivity of the routed 2025 chain to the parameters that are NOT observed (n, hydrograph shape/rise, reach storage)."""
import subprocess, json, itertools, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import score_river as S
ROOT = S.ROOT
BIN = f'{ROOT}/../target/release/examples/river_eval'
rows = []
for n, shape, tp, ks in itertools.product([0.05, 0.06, 0.08], [3, 6, 12], [1.5, 2.5], [1, 3]):
    tag = f'g_n{n}_a{shape}_tp{tp}_k{ks}'
    rd = f'{ROOT}/results_v04/grid'
    subprocess.run([BIN, '--chain', f'{ROOT}/windows/chain_2025.csv', '--out', rd, '--tag', tag, '--n', str(n), '--shape', str(shape), '--tp', str(tp),
                    '--k-scale', str(ks), '--q-peak', '8892', '--t0', '3', '--tend', '14', '--dt', '0.05'], check=True, capture_output=True)
    res = S.score(rd, tag, verbose=False)
    summ = {s['id']: s for s in json.load(open(f'{rd}/{tag}_summary.json'))}
    all_e = []
    for w in (0, 1, 2, 3, 4):
        R = S.load_reach(rd, w); mr, mc, wo = S.marks(w)
        all_e += list(R['bed_near'][mr, mc] + summ[w]['stage_peak_m'] - wo)
    all_e = np.array(all_e)
    rows.append(dict(n=n, shape=shape, tp=tp, ks=ks, q_kerr=summ[1]['q_peak_m3s'], lag_kerr_h=summ[1]['t_peak_h'] - summ[0]['t_peak_h'],
                     q_cp=summ[4]['q_peak_m3s'], bias=float(all_e.mean()), rmse=float(np.sqrt((all_e ** 2).mean())),
                     rmse_w=[round(res[w]['wse_rmse_surface'], 2) for w in (0, 1, 2, 3, 4)]))
json.dump(rows, open(f'{ROOT}/results_v04/sens_grid.json', 'w'), indent=1)
rows.sort(key=lambda r: r['rmse'])
print('best 12 by pooled WSE RMSE (148 marks):')
for r in rows[:12]:
    print(f"n={r['n']} shape={r['shape']} tp={r['tp']} k_scale={r['ks']} | Q_Kerr {r['q_kerr']:.0f} lag {r['lag_kerr_h']:.2f}h Q_CP {r['q_cp']:.0f} | bias {r['bias']:+.2f} rmse {r['rmse']:.2f} | per-window {r['rmse_w']}")
print('\nunattenuated default (n=.06 shape 3 tp 2.5 k 1):', [ (r['bias'],r['rmse']) for r in rows if (r['n'],r['shape'],r['tp'],r['ks'])==(0.06,3,2.5,1)])
print('worst 3:'); [print(r['n'],r['shape'],r['tp'],r['ks'],round(r['rmse'],2)) for r in rows[-3:]]
