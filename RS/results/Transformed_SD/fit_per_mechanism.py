"""Table 2 RS rows and Table 3 RS SD row: RS operator fit in the paper's form f = (x + y|D_n|)(1 + a p_own + b p_other) Phi~, per mechanism.
|D_n| = one player's ratings in millions (Samples / 2 / 1e6). Pooled fit over all cells first, as a check against Matan's original Table 2; the per-mechanism rows leave out the both-echo cell."""
import numpy as np, pandas as pd
from scipy.optimize import least_squares, differential_evolution
import os
B = os.path.dirname(os.path.abspath(__file__)) + '/'
real = pd.concat([pd.read_csv(B + d + '/real_' + m + '.csv') for d in ('NF_1', 'NF_5') for m in ('dp', 'sup')])
sim = pd.concat([pd.read_csv(B + d + '/sim_' + m + '.csv') for d in ('NF_1', 'NF_5') for m in ('dp', 'sup')])
df = pd.merge(real, sim, on=['Mechanism', 'Samples', 'P_own', 'P_other'], suffixes=('_real', '_sim'))
print('rows per mechanism/size:', df.groupby(['Mechanism', 'Samples']).size().to_dict())

def f(p, d):
    x, y, a, b = p
    D = d.Samples.values / 2 / 1e6
    return (x + y * D) * (1 + a * d.P_own.values + b * d.P_other.values) * d.Gain_Acc_sim.values

def rmse(v): return float(np.sqrt(np.mean(v ** 2)))

def fit(d):
    res = lambda p: (d.Gain_Acc_real.values - f(p, d)) / np.sqrt(len(d))
    best = None
    rng = np.random.default_rng(42)
    starts = [[1, 0, 0, 0]] + [list(rng.uniform(-5, 5, 4)) for _ in range(40)]
    for s in starts:
        r = least_squares(res, s, method='lm')
        if best is None or r.cost < best.cost: best = r
    de = differential_evolution(lambda p: np.sum(res(p) ** 2), [(-10, 10)] * 4, seed=42, tol=1e-12, maxiter=3000)
    p = best.x if 2 * best.cost <= de.fun else de.x
    return p, rmse(d.Gain_Acc_real.values - f(p, d))

def row(name, d):
    p, r = fit(d)
    cells = []
    for s in sorted(d.Samples.unique()):
        q = d[d.Samples == s]
        cells += [rmse(q.Gain_Acc_real.values - q.Gain_Acc_sim.values), rmse(q.Gain_Acc_real.values - f(p, q))]
    print('%-8s x=%.6g y=%.6g a=%.6g b=%.6g | overall %.6g | size1 SD %.6g Tr %.6g | size2 SD %.6g Tr %.6g' % ((name,) + tuple(p) + (r,) + tuple(cells)))

row('pooled', df)
# Paper rows: the cell where both players echo (nobody trains) is left out of the fit and the RMSEs
df = df[~((df.P_own == 1) & (df.P_other == 1))]
print('rows per mechanism/size without the both-echo cell:', df.groupby(['Mechanism', 'Samples']).size().to_dict())
for m in ('sup', 'dp'):
    row(m, df[df.Mechanism == m])
