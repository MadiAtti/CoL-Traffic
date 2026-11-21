"""Table 2 DD rows, Table 3 DD SD row and the RS<->DD transfer cells, from the 2026-10 DD rerun.
Operator in the paper's form f = (x + y|D_n|)(1 + a p_own + b p_other) Delta~, fitted per mechanism over both sizes,
leaving out the cell where both players echo. |D_n| = one player's compounds in thousands (Samples / 2 / 1e3).
Inputs are the per-player isotonic grids (seed mean, monotonized per player): the true surface is the mean of the two
players and the SD surface the mean of the four half-split players, each in its own (p_own, p_other) perspective.
Writes data_30%/ and data_80%/ (real_*, sim_*, Difference/*_difference_transformed.csv) beside this script."""
import os
import numpy as np, pandas as pd
from scipy.optimize import least_squares, differential_evolution

B = os.path.dirname(os.path.abspath(__file__)) + '/'
SIZES = {'Small_data_30%': ('data_30%', 70854), 'Large_data_80%': ('data_80%', 188945)}  # Samples: both players
RS = B + '../../../RS/results/Transformed_SD/'
RS_PARAMS = {'sup': [0.758332, -0.0442311, 0.608741, 0.066067], 'dp': [1.1639, -0.0838379, 0.0754697, 0.0173887]}


def own_grid(df, client):
    g = df[df.Client == client].pivot_table(index='P1_Param', columns='P2_Param', values='Gain_Acc_Smoothed')
    g = g.sort_index().sort_index(axis=1)
    return (g.values if client == 0 else g.values.T), g.index.values   # client 1 owns P2


def surface(folder, mech, files):
    grids = []
    for f in files:
        df = pd.read_csv(B + '../' + folder + '/CSVs_Isotonic/' + f)
        for c in sorted(df.Client.unique()):
            g, levels = own_grid(df, c)
            grids.append(g)
    return np.mean(grids, axis=0), levels, len(grids)


rows = []
for folder, (out, samples) in SIZES.items():
    for mech in ('sup', 'dp'):
        real, lv, nr = surface(folder, mech, ['%s_real.csv' % mech])
        sim, lv2, ns = surface(folder, mech, ['%s_sim_p1.csv' % mech, '%s_sim_p2.csv' % mech])
        assert np.allclose(lv, lv2) and (nr, ns) == (2, 4)
        po, pt = np.meshgrid(lv, lv, indexing='ij')
        for i in range(len(lv)):
            for j in range(len(lv)):
                rows.append(dict(Mechanism=mech, Samples=samples, P_own=po[i, j], P_other=pt[i, j],
                                 Gain_Acc_real=real[i, j], Gain_Acc_sim=sim[i, j], out=out))
df = pd.DataFrame(rows)
corner = (df.P_own == 1) & (df.P_other == 1)


def f(p, d, scale):
    x, y, a, b = p
    D = d.Samples.values / 2 / scale
    return (x + y * D) * (1 + a * d.P_own.values + b * d.P_other.values) * d.Gain_Acc_sim.values


def rmse(v):
    return float(np.sqrt(np.mean(v ** 2)))


def fit(d):
    res = lambda p: (d.Gain_Acc_real.values - f(p, d, 1e3)) / np.sqrt(len(d))
    best = None
    rng = np.random.default_rng(42)
    for s in [[1, 0, 0, 0]] + [list(rng.uniform(-5, 5, 4)) for _ in range(40)]:
        r = least_squares(res, s, method='lm')
        if best is None or r.cost < best.cost:
            best = r
    de = differential_evolution(lambda p: np.sum(res(p) ** 2), [(-10, 10)] * 4, seed=42, tol=1e-12, maxiter=3000)
    return best.x if 2 * best.cost <= de.fun else de.x


rs = pd.concat([pd.merge(pd.read_csv(RS + d + '/real_' + m + '.csv'), pd.read_csv(RS + d + '/sim_' + m + '.csv'),
                         on=['Mechanism', 'Samples', 'P_own', 'P_other'], suffixes=('_real', '_sim'))
                for d in ('NF_1', 'NF_5') for m in ('sup', 'dp')]).rename(
    columns={'Gain_Acc_real': 'Gain_Acc_real', 'Gain_Acc_sim': 'Gain_Acc_sim'})
rs = rs[~((rs.P_own == 1) & (rs.P_other == 1))]

params = {}
for m in ('sup', 'dp'):
    d = df[(df.Mechanism == m) & ~corner]
    p = params[m] = fit(d)
    cells = []
    for s in sorted(d.Samples.unique()):
        q = d[d.Samples == s]
        cells += [rmse(q.Gain_Acc_real.values - q.Gain_Acc_sim.values), rmse(q.Gain_Acc_real.values - f(p, q, 1e3))]
    print('Table 2 DD %-3s x=%.3g y=%.3g alpha=%.3g beta=%.3g | size1 SD %.3g Trans %.3g | size2 SD %.3g Trans %.3g'
          % ((m,) + tuple(p) + tuple(cells)))
    # Table 3: RS operator applied to DD, DD operator applied to RS (same mechanism, each in its own size unit)
    to_dd = [rmse(q.Gain_Acc_real.values - f(RS_PARAMS[m], q, 1e3)) for s, q in d.groupby('Samples')]
    r = rs[rs.Mechanism == m]
    to_rs = [rmse(q.Gain_Acc_real.values - f(p, q, 1e6)) for s, q in r.groupby('Samples')]
    print('Table 3 %-3s RS->DD %.3g %.3g | DD->RS %.3g %.3g' % ((m,) + tuple(to_dd) + tuple(to_rs)))

for out, g in df.groupby('out'):
    os.makedirs(B + out + '/Difference', exist_ok=True)
    for m in ('sup', 'dp'):
        q = g[g.Mechanism == m].copy()
        x, y, a, b = params[m]
        q['Scenario'] = 'real'
        q[['Scenario', 'Mechanism', 'Samples', 'P_own', 'P_other', 'Gain_Acc_real']].rename(
            columns={'Gain_Acc_real': 'Gain_Acc'}).to_csv(B + out + '/real_%s.csv' % m, index=False)
        q['Scenario'] = 'simulation'
        q[['Scenario', 'Mechanism', 'Samples', 'P_own', 'P_other', 'Gain_Acc_sim']].rename(
            columns={'Gain_Acc_sim': 'Gain_Acc'}).to_csv(B + out + '/sim_%s.csv' % m, index=False)
        q['x'], q['y'], q['alpha'], q['beta'] = x, y, a, b
        q['Transformed_f'] = f(params[m], q, 1e3)
        q['Difference'] = q.Gain_Acc_real - q.Transformed_f
        q[['Mechanism', 'Samples', 'P_own', 'P_other', 'Gain_Acc_real', 'Gain_Acc_sim', 'x', 'y', 'alpha', 'beta',
           'Transformed_f', 'Difference']].to_csv(B + out + '/Difference/%s_difference_transformed.csv' % m, index=False)
