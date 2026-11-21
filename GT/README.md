# GT: shared game-theory analysis (NT, DD, RS)

Each use-case produces its gain surfaces and its fit of the Self-Division (SD)
operator `f` with its own pipeline (`../NT`, `../DD`, `../RS`). This folder holds
the analysis that is common to all three: the game of paper Section 7 and the
cross-domain validation of paper Tables 2 and 3. Run everything from `GT/`
(conda environment `flower-fl-traffic`).

## Layout

```text
GT/
├── usecases.py                 where each use-case's inputs are, |D_n|, output folders
├── import_usecase.py           copies DD / RS outputs into inputs/ (nothing recomputed)
├── evaluate_game.py            game primitives: grids, capped utility, best-response dynamics
├── sd_outcomes.py              SD decisions vs true NE over the (C1, C2) grid, summary tables
├── search_nash_c.py            protection-level maps and start-order diagnostics
├── cross_validation.py         Tables 2 and 3 (f fitted on one use-case, applied to another)
├── make_cherry_pick_notebook.py  builds notebooks/sd_cherry_pick.ipynb
├── notebooks/sd_cherry_pick.ipynb  interactive cherry-pick table (C slider, LaTeX export)
├── legacy/                     earlier symmetric-C Nash tables (kept, not used for the paper)
├── inputs/<DD|RS>/             DD / RS matrices (<mode>/) and their own f (<Method>_params.json)
└── results/
    ├── <NT|DD|RS>/             c_ranges.json, sd_outcomes/, c_grid/
    └── cross_validation/       real/ (and placeholder/ for testing)
```

NT inputs are read in place: matrices from `../NT/results/<mode>/accuracy/games/average/`,
`f` from `../NT/results/final_80-30/<Method>_params.json`.

## Run

```bash
python import_usecase.py                       # DD and RS inputs
python sd_outcomes.py   --split 80-30          # NT; also --split DD, --split RS
python search_nash_c.py --split 80-30          # NT; also --split DD, --split RS
python cross_validation.py --data real         # Tables 2 and 3
python make_cherry_pick_notebook.py notebooks/sd_cherry_pick.ipynb   # then run it once
```

## Method

* **Utility**: `u_n = max(Delta_n, 0) - C_n (1 - p_n)`; gains are stored signed
  and capped only here.
* **True NE**: best-response dynamics on the measured game from `(0, 0)`.
* **Realized profile**: each player solves its own transformed SD subgame
  (`f` applied per player, then re-monotonized) in both orientations (its own
  weight on either half), keeps the equilibrium with the larger own predicted
  utility (random on a tie), and the two own choices are played on the true
  game. Complete information: the SD subgames use the true `(C1, C2)`.
* **Grid**: 50 x 50 `(C1, C2)` from 0 to a `C_max` at which both players reach
  `p = 1` on both axes (the `p = 1` band is the last ~10% of each axis); one
  `C_max` per mechanism for both sizes in NT and DD, one per size in RS
  (`results/<UC>/c_ranges.json`).
* **Start order**: Player 1 moves first. Only NT Suppression depends on the
  order (two mirror equilibria in many cells); its table rows are given for
  both first movers. NT DP, DD and RS are start-order invariant.
* **Outcomes** per player and cell: hit (realized = true NE); miss with a
  utility above the true NE (`++`), positive (`+`), negative (`-`, harmed) or
  zero (`0`). The summary pools both players (2 x 2500 player-cells).
* **Cross-validation**: `f` of a source use-case applied to a target's `Delta~`
  (same mechanism, the target's `|D_n|` and grid), RMSE against the target's
  `Delta`, the both-echo cell left out, `f` not monotonized; same definitions
  as the fits.

## Outputs for the paper

| Paper | File |
|---|---|
| Table 2 (`tab:boosting`) | `results/cross_validation/real/tab_boosting_rows.tex` |
| Table 3 (`tab:ver`) | `results/cross_validation/real/tab_ver_rows.tex` |
| Section 7 summary table | `results/<UC>/sd_outcomes/tab_outcomes_rows.tex` |
| Section 7 cherry-pick table | `notebooks/sd_cherry_pick.ipynb` (LaTeX export) |
| (C1, C2) maps | `results/<UC>/c_grid/<mode>_<method>_c_grid.png` (and `p1_SD/`, `p2_SD/`) |
| Start-order diagnostics | `results/<UC>/c_grid/start_order_diagnostics.png` |

## Legacy

`legacy/select_symmetric_c.py`, `legacy/generate_nash_tables.py`,
`legacy/nesh_visualisation.py` and the script mode of `evaluate_game.py` produce
the earlier symmetric-C Nash tables. They still run and write to
`../NT/results/nash/` and `../NT/results/final_<split>/`; they are not part of
the current paper pipeline.
