"""Build ``notebooks/sd_cherry_pick.ipynb``, the interactive cherry-pick table.

The notebook reads the ``sd_outcomes`` results of NT, DD and RS and shows, for a
chosen C (slider) and configuration, the pairs (0, 0), (C, 0), (0, C), (C, C)
per player: both SD predictions (NE, utility), the true NE and the realized
profile with its utility change; it exports the table as LaTeX. The slider
works without a running kernel. Below it, the summary table of each use-case.

Usage (from ``GT``; then run the notebook once to embed the data)::

    python make_cherry_pick_notebook.py notebooks/sd_cherry_pick.ipynb
"""
import sys
from pathlib import Path

import nbformat as nbf

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("notebooks/sd_cherry_pick.ipynb")
nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(r"""# SD cherry-pick table

Cherry-pick table of CoL-SD Section 7. Choose a configuration and a privacy weight $C$ with the slider; the table shows the four weight pairs $(C_1, C_2) \in \{(0,0), (C,0), (0,C), (C,C)\}$, one row per player.

| Column | Meaning |
|---|---|
| **P1's SD** / **P2's SD** | equilibrium $(p_1, p_2)$ predicted by that player's own transformed Self-Division game, and the utility it predicts for the row's player |
| **True NE** | equilibrium of the measured game and its utility |
| **Realized** | P1's own SD choice + P2's own SD choice, played on the measured game, and its utility |
| **Δu** | realized − true-NE utility (▲ higher, ▼ lower). Red row: realized utility < 0, the player is **harmed** |

Utility: $u_n = \max(\Delta_n, 0) - C_n (1 - p_n)$; P1 moves first. The slider works without a running kernel. To refresh the data, run `python sd_outcomes.py --split 80-30 (and DD, RS) in GT/`, then *Run All* here (kernel: **Python (flower-fl-traffic)**)."""))

cells.append(nbf.v4.new_code_cell(r'''import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import HTML, display

ROOT = Path.cwd() if (Path.cwd() / "results").is_dir() else Path.cwd().parent
DATA_DIRS = {   # use-case -> sd_outcomes folder (python sd_outcomes.py --split <split>, in GT/)
    "NT": ROOT / "results" / "NT" / "sd_outcomes",
    "DD": ROOT / "results" / "DD" / "sd_outcomes",
    "RS": ROOT / "results" / "RS" / "sd_outcomes",
}
OUTCOMES = ("hit", "miss, ++", "miss, +", "miss, −", "miss, 0", "no pure NE")
TOL = 1e-12
SIZE_NAMES = {"30percent": "Size 1 (30%)", "80percent": "Size 2 (80%)",
              "half": "Size 1 (50%)", "full": "Size 2 (100%)",
              "DD_30": "Size 1 (30%)", "DD_80": "Size 2 (80%)",
              "RS_NF1": "Size 1 (NF 1x)", "RS_NF5": "Size 2 (NF 5x)"}
MECH_NAMES = {"Suppression": "Sup", "Noise": "DP"}

CONFIGS, META = {}, {}
for usecase, data_dir in DATA_DIRS.items():
    for path in sorted(data_dir.glob("*_outcomes.npz")):
        d = dict(np.load(path))
        first = int(d["first"])
        name = (f"{usecase} · {SIZE_NAMES[str(d['mode'])]} · {MECH_NAMES[str(d['method'])]}"
                + (" · P2 starts" if first == 2 else ""))
        CONFIGS[name] = d
        META[name] = (usecase, MECH_NAMES[str(d["method"])], SIZE_NAMES[str(d["mode"])].split()[1], first)
CONFIGS = dict(sorted(CONFIGS.items(), key=lambda kv: (list(DATA_DIRS).index(META[kv[0]][0]),
                                                        META[kv[0]][1] != "DP", META[kv[0]][3], kv[0])))
print(f"Loaded {len(CONFIGS)} configurations: " + ", ".join(
    f"{uc} {sum(META[k][0] == uc for k in CONFIGS)}" for uc in DATA_DIRS))


def fmt_p(x):
    return f"{x:.2f}".rstrip("0").rstrip(".") if x % 1 else f"{x:.0f}"


def cherry_rows(d, k):
    """Cherry-pick rows for C = d['C'][k]: 4 weight pairs x 2 players."""
    cs, prot = d["C"], d["protection"]
    prof = lambda i, j: f"({fmt_p(prot[i])}, {fmt_p(prot[j])})"  # noqa: E731
    rows = []
    for k1, k2 in ((0, 0), (k, 0), (0, k), (k, k)):
        c = (k1, k2)
        no_ne = not d["true_converged"][c]
        for n in (1, 2):
            u_star = float(d[f"u{n}_star"][c])
            u_real = float(d[f"u{n}_real"][c])
            du = u_real - u_star
            rows.append({
                "pair": f"({cs[k1]:.3g}, {cs[k2]:.3g})", "player": f"P{n}",
                "sd1_ne": prof(d["sd1_own"][c], d["sd1_partner"][c]),
                "sd1_u": float(d["sd1_u_own" if n == 1 else "sd1_u_partner"][c]),
                "sd2_ne": prof(d["sd2_partner"][c], d["sd2_own"][c]),
                "sd2_u": float(d["sd2_u_partner" if n == 1 else "sd2_u_own"][c]),
                "true_ne": None if no_ne else prof(d["i_star"][c], d["j_star"][c]),
                "true_u": None if no_ne else u_star,
                "real_ne": prof(d["i_real"][c], d["j_real"][c]),
                "real_u": u_real,
                "du": None if no_ne else du,
                "change": "" if no_ne else ("▲" if du > TOL else "▼" if du < -TOL else "="),
                "harmed": u_real < -TOL,
                "outcome": OUTCOMES[int(d[f"outcome{n}"][c])],
            })
    return rows


def table(config, C):
    """The cherry-pick table as a DataFrame (for scripting / export)."""
    d = CONFIGS[config]
    return pd.DataFrame(cherry_rows(d, int(np.argmin(np.abs(d["C"] - C)))))


PAYLOAD = {name: {"C": [float(c) for c in d["C"]],
                  "rows": [cherry_rows(d, k) for k in range(len(d["C"]))]}
           for name, d in CONFIGS.items()}'''))

cells.append(nbf.v4.new_code_cell(r'''uid = uuid.uuid4().hex[:8]
html = r"""
<style>
#cp-UID { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; font-size: 13px;
          color: #1a1a1a; background: #ffffff; padding: 12px; border-radius: 6px; max-width: 1200px; }
#cp-UID .ctl { display: flex; flex-wrap: wrap; gap: 18px; align-items: center; margin-bottom: 10px; }
#cp-UID select, #cp-UID input[type=range] { font-size: 13px; }
#cp-UID input[type=range] { width: 260px; vertical-align: middle; }
#cp-UID .cval { display: inline-block; min-width: 38px; font-weight: 600; }
#cp-UID table { border-collapse: collapse; width: 100%; }
#cp-UID th, #cp-UID td { border: 1px solid #c9c9c9; padding: 4px 7px; text-align: center; white-space: nowrap; }
#cp-UID thead th { background: #eef1f5; }
#cp-UID tr.harmed td { background: #f8d7da; }
#cp-UID tr.down td { background: #fff3cd; }
#cp-UID tr.up td { background: #d1e7dd; }
#cp-UID tr.pairend td { border-bottom: 2px solid #555; }
#cp-UID td.num { font-variant-numeric: tabular-nums; }
#cp-UID .legend span { display: inline-block; padding: 1px 8px; margin-right: 8px; border: 1px solid #c9c9c9; }
#cp-UID pre { background: #f6f6f6; color: #1a1a1a; padding: 8px; overflow-x: auto; font-size: 11.5px; }
</style>
<div id="cp-UID">
  <div class="ctl">
    <label>Configuration <select class="cfg"></select></label>
    <label>C = <span class="cval"></span> <input class="c" type="range" min="0" step="1"></label>
    <label><input class="tex" type="checkbox"> LaTeX</label>
  </div>
  <table>
    <thead>
      <tr><th rowspan="2">(C<sub>1</sub>, C<sub>2</sub>)</th><th rowspan="2">Player</th>
          <th colspan="2">P1's SD</th><th colspan="2">P2's SD</th><th colspan="2">True NE</th>
          <th colspan="2">Realized</th><th colspan="2">Δu</th><th rowspan="2">Outcome</th></tr>
      <tr><th>NE</th><th>u</th><th>NE</th><th>u</th><th>NE</th><th>u</th><th>NE</th><th>u</th>
          <th colspan="2">vs. true NE</th></tr>
    </thead>
    <tbody></tbody>
  </table>
  <p class="legend"><span style="background:#f8d7da">harmed (realized u &lt; 0)</span>
     <span style="background:#fff3cd">lower than true NE</span>
     <span style="background:#d1e7dd">higher than true NE</span> · P1 moves first</p>
  <pre class="texout" style="display:none"></pre>
</div>
<script>
(function () {
  const DATA = PAYLOAD_JSON;
  const root = document.getElementById("cp-UID");
  const cfg = root.querySelector(".cfg"), slider = root.querySelector(".c"),
        cval = root.querySelector(".cval"), tbody = root.querySelector("tbody"),
        tex = root.querySelector(".tex"), texout = root.querySelector(".texout");
  const num = v => v === null ? "—" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(3);
  const texNum = v => v === null ? "--" : "$" + (v >= 0 ? "+" : "-") + Math.abs(v).toFixed(3) + "$";
  Object.keys(DATA).forEach(n => { const o = document.createElement("option"); o.textContent = n; cfg.appendChild(o); });
  function initSlider() {
    const C = DATA[cfg.value].C;
    slider.max = C.length - 1;
    const i = C.findIndex(c => Math.abs(c - 0.10) < 1e-9);
    slider.value = i >= 0 ? i : Math.floor(C.length / 5);
  }
  function render() {
    const d = DATA[cfg.value], k = +slider.value;
    cval.textContent = Number(d.C[k].toPrecision(3)).toString();
    const rows = d.rows[k];
    tbody.innerHTML = rows.map((r, i) => {
      const cls = [r.harmed ? "harmed" : r.change === "▼" ? "down" : r.change === "▲" ? "up" : "",
                   i % 2 ? "pairend" : ""].join(" ");
      const pair = i % 2 ? "" : `<td rowspan="2">${r.pair}</td>`;
      return `<tr class="${cls}">${pair}<td>${r.player}</td>` +
        `<td>${r.sd1_ne}</td><td class="num">${num(r.sd1_u)}</td>` +
        `<td>${r.sd2_ne}</td><td class="num">${num(r.sd2_u)}</td>` +
        `<td>${r.true_ne ?? "no pure NE"}</td><td class="num">${num(r.true_u)}</td>` +
        `<td>${r.real_ne}</td><td class="num">${num(r.real_u)}</td>` +
        `<td class="num">${num(r.du)}</td><td>${r.change}</td><td>${r.outcome}</td></tr>`;
    }).join("");
    if (tex.checked) {
      const lines = [`% ${cfg.value}, C = ${Number(d.C[k].toPrecision(3))}`,
        "\\begin{tabular}{cc|cc|cc|cc|ccc}",
        "$(C_1, C_2)$ & Player & \\multicolumn{2}{c|}{P1's SD} & \\multicolumn{2}{c|}{P2's SD} & " +
        "\\multicolumn{2}{c|}{True NE} & \\multicolumn{3}{c}{Realized} \\\\",
        " & & NE & $u$ & NE & $u$ & NE & $u$ & NE & $u$ & $\\Delta u$ \\\\ \\hline"];
      rows.forEach((r, i) => lines.push([i % 2 ? "" : `$${r.pair}$`, r.player,
        `$${r.sd1_ne}$`, texNum(r.sd1_u), `$${r.sd2_ne}$`, texNum(r.sd2_u),
        r.true_ne ? `$${r.true_ne}$` : "--", texNum(r.true_u),
        `$${r.real_ne}$`, texNum(r.real_u), texNum(r.du)].join(" & ") + (i % 2 ? " \\\\ \\hline" : " \\\\")));
      lines.push("\\end{tabular}");
      texout.textContent = lines.join("\n");
      texout.style.display = "block";
    } else { texout.style.display = "none"; }
  }
  cfg.addEventListener("change", () => { initSlider(); render(); });
  slider.addEventListener("input", render);
  tex.addEventListener("change", render);
  initSlider(); render();
})();
</script>
"""
display(HTML(html.replace("UID", uid).replace("PAYLOAD_JSON", json.dumps(PAYLOAD))))'''))

cells.append(nbf.v4.new_markdown_cell(r"""## Summary over all 2500 $(C_1, C_2)$ cells

Each player in each cell is one observation (2 × 2500 per configuration). Each case uses a 50 × 50 grid from 0 to a C_max chosen so that about 20% of the grid ends at (max, max) in the true game (NT, DD: one C_max per mechanism for both sizes; RS: per size). **Hit**: the realized profile equals the true NE. **Miss**: it does not, split by that player's realized utility: **++** above its true-NE utility, **+** positive but not above it, **−** negative (harmed), **0** zero."""))

cells.append(nbf.v4.new_code_cell(r'''th = "border:1px solid #c9c9c9; padding:5px 12px; background:#eef1f5; text-align:center;"
td = "border:1px solid #c9c9c9; padding:5px 12px; text-align:center; font-variant-numeric:tabular-nums;"
html = ""
for usecase in DATA_DIRS:
    body = ""
    for name, d in CONFIGS.items():
        uc, mech, size, first = META[name]
        if uc != usecase:
            continue
        pooled = np.concatenate([d["outcome1"].ravel(), d["outcome2"].ravel()])
        vals = [100.0 * np.mean(pooled == k) for k in range(5)]
        start = ("–" if mech == "DP" else f"P{first}") if usecase == "NT" else None
        body += (f"<tr><td style='{td}'>{mech}</td><td style='{td}'>{size}</td>"
                 + (f"<td style='{td}'>{start}</td>" if start is not None else "")
                 + "".join(f"<td style='{td}'>{v:.1f}%</td>" for v in vals) + "</tr>")
    first_head = f'<th rowspan="2" style="{th}">First mover</th>' if usecase == "NT" else ""
    html += f"""<h4 style="margin:14px 0 6px">{usecase}</h4>
<table style="border-collapse:collapse; font-size:13px; background:#ffffff; color:#1a1a1a;">
  <thead>
    <tr><th rowspan="2" style="{th}">Mech.</th><th rowspan="2" style="{th}">Size</th>
        {first_head}<th rowspan="2" style="{th}">Hit</th>
        <th colspan="4" style="{th}">Miss</th></tr>
    <tr><th style="{th}">++</th><th style="{th}">+</th><th style="{th}">&minus;</th><th style="{th}">0</th></tr>
  </thead><tbody>{body}</tbody></table>"""
display(HTML(html + "<p style='font-style:italic'>Only NT Suppression depends on the start order and is "
             "shown for both first movers; NT DP, DD and RS are start-order invariant.</p>"))'''))

cells.append(nbf.v4.new_markdown_cell(r"""## Scripting

`table(config, C)` returns the same table as a DataFrame, e.g. `table("NT · Size 1 (30%) · Sup", 0.15)`."""))
cells.append(nbf.v4.new_code_cell(r'''table(next(iter(CONFIGS)), 0.10)'''))

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "flower-fl-traffic", "display_name": "Python (flower-fl-traffic)",
                                "language": "python"}
nb["metadata"]["language_info"] = {"name": "python"}
if __name__ == "__main__":
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(f"wrote {out}")
