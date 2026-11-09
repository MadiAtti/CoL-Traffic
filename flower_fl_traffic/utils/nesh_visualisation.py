"""
Render the Nash-equilibrium comparison as a readable PNG table.

The standalone entry point delegates to ``generate_nash_tables`` so tables
are written to the appropriate ``results/final_<split>/`` directories.

TABLE LAYOUT (one PNG per mode)
-------------------------------
Two blocks, DP (Noise) on top and SUP (Suppression) below. Inside a block every
row is one weight pair (w_first_player, w_second_player). The columns are

    p11-p12        Nash cell of the first simulated (self-division) game
    p21-p22        Nash cell of the second simulated game
    p1-p2 (real)   Nash cell of the real game
    Sim matches    which simulated game(s) reproduce the real Nash cell

A cell shows the grid index (row, column) of the equilibrium:
(0, 0) = nobody defends, the last index = maximum privacy.
If the result depends on who moves first, both are shown as "(a, b) / (c, d)"
= first player starts / second player starts. A trailing "*" marks a dynamics
that did not converge to a Nash equilibrium (cycle / iteration limit).
"""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

METHOD_LABEL = {"Noise": "DP", "Suppression": "SUP"}

# (key in the JSON "games" block, column header)
GAME_COLUMNS = [("sim_1", "p11 – p12"),
                ("sim_2", "p21 – p22"),
                ("real", "p1 – p2  (real)")]

SIM_KEYS = ["sim_1", "sim_2"]
SIM_NAMES = {"sim_1": "P1", "sim_2": "P2"}     # used in the summary column

ORDERS = ["first_player_starts", "second_player_starts"]

# Which weight pairs become rows:
#   "unordered" -> only w_first <= w_second (3 weights -> 6 rows)
#   "all"       -> every ordered pair        (3 weights -> 9 rows)
WEIGHT_PAIRS = "all"

# colours
C_HEADER = "#1f2937"
C_MATCH = "#c7efd4"     # simulated equilibrium equals the real one
C_MISS = "#f8d0d0"      # simulated equilibrium differs from the real one
C_PARTIAL = "#fdeab3"   # summary: only one of the two sim games matches
C_REAL = "#e5e9f0"
C_METHOD = {"DP": "#dbe7fb", "SUP": "#fde3cf"}
C_STRIPE = ("#ffffff", "#f6f7f9")
C_GRID = "#b8bec8"


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def select_rows(entries):
    """Return the (weight key, entry) pairs that become table rows, sorted by weights."""
    rows = []
    for key, e in entries.items():
        w1, w2 = e["weights"]["first_player"], e["weights"]["second_player"]
        if WEIGHT_PAIRS == "unordered" and w1 > w2:
            continue
        rows.append((w1, w2, e))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def fmt_cell(game):
    """(p1, p2) védelemi értékeket ír ki az indexek helyett."""
    prot = [tuple(round(x, 2) for x in game[o]["protection"]) for o in ORDERS]
    text = f"({prot[0][0]}, {prot[0][1]})"
    if prot[0] != prot[1]:
        text += f" / ({prot[1][0]}, {prot[1][1]})"
    if not all(game[o]["converged"] for o in ORDERS):
        text += " *"
    return text


def sim_matches(entry, sim_key):
    """A simulated game matches if its cell equals the real one for BOTH move orders."""
    return all(entry["comparison"][o][sim_key]["matches_real"] for o in ORDERS)


def summary_label(entry):
    """Text + colour of the 'Sim matches real' column."""
    hits = [k for k in SIM_KEYS if sim_matches(entry, k)]
    if len(hits) == len(SIM_KEYS):
        return "P1 & P2", C_MATCH
    if hits:
        return SIM_NAMES[hits[0]] + " only", C_PARTIAL
    return "none", C_MISS

def evaluate_sim_choice_in_real(entry, order):
    """
    Kiszámolja a p1 és p2 konkrét Real utility értékét abban az esetben,
    ha a Sim 1 (p11) és Sim 2 (p22) által választott védelmi szintekkel játszanak.
    Csak az előjeleket jeleníti meg: +, - vagy 0.
    """
    real_game = entry["games"]["real"][order]
    sim1_game = entry["games"]["sim_1"][order]
    sim2_game = entry["games"]["sim_2"][order]

    # Választott szimulált indexek
    chosen_i = sim1_game["cell"][0]  # p11 választása
    chosen_j = sim2_game["cell"][1]  # p22 választása

    # Real játék eredeti egyensúlyi indexei
    real_i, real_j = real_game["cell"]

    # 6. oszlop szövege: választott védelmi értékek (p11, p22)
    p11_val = round(sim1_game["protection"][0], 2)
    p22_val = round(sim2_game["protection"][1], 2)
    choice_txt = f"({p11_val}, {p22_val})"

    # Súlyok és szivárgások (Leakage = 1 - protection)
    w1 = entry["weights"]["first_player"]
    w2 = entry["weights"]["second_player"]
    leak1 = 1.0 - sim1_game["protection"][0]
    leak2 = 1.0 - sim2_game["protection"][1]

    # Ha a választott cella pontosan a Real Nash cellája, a tárolt utility-t használjuk
    if (chosen_i, chosen_j) == (real_i, real_j):
        u1_real = real_game["utility"][0]
        u2_real = real_game["utility"][1]
    else:
        # Ha eltér, a kiválasztott cella utility-je a Real játékban
        u1_real = real_game["gain"][0] - (w1 * leak1)
        u2_real = real_game["gain"][1] - (w2 * leak2)

    # Előjelek meghatározása kerekítés után
    u1_round = round(u1_real, 2)
    u2_round = round(u2_real, 2)

    sign1 = "+" if u1_round > 0 else ("-" if u1_round < 0 else "0")
    sign2 = "+" if u2_round > 0 else ("-" if u2_round < 0 else "0")

    # 7. oszlop szövege: csak az előjelek
    util_txt = f"({sign1}, {sign2})"

    return choice_txt, util_txt

# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def draw_table(data, mode, out_path):
    meta = data["metadata"]
    methods = [m for m in meta["methods"] if m in METHOD_LABEL]

    # Pontosan 7 oszlop szélessége (inch):
    # 0: Method, 1: Weights, 2: Sim1, 3: Sim2, 4: Real, 5: Sim Choice (p11, p22), 6: Real Shift
    col_w = [0.9, 1.7, 2.3, 2.3, 2.3, 2.3, 2.5]
    
    headers = [
        "Method", 
        "Weights\n(w1, w2)", 
        "p11 – p12", 
        "p21 – p22", 
        "p1 – p2  (real)", 
        "Sim Choice\n(p11, p22)", 
        "Simulated Choice\nin Real Game"
    ]
    
    row_h, head_h = 0.52, 0.8
    title_h, foot_h = 0.95, 1.55

    blocks = {m: select_rows(data["results"][mode][m]) for m in methods}
    n_rows = sum(len(b) for b in blocks.values())
    width = sum(col_w)
    height = title_h + head_h + n_rows * row_h + foot_h

    fig = plt.figure(figsize=(width, height), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    x_edges = [0]
    for w in col_w:
        x_edges.append(x_edges[-1] + w)

    def box(col, y, h, color, text="", bold=False, color_text="#111827", size=13, span=1):
        x0, x1 = x_edges[col], x_edges[col + span]
        ax.add_patch(Rectangle((x0, y), x1 - x0, h, facecolor=color,
                               edgecolor=C_GRID, linewidth=0.8))
        if text:
            ax.text((x0 + x1) / 2, y + h / 2, text, ha="center", va="center",
                    fontsize=size, fontweight="bold" if bold else "normal", color=color_text)

    # Title
    ax.text(width / 2, 0.38, f"Nash equilibria: {meta.get('table_label', mode)} data size",
            ha="center", va="center", fontsize=19, fontweight="bold")
    ax.text(width / 2, 0.72,
            "grid index (row, column) of the equilibrium;  (0, 0) = no defence for both",
            ha="center", va="center", fontsize=11.5, color="#4b5563")

    # Header
    y = title_h
    for c, h in enumerate(headers):
        box(c, y, head_h, C_HEADER, h, bold=True, color_text="white", size=12.5)
    y += head_h

    # Body
    for m in methods:
        label = METHOD_LABEL[m]
        rows = blocks[m]
        y_block = y
        for r, (w1, w2, e) in enumerate(rows):
            stripe = C_STRIPE[r % 2]
            
            # 1. oszlop: Weights
            box(1, y, row_h, stripe, f"({w1:g}, {w2:g})", size=13)

            # 2-4. oszlopok: Sim 1 (p11-p12), Sim 2 (p21-p22), Real (p1-p2)
            for c, (key, _) in enumerate(GAME_COLUMNS, start=2):
                txt = fmt_cell(e["games"][key])
                if key == "real":
                    color = C_REAL
                else:
                    color = C_MATCH if sim_matches(e, key) else C_MISS
                box(c, y, row_h, color, txt, bold=(key == "real"), size=12)

            # Szimulált értékek és előjelek lekérése
            choice_txt, diff_txt = evaluate_sim_choice_in_real(e, "first_player_starts")

            # 5. oszlop: Sim Choice (p11, p22) - normál csíkozott háttérrel
            box(5, y, row_h, stripe, choice_txt, bold=False, size=12)

            # 6. oszlop (index: 6): Simulated Choice in Real Game (színkódolás nélkül, normál háttérrel)
            box(6, y, row_h, stripe, diff_txt, bold=False, size=12)

            y += row_h

        # Merged method cell spanning the whole block (0. oszlop)
        box(0, y_block, y - y_block, C_METHOD[label], label, bold=True, size=17)
        ax.plot([0, width], [y, y], color="#374151", linewidth=2.0)

    # Footnote
    notes = [
        "Cell content: (row, column) = (first player's index, second player's index); "
        "'a  /  b' = first player starts  /  second player starts; '*' = did not converge.",
        "Colours: green = simulated Nash equals the real one (for both move orders), "
        "red = differs, yellow = only one of the two simulated games matches.",
        f"{meta['utility_formula']}.  (w1, w2) = leakage-penalty weight of the "
        f"first / second player of the game.",
    ]
    if WEIGHT_PAIRS == "unordered":
        notes.append("Only weight pairs with w1 <= w2 are shown.")
    y_note = y + 0.3
    for line in notes:
        ax.text(0.12, y_note, line, ha="left", va="center", fontsize=10, color="#4b5563")
        y_note += 0.30

    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.generate_nash_tables import main

    main()