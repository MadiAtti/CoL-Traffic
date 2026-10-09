import os
import warnings
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import seaborn as sns
from scipy.optimize import minimize, differential_evolution
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error
from omegaconf import OmegaConf

base_path = "results/"

analyze = "accuracy"  # or "loss"

TARGET_FILES = ("dp_real.csv", "sup_real.csv")
INPUT_DIR_NAMES = {"isotonic_csv"}
SCENARIO_LABELS = {
    "full": "FULL",
    "p1": "P1 SIMULATION",
    "p2": "P2 SIMULATION",
}

def get_title_metric_name():
    if analyze == "accuracy":
        return "Isotonic Accuracy Gain"
    return "Relative Loss Change"


# --- Képen látható egyedi színskála: Lilás-Rózsaszín (Negatív) -> Fehér (0) -> Kék (Pozitív) ---
PURPLE_WHITE_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "PurpleWhiteBlue",
    [
        "#b5338a",  # Sötét bíbor / lila (negatív értékek)
        "#e891c3",  # Világos rózsaszín/lila
        "#ffffff",  # Fehér (0.0 körüli értékek)
        "#93b5e1",  # Világoskék
        "#1f4299"   # Sötétkék (pozitív értékek)
    ]
)

RWG_CMAP = LinearSegmentedColormap.from_list("RedWhiteGreen", ["#d73027", "#ffffff", "#1a9850"])
HEATMAP_FONT = {
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}

DIFF_FONT = {
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}


def _make_norm_diff(matrix: np.ndarray) -> TwoSlopeNorm:
    valid_vals = matrix[~np.isnan(matrix)]
    if len(valid_vals) == 0:
        return TwoSlopeNorm(vmin=-1e-2, vcenter=0.0, vmax=1e-2)

    v_min = float(np.nanmin(valid_vals))
    v_max = float(np.nanmax(valid_vals))

    if v_min >= 0:
        v_min = -1e-2
    if v_max <= 0:
        v_max = 1e-2

    return TwoSlopeNorm(vmin=v_min, vcenter=0.0, vmax=v_max)

def apply_f(phi_tilde, D_n, grid, params):
    x, y, alpha, beta = params
    g = x + y * D_n
    pown_grid, pother_grid = np.meshgrid(grid, grid, indexing='ij')
    h = 1 + alpha * pown_grid + beta * pother_grid
    return g * h * phi_tilde


def rmse(a, b):
    return np.sqrt(mean_squared_error(a.ravel(), b.ravel()))

def fit_f_joint(pred_full, real_full, D_full,
    pred_half, real_half, D_half,
    grid_full, grid_half, verbose=True):


    def loss(params):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transformed_full = apply_f(pred_full, D_full, grid_full, params)
            transformed_half = apply_f(pred_half, D_half, grid_half, params)
        return rmse(transformed_full, real_full) + rmse(transformed_half, real_half)

    bounds = [(-10.0, 10.0)] * 4

    de_result = differential_evolution(
    loss,
    bounds=bounds,
    maxiter=10000,
    tol=1e-10,
    seed=42,
    popsize=20,
    mutation=(0.5, 1),
    recombination=0.9,
    polish=True,
    workers=1,
    disp=verbose,
    )

    nm_result = minimize(
    loss,
    x0=de_result.x,
    method='Nelder-Mead',
    options={'maxiter': 100000, 'xatol': 1e-10, 'fatol': 1e-10}
    )

    params = nm_result.x if nm_result.fun < de_result.fun else de_result.x
    transformed_full = apply_f(pred_full, D_full, grid_full, params)
    transformed_half = apply_f(pred_half, D_half, grid_half, params)

    return params, transformed_full, transformed_half

def get_params_for_method(method_name, mode, config_data):
    if method_name == "Noise":
        level_map = {
            "full": "full_noise_levels",
            "half": "half_noise_levels",
            "80percent": "80percent_noise_levels",
            "30percent": "30percent_noise_levels",
        }
        level = level_map.get(mode, "full_noise_levels")
        return config_data[level], "noise_p1", "noise_p2"
    else:
        return config_data["sup_levels"], "features_p1", "features_p2"


def get_display_ticks(method, mode="full"):
    base_cfg = OmegaConf.load("conf/base.yaml")
    config_data = base_cfg.config
    params, _, _ = get_params_for_method(method, mode, config_data)
    
    if method == "Noise":
        ticks = [f"{(x / (x + 1)):.2f}" if x is not None else "0.00" for x in params]
    else:
        ticks = [f"{((14 - x) / 14):.2f}" if x is not None else "0.00" for x in params]
        
    return ticks + ["1"]


def save_diff_heatmap(matrix: np.ndarray, title: str, filepath: str,
                      xlabel: str = "P_other", ylabel: str = "P_own",
                      cbar_label: str = "Gain difference",
                      tick_labels=None) -> None:
    norm = _make_norm_diff(matrix)
    with plt.rc_context(DIFF_FONT):
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap=PURPLE_WHITE_BLUE_CMAP,
            norm=norm,
            annot_kws={"size": 11},
            xticklabels=tick_labels if tick_labels is not None else "auto",
            yticklabels=tick_labels if tick_labels is not None else "auto",
            cbar_kws={"label": cbar_label},
            square=True
        )
        ax.invert_yaxis()  # <-- (0,0) bal alul, (1,1) jobb felül
        ax.tick_params(axis="y", rotation=0)  # y feliratok maradjanak vízszintesek
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
    print(f"Gain Difference Heatmap mentve: {filepath}")


def _make_norm(matrix: np.ndarray) -> TwoSlopeNorm:
    valid_vals = matrix[~np.isnan(matrix)]

    if len(valid_vals) > 1:
        v_min = float(np.partition(valid_vals, 1)[1])
    elif len(valid_vals) == 1:
        v_min = float(valid_vals[0])
    else:
        v_min = -1e-2

    v_max = float(np.nanmax(matrix))
    if v_min >= 0:
        v_min = -1e-2
    if v_max <= 0:
        v_max = 1e-2
    return TwoSlopeNorm(vmin=v_min, vcenter=0.0, vmax=v_max)


def _save_heatmap(matrix: np.ndarray, title: str, xlabel: str, ylabel: str,
                  save_path: str, tick_labels=None, std_matrix: np.ndarray = None) -> None:
    """
    Frissített hőtérkép rajzoló: Ha meg van adva std_matrix, formázott 
    'átlag \n ±szórás' szöveges annotációt rajzol ki minden mezőbe.
    """
    norm = _make_norm(matrix)
    
    # Szöveges annotációs mátrix előállítása
    if std_matrix is not None:
        annot_matrix = np.empty(matrix.shape, dtype=object)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                std = std_matrix[i, j]
                if np.isnan(val):
                    annot_matrix[i, j] = ""
                else:
                    std_val = 0.0 if np.isnan(std) else std
                    annot_matrix[i, j] = f"{val:.3f}\n±{std_val:.3f}"
        annot_arg = annot_matrix
        fmt_arg = ""  # Mivel az annot_matrix már előre megformázott stringeket tartalmaz
    else:
        annot_arg = True
        fmt_arg = ".3f"

    with plt.rc_context(HEATMAP_FONT):
        plt.figure(figsize=(12, 9))
        ax = sns.heatmap(
            matrix,
            annot=annot_arg,
            fmt=fmt_arg,
            cmap=RWG_CMAP,
            norm=norm,
            annot_kws={"size": 10},
            xticklabels=tick_labels if tick_labels is not None else "auto",
            yticklabels=tick_labels if tick_labels is not None else "auto",
            cbar_kws={"label": get_title_metric_name()},
        )
        ax.invert_yaxis()  # <-- y tengely megtükrözése
        ax.tick_params(axis="y", rotation=0)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    print(f"Saved: {save_path}")


def monotonize_matrix(matrix):
    result = matrix.copy()
    ir = IsotonicRegression(increasing=False)
    for i in range(matrix.shape[0]):
        result[i, :] = ir.fit_transform(np.arange(matrix.shape[0]), result[i, :])
    for j in range(matrix.shape[1]):
        result[:, j] = ir.fit_transform(np.arange(matrix.shape[0]), result[:, j])
    return result


def get_final_matrices():
    matrices = {}
    for size in ["full", "half"]:
        matrices[size] = {}
        for method in ["Noise", "Suppression"]:
            real = np.load(f"{base_path}{size}/accuracy/games/average/{method}_Final_Real.npy")
            pred = np.load(f"{base_path}{size}/accuracy/games/average/{method}_Final_Pred.npy")
            
            # Szórásfájlok beolvasása (ha nem léteznek, nullákkal helyettesítjük)
            real_std_path = f"{base_path}{size}/accuracy/games/average/{method}_Final_Real_Std.npy"
            pred_std_path = f"{base_path}{size}/accuracy/games/average/{method}_Final_Pred_Std.npy"
            
            real_std = np.load(real_std_path) if os.path.exists(real_std_path) else np.zeros_like(real)
            pred_std = np.load(pred_std_path) if os.path.exists(pred_std_path) else np.zeros_like(pred)

            matrices[size][method] = {
                "real": real, 
                "pred": pred,
                "real_std": real_std,
                "pred_std": pred_std
            }
    return matrices

def print_params(params):
    param_names = ['x', 'y', 'alpha', 'beta']
    return ', '.join(f"{name}={value:.8f}" for name, value in zip(param_names, params))


def main():
    matrices = get_final_matrices()

    for method in ["Noise", "Suppression"]:
        print(f"\nProcessing {method}...")

        real_full = monotonize_matrix(matrices["full"][method]["real"])
        pred_full = monotonize_matrix(matrices["full"][method]["pred"])
        real_half = monotonize_matrix(matrices["half"][method]["real"])
        pred_half = monotonize_matrix(matrices["half"][method]["half"] if "half" in matrices["half"][method] else matrices["half"][method]["pred"])

        real_full_std = matrices["full"][method]["real_std"]
        pred_full_std = matrices["full"][method]["pred_std"]
        real_half_std = matrices["half"][method]["real_std"]
        pred_half_std = matrices["half"][method]["pred_std"]

        # Különbség kiszámítása
        diff_full = pred_full - real_full
        diff_half = pred_half - real_half

        os.makedirs(f"{base_path}final/", exist_ok=True)

        full_ticks = get_display_ticks(method, mode="full")
        half_ticks = get_display_ticks(method, mode="half")

        if method == "Noise":
            method_label = "DP"
            method_name = "DP"
        else:
            method_label = "SUP"
            method_name = "Sup"

        # --- Gain Difference Hőtérképek ---
        save_diff_heatmap(
            diff_full,
            title="Gain Difference",
            filepath=f"{base_path}final/DIFF_NT_{method_name}_size2.png",
            xlabel="P_other",
            ylabel="P_own",
            cbar_label="Gain difference",
            tick_labels=full_ticks
        )

        save_diff_heatmap(
            diff_half,
            title="Gain Difference",
            filepath=f"{base_path}final/DIFF_NT_{method_name}_size1.png",
            xlabel="P_other",
            ylabel="P_own",
            cbar_label="Gain difference",
            tick_labels=half_ticks
        )

        SD_title = f"Isotonic RMSE Gain - SIM AVG (4 SD variants: P11, P12, P21, P22) - {method_label}\n(Mean ± STD over 4 variants)"
        real_title = f"RMSE Gain - REAL - AVG(P_own, P_other) ({method_label})\n(Mean ± STD over Seeds)"

        # --- Átlag + Szórás Hőtérképek Generálása ---
        _save_heatmap(
            matrix=real_full,
            std_matrix=real_full_std,
            title=real_title,
            xlabel=f"P_other Privacy Parameter ({method_label})",  
            ylabel=f"P_own Privacy Parameter ({method_label})", 
            save_path=f"{base_path}final/NT_{method_name}_size2.png",
            tick_labels=full_ticks,
        )

        _save_heatmap(
            matrix=pred_full,
            std_matrix=pred_full_std,
            title=SD_title,
            xlabel=f"P_other Privacy Parameter ({method_label})",  
            ylabel=f"P_own Privacy Parameter ({method_label})", 
            save_path=f"{base_path}final/SD_NT_{method_name}_size2.png",
            tick_labels=full_ticks,
        )

        _save_heatmap(
            matrix=real_half,
            std_matrix=real_half_std,
            title=real_title,
            xlabel=f"P_other Privacy Parameter ({method_label})",  
            ylabel=f"P_own Privacy Parameter ({method_label})", 
            save_path=f"{base_path}final/NT_{method_name}_size1.png",
            tick_labels=half_ticks,
        )

        _save_heatmap(
            matrix=pred_half,
            std_matrix=pred_half_std,
            title=SD_title,
            xlabel=f"P_other Privacy Parameter ({method_label})",  
            ylabel=f"P_own Privacy Parameter ({method_label})", 
            save_path=f"{base_path}final/SD_NT_{method_name}_size1.png",
            tick_labels=half_ticks,
        )

        G_full = real_full.shape[0]
        G_half = real_half.shape[0]
        grid_full = np.linspace(0, 1, G_full)
        grid_half = np.linspace(0, 1, G_half)

        D_full = 3163462
        D_half = 1581731

        params, transformed_full, transformed_half = fit_f_joint(
        pred_full, real_full, D_full,
        pred_half, real_half, D_half,
        grid_full, grid_half,
        verbose=False
        )

        print(f"\nBefore fitting RMSE Full: {rmse(pred_full, real_full):.6f}, RMSE Half: {rmse(pred_half, real_half):.6f}")
        print(f"\nFitted parameters for {method}: {print_params(params)}, RMSE Full: {rmse(transformed_full, real_full):.6f}, RMSE Half: {rmse(transformed_half, real_half):.6f}")



if __name__ == "__main__":
    main()