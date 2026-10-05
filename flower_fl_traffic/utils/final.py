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

    # Ha nincsenek negatív vagy pozitív értékek, egy kis eltolás a stabil skálához
    if v_min >= 0:
        v_min = -1e-2
    if v_max <= 0:
        v_max = 1e-2

    return TwoSlopeNorm(vmin=v_min, vcenter=0.0, vmax=v_max)


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
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap=PURPLE_WHITE_BLUE_CMAP,
            norm=norm,
            annot_kws={"size": 11},
            xticklabels=tick_labels if tick_labels is not None else "auto",
            yticklabels=tick_labels if tick_labels is not None else "auto",
            cbar_kws={"label": cbar_label},
            square=True  # Négyzetes cellák a pontos egyezéshez
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
    print(f"Gain Difference Heatmap mentve: {filepath}")


def monotonize_matrix(matrix):
    result = matrix.copy()
    ir = IsotonicRegression(increasing=False)
    for i in range(matrix.shape[0]):
        result[i, :] = ir.fit_transform(np.arange(matrix.shape[0]), result[i, :])
    for j in range(matrix.shape[1]):
        result[:, j] = ir.fit_transform(np.arange(matrix.shape[0]), result[:, j])
    return result


def apply_f(phi_tilde, D_n, grid, params):
    x, y, alpha, beta, gamma = params
    g = x + y * D_n
    pown_grid, pother_grid = np.meshgrid(grid, grid, indexing='ij')
    h = alpha + beta * pown_grid + gamma * pother_grid
    return g * h * phi_tilde


def rmse(a, b):
    return np.sqrt(mean_squared_error(a.ravel(), b.ravel()))


def get_final_matrices():
    matrices = {}
    for size in ["full", "half"]:
        matrices[size] = {}
        for method in ["Noise", "Suppression"]:
            real = np.load(f"{base_path}{size}/accuracy/games/average/{method}_Final_Real.npy")
            pred = np.load(f"{base_path}{size}/accuracy/games/average/{method}_Final_Pred.npy")
            matrices[size][method] = {"real": real, "pred": pred}
    return matrices


def fit_f_joint(pred_full, real_full, D_full,
                pred_half, real_half, D_half,
                grid_full, grid_half, verbose=True):

    def loss(params):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transformed_full = apply_f(pred_full, D_full, grid_full, params)
            transformed_half = apply_f(pred_half, D_half, grid_half, params)
            return rmse(transformed_full, real_full) + rmse(transformed_half, real_half)

    bounds = [(-10.0, 10.0)] * 5

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


def main():
    matrices = get_final_matrices()

    D_full = 1
    D_half = 0.5

    for method in ["Noise", "Suppression"]:
        print(f"\nProcessing {method}...")

        real_full = monotonize_matrix(matrices["full"][method]["real"])
        pred_full = monotonize_matrix(matrices["full"][method]["pred"])
        real_half = monotonize_matrix(matrices["half"][method]["real"])
        pred_half = monotonize_matrix(matrices["half"][method]["half"] if "half" in matrices["half"][method] else matrices["half"][method]["pred"])

        G_full = real_full.shape[0]
        G_half = real_half.shape[0]
        grid_full = np.linspace(0, 1, G_full)
        grid_half = np.linspace(0, 1, G_half)

        params, transformed_full, transformed_half = fit_f_joint(
            pred_full, real_full, D_full,
            pred_half, real_half, D_half,
            grid_full, grid_half,
            verbose=False
        )

        # Különbség kiszámítása (Transformed - Real)
        diff_full = transformed_full - real_full
        diff_half = transformed_half - real_half

        os.makedirs(f"{base_path}full/transformed/", exist_ok=True)
        os.makedirs(f"{base_path}half/transformed/", exist_ok=True)

        full_ticks = get_display_ticks(method, mode="full")
        half_ticks = get_display_ticks(method, mode="half")

        # --- A képen látható Gain Difference Hőtérképek generálása ---
        save_diff_heatmap(
            diff_full,
            title="Gain Difference",
            filepath=f"{base_path}full/transformed/{method}_Gain_Difference.png",
            xlabel="P_other",
            ylabel="P_own",
            cbar_label="Gain difference",
            tick_labels=full_ticks
        )

        save_diff_heatmap(
            diff_half,
            title="Gain Difference",
            filepath=f"{base_path}half/transformed/{method}_Gain_Difference.png",
            xlabel="P_other",
            ylabel="P_own",
            cbar_label="Gain difference",
            tick_labels=half_ticks
        )


if __name__ == "__main__":
    main()