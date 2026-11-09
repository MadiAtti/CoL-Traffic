"""
Final heatmaps and simulated->real model fitting for the federated-learning
privacy game. Input: the averaged matrices written by build_matrices.py.
"""

import json
import os
import warnings

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from omegaconf import OmegaConf
from scipy.optimize import differential_evolution, minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error

# --------------------------------------------------------------------------- #
# Global settings
# --------------------------------------------------------------------------- #
base_path = "results/"
analyze = "accuracy"
DATASET_SIZE_DESCRIPTORS = {
    "full": 1.581731,
    "half": 0.790865,
    "80percent": 1.265384,
    "30percent": 0.474519,
}


def get_title_metric_name():
    """Return the colour-bar label that matches the analysed metric."""
    if analyze == "accuracy":
        return "Gain"
    return "Relative Loss Change"


# --------------------------------------------------------------------------- #
# Colour maps and font settings
# --------------------------------------------------------------------------- #
PURPLE_WHITE_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "PurpleWhiteBlue",
    [
        "#b5338a",  # dark magenta
        "#e891c3",  # light pink
        "#ffffff",  # white
        "#93b5e1",  # light blue
        "#1f4299",  # dark blue
    ],
)

RWG_CMAP = LinearSegmentedColormap.from_list(
    "RedWhiteGreen", ["#d73027", "#ffffff", "#1a9850"]
)

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


# --------------------------------------------------------------------------- #
# Colour normalisation helpers
# --------------------------------------------------------------------------- #
def _make_norm_diff(matrix: np.ndarray) -> TwoSlopeNorm:
    """Build a symmetric, zero-centered color normalization for differences."""
    valid_vals = matrix[np.isfinite(matrix)]
    if len(valid_vals) == 0:
        return TwoSlopeNorm(vmin=-1e-2, vcenter=0.0, vmax=1e-2)

    v_min = float(np.nanmin(valid_vals))
    v_max = float(np.nanmax(valid_vals))

    if v_min >= 0:
        v_min = -1e-2
    if v_max <= 0:
        v_max = 1e-2

    return TwoSlopeNorm(vmin=v_min, vcenter=0.0, vmax=v_max)


def _make_norm(matrix: np.ndarray) -> TwoSlopeNorm:
    """Build a zero-centered color normalization using finite matrix values."""
    valid_vals = matrix[np.isfinite(matrix)]

    if len(valid_vals) > 1:
        v_min = float(np.partition(valid_vals, min(1, len(valid_vals) - 1))[1])
    elif len(valid_vals) == 1:
        v_min = float(valid_vals[0])
    else:
        v_min = -1e-2

    v_max = float(np.max(valid_vals)) if len(valid_vals) else 1e-2
    if v_min >= 0:
        v_min = -1e-2
    if v_max <= 0:
        v_max = 1e-2
    return TwoSlopeNorm(vmin=v_min, vcenter=0.0, vmax=v_max)


# --------------------------------------------------------------------------- #
# Grid computation & Transformation model
# --------------------------------------------------------------------------- #
def get_params_for_method(method_name, mode, config_data):
    """Return configured parameter values and JSON keys for one method."""
    if method_name == "Noise":
        level_map = {
            "full": "full_noise_levels",
            "half": "half_noise_levels",
            "80percent": "percent80_noise_levels",
            "30percent": "percent30_noise_levels",
        }
        if mode not in level_map:
            raise ValueError(f"Unsupported dataset mode for noise: {mode}")
        return config_data[level_map[mode]], "noise_p1", "noise_p2"
    if method_name == "Suppression":
        return config_data["sup_levels"], "features_p1", "features_p2"
    raise ValueError(f"Unsupported privacy method: {method_name}")


def get_actual_grid(method, mode="full"):
    """Map configured privacy settings to normalized protection values.

    The returned values follow the configured grid and include a final ``1.0``
    for the maximum-privacy echo client.
    """
    base_cfg = OmegaConf.load("conf/base.yaml")
    config_data = base_cfg.config
    params, _, _ = get_params_for_method(method, mode, config_data)

    grid = []
    for x in params:
        if x is None:
            grid.append(0.0)
        elif method == "Noise":
            grid.append(x / (x + 1.0))
        else:  # Suppression
            grid.append((14.0 - x) / 14.0)

    grid.append(1.0)  # Maximum védelmi szint
    return np.array(grid, dtype=float)


def get_display_ticks(method, mode="full"):
    """Return formatted privacy-grid labels for plot axes."""
    base_cfg = OmegaConf.load("conf/base.yaml")
    config_data = base_cfg.config
    params, _, _ = get_params_for_method(method, mode, config_data)

    if method == "Noise":
        ticks = [f"{(x / (x + 1)):.2f}" if x is not None else "0.00" for x in params]
    else:
        ticks = [f"{((14 - x) / 14):.2f}" if x is not None else "0.00" for x in params]

    return ticks + ["1"]


def get_dataset_size_descriptor(mode):
    """Return one participant's configured dataset-size descriptor."""
    try:
        return DATASET_SIZE_DESCRIPTORS[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset mode for size descriptor: {mode}") from exc


def apply_f(phi_tilde, D_n, grid, params):
    """Apply the fitted multiplicative correction to a predicted matrix.

    Args:
        phi_tilde: Predicted payoff matrix.
        D_n: Dataset-size descriptor.
        grid: Normalized protection levels used by both matrix axes.
        params: Fitted coefficients ``(x, y, alpha, beta)``.

    Returns:
        The transformed matrix, preserving the input shape.
    """
    x, y, alpha, beta = params
    g = x + y * D_n
    pown_grid, pother_grid = np.meshgrid(grid, grid, indexing="ij")
    h = 1 + alpha * pown_grid + beta * pother_grid
    return g * h * phi_tilde


def rmse(a, b, mask=None):
    """Compute RMSE over finite entries, optionally restricted by a mask."""
    if mask is not None:
        a = a[mask]
        b = b[mask]
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        raise ValueError("RMSE cannot be computed without finite matrix values.")
    a = a[valid]
    b = b[valid]
    return np.sqrt(mean_squared_error(a.ravel(), b.ravel()))


def fit_f_joint(pred_big, real_big, D_big,
                pred_small, real_small, D_small,
                grid_big, grid_small, verbose=True):
    """Fit shared transformation coefficients to two dataset-size matrices.

    Args:
        pred_big: Predicted matrix for the larger dataset.
        real_big: Empirical matrix for the larger dataset.
        D_big: Dataset-size descriptor for the larger dataset.
        pred_small: Predicted matrix for the smaller dataset.
        real_small: Empirical matrix for the smaller dataset.
        D_small: Dataset-size descriptor for the smaller dataset.
        grid_big: Normalized privacy levels for the larger matrix.
        grid_small: Normalized privacy levels for the smaller matrix.
        verbose: Whether to display optimizer progress.

    Returns:
        Fitted parameters, transformed matrices, and their finite-value masks.
    """
    for prediction, actual, grid, label in (
        (pred_big, real_big, grid_big, "big"),
        (pred_small, real_small, grid_small, "small"),
    ):
        if prediction.shape != actual.shape:
            raise ValueError(f"{label} predicted/real matrix shapes do not match.")
        if prediction.ndim != 2 or prediction.shape != (len(grid), len(grid)):
            raise ValueError(f"{label} matrix shape does not match its privacy grid.")

    mask_big = np.isfinite(pred_big) & np.isfinite(real_big)
    mask_big[-1, -1] = False
    mask_small = np.isfinite(pred_small) & np.isfinite(real_small)
    mask_small[-1, -1] = False
    if not np.any(mask_big) or not np.any(mask_small):
        raise ValueError("Each dataset split needs at least one finite non-corner cell.")

    def loss(params):
        """Return the combined masked RMSE for the current coefficients."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transformed_big = monotonize_matrix(
                apply_f(pred_big, D_big, grid_big, params)
            )
            transformed_small = monotonize_matrix(
                apply_f(pred_small, D_small, grid_small, params)
            )
        return (rmse(transformed_big, real_big, mask=mask_big)
                + rmse(transformed_small, real_small, mask=mask_small))

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
        method="Nelder-Mead",
        options={"maxiter": 100000, "xatol": 1e-10, "fatol": 1e-10},
    )

    params = nm_result.x if nm_result.fun < de_result.fun else de_result.x
    transformed_big = monotonize_matrix(
        apply_f(pred_big, D_big, grid_big, params)
    )
    transformed_small = monotonize_matrix(
        apply_f(pred_small, D_small, grid_small, params)
    )

    return params, transformed_big, transformed_small, mask_big, mask_small


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def save_diff_heatmap(matrix: np.ndarray, title: str, filepath: str,
                      xlabel: str = "P_other", ylabel: str = "P_own",
                      cbar_label: str = r"$\Delta - f(\tilde{\Delta})$",
                      tick_labels=None) -> None:
    """Save a zero-centered heatmap of the post-correction residual."""
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
            square=True,
        )
        ax.invert_yaxis()
        ax.tick_params(axis="y", rotation=0)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
    print(f"Gain difference heatmap saved: {filepath}")


def _save_heatmap(matrix: np.ndarray, title: str, xlabel: str, ylabel: str,
                  save_path: str, tick_labels=None,
                  std_matrix: np.ndarray = None) -> None:
    """Save one payoff heatmap, optionally annotating mean and standard deviation."""
    norm = _make_norm(matrix)

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
        fmt_arg = ""
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
        ax.invert_yaxis()
        ax.tick_params(axis="y", rotation=0)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    print(f"Saved: {save_path}")


def monotonize_matrix(matrix):
    """Apply decreasing isotonic regression to finite values in every row/column."""
    result = np.asarray(matrix, dtype=float).copy()
    ir = IsotonicRegression(increasing=False)

    def monotonize(values):
        """Fit decreasing isotonic values while leaving missing entries intact."""
        valid = np.isfinite(values)
        if np.count_nonzero(valid) > 1:
            indices = np.flatnonzero(valid)
            values[valid] = ir.fit_transform(indices, values[valid])

    for i in range(matrix.shape[0]):
        monotonize(result[i, :])
    for j in range(matrix.shape[1]):
        monotonize(result[:, j])
    return result


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def get_final_matrices(size_big, size_small):
    """Load matrices dynamically according to the dataset sizes."""
    matrices = {}
    for size, key in [(size_big, "big"), (size_small, "small")]:
        matrices[key] = {}
        for method in ["Noise", "Suppression"]:
            folder = f"{base_path}{size}/accuracy/games/average/"
            real = np.load(f"{folder}{method}_Final_Real.npy")
            pred = np.load(f"{folder}{method}_Final_Pred.npy")
            if real.ndim != 2 or pred.shape != real.shape:
                raise ValueError(
                    f"Incompatible real/predicted matrices for {size}/{method}: "
                    f"{real.shape} vs {pred.shape}"
                )

            real_std_path = f"{folder}{method}_Final_Real_Std.npy"
            pred_std_path = f"{folder}{method}_Final_Pred_Std.npy"

            real_std = (np.load(real_std_path) if os.path.exists(real_std_path)
                        else np.zeros_like(real))
            pred_std = (np.load(pred_std_path) if os.path.exists(pred_std_path)
                        else np.zeros_like(pred))
            if real_std.shape != real.shape or pred_std.shape != pred.shape:
                raise ValueError(f"Standard-deviation matrix shape mismatch for {size}/{method}.")

            matrices[key][method] = {
                "real": real,
                "pred": pred,
                "real_std": real_std,
                "pred_std": pred_std,
            }
    return matrices


def print_params(params):
    """Format the fitted correction coefficients as a comma-separated string."""
    param_names = ["x", "y", "alpha", "beta"]
    return ", ".join(f"{name}={value:.8f}" for name, value in zip(param_names, params))


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main(dataset_split="100-50"):
    """Generate heatmaps and corrected matrices for a supported data split.

    Args:
        dataset_split: Either ``"100-50"`` or ``"80-30"``.

    Raises:
        ValueError: If ``dataset_split`` is not one of the supported pairs.
    """
    mode = "both"

    # Dynamic setup based on split
    if dataset_split == "100-50":
        size_big, size_small = "full", "half"
    elif dataset_split == "80-30":
        size_big, size_small = "80percent", "30percent"
    else:
        raise ValueError(f"Unknown dataset_split: {dataset_split}")
    D_big = get_dataset_size_descriptor(size_big)
    D_small = get_dataset_size_descriptor(size_small)

    matrices = get_final_matrices(size_big, size_small)

    for method in ["Noise", "Suppression"]:
        print(f"\nProcessing {method} ({dataset_split})...")

        real_big = matrices["big"][method]["real"]
        pred_big = matrices["big"][method]["pred"]
        real_small = matrices["small"][method]["real"]
        pred_small = matrices["small"][method]["pred"]

        real_big_std = matrices["big"][method]["real_std"]
        pred_big_std = matrices["big"][method]["pred_std"]
        real_small_std = matrices["small"][method]["real_std"]
        pred_small_std = matrices["small"][method]["pred_std"]

        # Get actual grids using config values
        grid_big = get_actual_grid(method, mode=size_big)
        grid_small = get_actual_grid(method, mode=size_small)

        out_dir = f"{base_path}final_{dataset_split}/"
        os.makedirs(out_dir, exist_ok=True)

        # Fit the simulated-to-real correction before plotting its residual.
        params, transformed_big, transformed_small, mask_big, mask_small = fit_f_joint(
            pred_big, real_big, D_big,
            pred_small, real_small, D_small,
            grid_big, grid_small,
            verbose=False,
        )

        rmse_before_big = rmse(pred_big, real_big, mask=mask_big)
        rmse_before_small = rmse(pred_small, real_small, mask=mask_small)
        rmse_after_big = rmse(transformed_big, real_big, mask=mask_big)
        rmse_after_small = rmse(transformed_small, real_small, mask=mask_small)

        print(f"Before fitting RMSE Big ({size_big}): {rmse_before_big:.6f}, Small ({size_small}): {rmse_before_small:.6f}")
        print(f"Fitted parameters for {method}: {print_params(params)}")
        print(f"After fitting RMSE Big ({size_big}): {rmse_after_big:.6f}, Small ({size_small}): {rmse_after_small:.6f}")

        # ---- Heatmaps ----
        if mode in ["heatmap", "both"]:
            diff_big = real_big - transformed_big
            diff_small = real_small - transformed_small

            big_ticks = get_display_ticks(method, mode=size_big)
            small_ticks = get_display_ticks(method, mode=size_small)

            method_label = "DP" if method == "Noise" else "SUP"
            method_name = "DP" if method == "Noise" else "Sup"

            # Difference heatmaps
            save_diff_heatmap(
                diff_big,
                title=rf"Correction residual: $\Delta - f(\tilde{{\Delta}})$ ({size_big})",
                filepath=f"{out_dir}DIFF_NT_{method_name}_size2.png",
                tick_labels=big_ticks,
            )
            save_diff_heatmap(
                diff_small,
                title=rf"Correction residual: $\Delta - f(\tilde{{\Delta}})$ ({size_small})",
                filepath=f"{out_dir}DIFF_NT_{method_name}_size1.png",
                tick_labels=small_ticks,
            )

            # Gain heatmaps
            SD_title = f"Gain - SIM AVG ({method_label})\n(Mean ± STD over 4 variants)"
            real_title = f"Gain - REAL ({method_label})\n(Mean ± STD over Seeds)"

            _save_heatmap(
                matrix=real_big,
                std_matrix=real_big_std,
                title=f"{real_title} - {size_big}",
                xlabel=f"P_other ({method_label})",
                ylabel=f"P_own ({method_label})",
                save_path=f"{out_dir}NT_{method_name}_size2.png",
                tick_labels=big_ticks,
            )
            _save_heatmap(
                matrix=pred_big,
                std_matrix=pred_big_std,
                title=f"{SD_title} - {size_big}",
                xlabel=f"P_other ({method_label})",
                ylabel=f"P_own ({method_label})",
                save_path=f"{out_dir}SD_NT_{method_name}_size2.png",
                tick_labels=big_ticks,
            )

            _save_heatmap(
                matrix=real_small,
                std_matrix=real_small_std,
                title=f"{real_title} - {size_small}",
                xlabel=f"P_other ({method_label})",
                ylabel=f"P_own ({method_label})",
                save_path=f"{out_dir}NT_{method_name}_size1.png",
                tick_labels=small_ticks,
            )
            _save_heatmap(
                matrix=pred_small,
                std_matrix=pred_small_std,
                title=f"{SD_title} - {size_small}",
                xlabel=f"P_other ({method_label})",
                ylabel=f"P_own ({method_label})",
                save_path=f"{out_dir}SD_NT_{method_name}_size1.png",
                tick_labels=small_ticks,
            )

        # ---- Save the fitted simulated-to-real correction ----
        if mode in ["transform", "both"]:
            params_dict = {
                "method": method,
                "split": dataset_split,
                "params": {
                    "x": float(params[0]),
                    "y": float(params[1]),
                    "alpha": float(params[2]),
                    "beta": float(params[3]),
                },
                "rmse_before": {
                    "big": float(rmse_before_big),
                    "small": float(rmse_before_small),
                },
                "rmse_after": {
                    "big": float(rmse_after_big),
                    "small": float(rmse_after_small),
                },
            }

            with open(f"{out_dir}{method}_params.json", "w") as f:
                json.dump(params_dict, f, indent=4)


if __name__ == "__main__":
    main(dataset_split="100-50")  # or "80-30"