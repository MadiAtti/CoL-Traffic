import os
import warnings
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import minimize, differential_evolution
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error

base_path = "results/"


def monotonize_matrix(matrix):
    """
    Isotonic Regression a mátrix monotonitásának biztosítására.
    A PATF property szerint növekvő p -> csökkenő accuracy gain,
    ezért mindkét tengelyen csökkenő monotonitást kényszerítünk.
    """
    result = matrix.copy()
    ir = IsotonicRegression(increasing=False)
    for i in range(matrix.shape[0]):
        result[i, :] = ir.fit_transform(np.arange(matrix.shape[0]), result[i, :])
    for j in range(matrix.shape[1]):
        result[:, j] = ir.fit_transform(np.arange(matrix.shape[0]), result[:, j])
    return result


def apply_f(phi_tilde, D_n, grid, params):
    """
    f(Φ̃) = (x + y * D_n) * (α + β * pown + γ * pother) * Φ̃
    """
    x, y, alpha, beta, gamma = params
    g = x + y * D_n
    pown_grid, pother_grid = np.meshgrid(grid, grid, indexing='ij')
    h = alpha + beta * pown_grid + gamma * pother_grid
    return g * h * phi_tilde


def rmse(a, b):
    return np.sqrt(mean_squared_error(a.ravel(), b.ravel()))


def save_heatmap(matrix, title, filepath):
    """
    Heatmap mentése a seaborn stílusnak megfelelően (RdYlGn színskála, annotációk, középre igazított center=0).
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
    )
    plt.title(title)
    plt.xlabel("Client 2 Params")
    plt.ylabel("Client 1 Params")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Heatmap mentve: {filepath}")


def get_final_matrices():
    matrices = {}
    for size in ["full", "half"]:
        matrices[size] = {}
        for method in ["Noise", "Suppression"]:
            real = np.load(f"{base_path}{size}/accuracy/games/average/{method}_Final_Real.npy")
            pred = np.load(f"{base_path}{size}/accuracy/games/average/{method}_Final_Pred.npy")

            print(f"\n{size} | {method}")
            print(f"  real range: [{real.min():.4f}, {real.max():.4f}]")
            print(f"  pred range: [{pred.min():.4f}, {pred.max():.4f}]")

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

    bounds = [
        (-10.0, 10.0),  # x
        (-10.0, 10.0),  # y
        (-10.0, 10.0),  # alpha
        (-10.0, 10.0),  # beta
        (-10.0, 10.0),  # gamma
    ]

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
    x, y, alpha, beta, gamma = params

    transformed_full = apply_f(pred_full, D_full, grid_full, params)
    transformed_half = apply_f(pred_half, D_half, grid_half, params)

    if verbose:
        print(f"  DE loss:  {de_result.fun:.8f}")
        print(f"  NM loss:  {nm_result.fun:.8f}")
        print(f"  Paraméterek: x={x:.4f}, y={y:.4f}, α={alpha:.4f}, β={beta:.4f}, γ={gamma:.4f}")
        print(f"  RMSE full  : {rmse(pred_full, real_full):.6f} -> {rmse(transformed_full, real_full):.6f}")
        print(f"  RMSE half  : {rmse(pred_half, real_half):.6f} -> {rmse(transformed_half, real_half):.6f}")

    return params, transformed_full, transformed_half


def main():
    matrices = get_final_matrices()

    D_full = 1   # <- cseréld le a tényleges full dataset méretre
    D_half = 0.5 # <- cseréld le a tényleges half dataset méretre

    for method in ["Noise", "Suppression"]:
        print(f"\n{'='*50}")
        print(f"Method: {method}")
        print(f"{'='*50}")

        real_full = monotonize_matrix(matrices["full"][method]["real"])
        pred_full = monotonize_matrix(matrices["full"][method]["pred"])
        real_half = monotonize_matrix(matrices["half"][method]["real"])
        pred_half = monotonize_matrix(matrices["half"][method]["pred"])

        G_full = real_full.shape[0]
        G_half = real_half.shape[0]
        grid_full = np.linspace(0, 1, G_full)
        grid_half = np.linspace(0, 1, G_half)

        params, transformed_full, transformed_half = fit_f_joint(
            pred_full, real_full, D_full,
            pred_half, real_half, D_half,
            grid_full, grid_half
        )

        # Mappák létrehozása
        os.makedirs(f"{base_path}full/transformed/", exist_ok=True)
        os.makedirs(f"{base_path}half/transformed/", exist_ok=True)

        # .npy fájlok mentése
        np.save(f"{base_path}full/transformed/{method}_Transformed.npy", transformed_full)
        np.save(f"{base_path}half/transformed/{method}_Transformed.npy", transformed_half)
        np.save(f"{base_path}full/transformed/{method}_Params.npy", params)

        # Heatmapok mentése az új, Seaborn alapú stílusban (.png)
        save_heatmap(transformed_full, f"Full Transformed – {method} (Relative Accuracy Improvement)", f"{base_path}full/transformed/{method}_Transformed.png")
        save_heatmap(transformed_half, f"Half Transformed – {method} (Relative Accuracy Improvement)", f"{base_path}half/transformed/{method}_Transformed.png")
        
        save_heatmap(real_full, f"Full Real – {method} (Relative Accuracy Improvement)", f"{base_path}full/transformed/{method}_Real_heatmap.png")
        save_heatmap(real_half, f"Half Real – {method} (Relative Accuracy Improvement)", f"{base_path}half/transformed/{method}_Real_heatmap.png")

        corr = np.corrcoef(pred_full, real_full)[0, 1]
        print(f"\n{method}")
        print(f"  korreláció: {corr:.4f}")
        print(f"  pred negatív: {(pred_full < 0).mean():.1%}")
        print(f"  real negatív: {(real_full < 0).mean():.1%}")

        plt.figure(figsize=(6, 6))
        plt.scatter(pred_full, real_full, alpha=0.5, s=20)
        plt.axline((0, 0), slope=1, color='r', linestyle='--', label='y=x')
        plt.xlabel("pred")
        plt.ylabel("real")
        plt.title(f"{method} – pred vs real")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"scatter_{method}.png")
        plt.close()
        print(f"  scatter mentve: scatter_{method}.png")

if __name__ == "__main__":
    main()