import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, dual_annealing, least_squares, minimize


def learn_equation5_parameters():
  directories = ["NF_1", "NF_5"]
  all_real, all_sim = [], []

  # 1. Load data from directory structure
  for d in directories:
    paths = {
        "real_dp": os.path.join(d, "real_dp.csv"),
        "real_sup": os.path.join(d, "real_sup.csv"),
        "sim_dp": os.path.join(d, "sim_dp.csv"),
        "sim_sup": os.path.join(d, "sim_sup.csv"),
    }
    if os.path.exists(paths["real_dp"]):
      all_real.append(pd.read_csv(paths["real_dp"]))
    if os.path.exists(paths["real_sup"]):
      all_real.append(pd.read_csv(paths["real_sup"]))
    if os.path.exists(paths["sim_dp"]):
      all_sim.append(pd.read_csv(paths["sim_dp"]))
    if os.path.exists(paths["sim_sup"]):
      all_sim.append(pd.read_csv(paths["sim_sup"]))

  if not all_real or not all_sim:
    print("Data files not found in the specified directories.")
    return

  df_real = pd.concat(all_real, ignore_index=True)
  df_sim = pd.concat(all_sim, ignore_index=True)

  # Merge on unique configuration features
  df = pd.merge(
      df_real,
      df_sim,
      on=["Mechanism", "Samples", "P_own", "P_other"],
      suffixes=("_real", "_sim"),
  )

  # Variable extraction for Equation 5
  D_n = df["Samples"].values
  p_own = df["P_own"].values
  p_other = df["P_other"].values
  phi_real = df["Gain_Acc_real"].values
  phi_sim = df["Gain_Acc_sim"].values

  # Baseline RMSE without transformation function
  rmse_without_overall = np.sqrt(np.mean((phi_real - phi_sim) ** 2))

  # Objective functions: scalar RMSE and residual vector
  def objective(params):
    x, y, alpha, beta, gamma = params
    f_phi = (
        (x + y * D_n) * (alpha + beta * p_own + gamma * p_other) * phi_sim
    )
    return np.sqrt(np.mean((phi_real - f_phi) ** 2))

  def residuals(params):
    x, y, alpha, beta, gamma = params
    f_phi = (
        (x + y * D_n) * (alpha + beta * p_own + gamma * p_other) * phi_sim
    )
    return (phi_real - f_phi) / np.sqrt(len(phi_real))

  bounds = [(-10, 10), (-1e-3, 1e-3), (-10, 10), (-10, 10), (-10, 10)]
  lb = [b[0] for b in bounds]
  ub = [b[1] for b in bounds]
  init_guess = [1.0, 0.0, 1.0, 0.0, 0.0]

  results = {}

  # =========================================================================
  # OPTION 1: Differential Evolution (Global Stochastic Genetic Search)
  # =========================================================================
  t0 = time.time()
  res_de = differential_evolution(objective, bounds, seed=42)
  results["Differential Evolution"] = {
      "params": res_de.x,
      "rmse": res_de.fun,
      "time": time.time() - t0,
  }

  # =========================================================================
  # OPTION 2: Trust-Region Reflective (Specialized Bounded Least Squares)
  # =========================================================================
  t0 = time.time()
  res_trf = least_squares(residuals, init_guess, bounds=(lb, ub), method="trf")
  rmse_trf = np.sqrt(np.sum(residuals(res_trf.x) ** 2))
  results["Trust-Region Reflective (Least Squares)"] = {
      "params": res_trf.x,
      "rmse": rmse_trf,
      "time": time.time() - t0,
  }

  # =========================================================================
  # OPTION 3: Multi-Start L-BFGS-B (Bounded Gradient Descent with Restarts)
  # =========================================================================
  t0 = time.time()
  best_ms_fun = float("inf")
  best_ms_params = None
  np.random.seed(42)
  for _ in range(30):
    p0 = [np.random.uniform(b[0], b[1]) for b in bounds]
    res_m = minimize(objective, p0, method="L-BFGS-B", bounds=bounds)
    if res_m.fun < best_ms_fun:
      best_ms_fun = res_m.fun
      best_ms_params = res_m.x
  results["Multi-Start L-BFGS-B (30 Restarts)"] = {
      "params": best_ms_params,
      "rmse": best_ms_fun,
      "time": time.time() - t0,
  }

  # =========================================================================
  # OPTION 4: Dual Annealing (Generalized Simulated Annealing + Local Search)
  # =========================================================================
  t0 = time.time()
  res_da = dual_annealing(objective, bounds, seed=42)
  results["Dual Annealing"] = {
      "params": res_da.x,
      "rmse": res_da.fun,
      "time": time.time() - t0,
  }

  # Keep every method tied at the lowest RMSE at the report precision.
  best_rmse = min(data["rmse"] for data in results.values())
  best_methods = [
      method_name
      for method_name, data in results.items()
      if np.isclose(data["rmse"], best_rmse, rtol=0.0, atol=0.5e-6)
  ]

  # 3. Output comparison and Table 2 information to text file
  with open("Table2_RMSE_Results.txt", "w") as f:
    f.write("=====================================================\n")
    f.write("      OPTIMIZATION BENCHMARK & COMPARISON            \n")
    f.write("=====================================================\n")
    f.write(
        "Baseline Overall RMSE (Without Function):"
        f" {rmse_without_overall:.6f}\n\n"
    )

    for method_name, data in results.items():
      f.write(f"[{method_name}]\n")
      f.write(
          f"  Achieved RMSE: {data['rmse']:.6f} | Runtime: {data['time']:.4f}s\n"
      )
      p = data["params"]
      f.write(
          f"  Parameters: x={p[0]:.6f}, y={p[1]:.6e}, alpha={p[2]:.6f},"
          f" beta={p[3]:.6f}, gamma={p[4]:.6f}\n\n"
      )

    f.write("=====================================================\n")
    f.write("  SELECTED BEST METHODS (TIED MINIMUMS)\n")
    f.write("=====================================================\n")
    f.write(f"Optimal Minimum RMSE (Overall): {best_rmse:.6f}\n")
    for method_name in best_methods:
      p = results[method_name]["params"]
      f.write(f"Method: {method_name}\n")
      f.write(
        f"Universal Weights: x={p[0]:.6f}, y={p[1]:.6e}, alpha={p[2]:.6f},"
        f" beta={p[3]:.6f}, gamma={p[4]:.6f}\n\n"
      )

    f.write("=====================================================\n")
    f.write("    TABLE 2: RMSE PER DATASET SIZE (EVALUATION)      \n")
    f.write("=====================================================\n")
    for method_name in best_methods:
      x, y, alpha, beta, gamma = results[method_name]["params"]
      f.write(f"Evaluation using: {method_name}\n")
      for size in sorted(df["Samples"].unique()):
        df_s = df[df["Samples"] == size]
        D_s = df_s["Samples"].values
        po_s = df_s["P_own"].values
        pot_s = df_s["P_other"].values
        phir_s = df_s["Gain_Acc_real"].values
        phis_s = df_s["Gain_Acc_sim"].values

        rmse_wo_s = np.sqrt(np.mean((phir_s - phis_s) ** 2))
        f_phi_s = (x + y * D_s) * (alpha + beta * po_s + gamma * pot_s) * phis_s
        rmse_w_s = np.sqrt(np.mean((phir_s - f_phi_s) ** 2))
        reduction = ((rmse_wo_s - rmse_w_s) / rmse_wo_s) * 100.0

        f.write(f"Dataset Size (Samples = {size}):\n")
        f.write(f"  Raw SD RMSE (Without Function): {rmse_wo_s:.6f}\n")
        f.write(
            f"  Transformed SD RMSE (With Function f(Phi)): {rmse_w_s:.6f}\n"
        )
        f.write(f"  Error Reduction: {reduction:.2f}%\n\n")

  print(
        f"Optimization complete. Best methods: {', '.join(best_methods)} "
        f"(RMSE: {best_rmse:.6f})"
  )
  print("Results saved to 'Table2_RMSE_Results.txt'.")


if __name__ == "__main__":
  learn_equation5_parameters()