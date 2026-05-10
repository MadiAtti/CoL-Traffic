import os
import numpy as np

def load_games(games):
    results = []
    for game_file in os.listdir(games):
        if game_file.endswith("_bimatrix.npy"):
            
            # Load the bimatrix containing the accuracy drops for P1 and P2 across all parameter combinations
            bimatrix_path = os.path.join(games, game_file)
            bimatrix = np.load(bimatrix_path)
            # print(f"Loaded bimatrix from {bimatrix_path} with shape {bimatrix.shape}")

            # Determine method and scenario from filename and add to results
            method = "Suppression" if "Suppression" in game_file else "Noise"
            sc_name = "Full" if "Full" in game_file else ("P1" if "P1" in game_file else "P2")
            results.append((bimatrix, method, sc_name))

    return results

def search_NE(bimatrix, method, sc_name):
    # A Nash Equilibrium is a strategy profile where no player can unilaterally deviate to improve their payoff
    # In our case, we want to find parameter combinations where neither P1 nor P2 can improve their accuracy by changing their parameter alone

    size = bimatrix.shape[0]
    NE_points = []

    for i in range(size):
        for j in range(size):
            p1_diff = bimatrix[i, j, 0]  # Accuracy drop for P1 at (i, j)
            p2_diff = bimatrix[i, j, 1]  # Accuracy drop for P2 at (i, j)

            # Check if P1 can improve by changing its parameter (i.e., moving to any other row while keeping j fixed)
            p1_can_improve = any(bimatrix[k, j, 0] > p1_diff for k in range(size) if k != i)

            # Check if P2 can improve by changing its parameter (i.e., moving to any other column while keeping i fixed)
            p2_can_improve = any(bimatrix[i, k, 1] > p2_diff for k in range(size) if k != j)

            # If neither can improve, we found a Nash Equilibrium point
            if not p1_can_improve and not p2_can_improve:
                NE_points.append((i, j))

    return NE_points

def compare_NE_points(NE_points):
    # Organize the NE points by method and scenario for easier comparison
    data = {}

    # Group NE points by method and scenario
    for points, method, sc_name in NE_points:
        if method not in data:
            data[method] = {"P1": [], "P2": [], "Full": []}
        
        data[method][sc_name] = points

    # Print the NE points for each method and scenario to compare them
    for method, scenarios in data.items():
        p1_pts = scenarios["P1"]
        p2_pts = scenarios["P2"]
        full_pts = scenarios["Full"]
        
        print(f"{method.upper()} NE points: \n\t- P1: {p1_pts}, \n\t- P2: {p2_pts}, \n\t- Full: {full_pts}")

if __name__ == "__main__":
    mode = "half"  # Change this to "full", "half" or "quarter" if needed
    base_path = "results/" + mode + "/"
    # Loop through seeds and process the data to create heatmaps for both P1 and P2 accuracy drops
    for seed in range(0, 10):
        print("-" * 50)
        print(f"Processing seed {seed}...")
        games_path = f"{base_path}games/seed{seed}"
        if not os.path.exists(games_path):
            print(f"Missing games for seed {seed}, skipping...")
            continue

        # Load the bimatrix files for the current seed and print their shapes to verify they are loaded correctly
        games = load_games(games_path)

        # Search for Nash Equilibrium points in each bimatrix
        NE_points = []
        for bimatrix, method, sc_name in games:
            NE_points.append((search_NE(bimatrix, method, sc_name), method, sc_name))

        # Compare P1 and P2 Nesh Equilibrium points with Full Nesh Equilibrium points to see if they align or differ significantly
        compare_NE_points(NE_points)
        print("-" * 50)