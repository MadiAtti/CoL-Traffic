import itertools
import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

import os
import numpy as np
from omegaconf import OmegaConf

def load_games(games_path):
    results = {}
    
    # Paraméterek, amik mentén végigiterálunk
    methods = ["Noise", "Suppression"]
    scenarios = [("Full_FL", "Real"), ("subnet", "Simulated")]
    players = ["P1", "P2"]

    for method in methods:
        if method not in results:
            results[method] = {}
        
        for sc_name, sc_type in scenarios:
            # A metóduson belül elkülönítjük a Real és Simulated eseteket
            if sc_type not in results[method]:
                results[method][sc_type] = {}

            for player in players:
                if player not in results[method][sc_type]:
                    results[method][sc_type][player] = {}

                bimatrix_path = f"{games_path}/{method}_{player}_{sc_name}_bimatrix.npy"
                
                if os.path.exists(bimatrix_path):
                    bimatrix = np.load(bimatrix_path)
                    results[method][sc_type][player] = bimatrix
                else:
                    print(f"Warning: Bimatrix file not found for {method} ({sc_name}, {player}) at {bimatrix_path}")
    return results

def get_utility_matrix(weight, matrix, method, player):
    real_matrix = matrix[method]['Real'][player]
    simulated_matrix = matrix[method]['Simulated'][player]

    utility_matricies = []

    cfg = OmegaConf.load("conf/base.yaml") 
    params = []

    if method == "Noise":
        params = cfg.config.noise_levels
    else:  # method == "Suppression"
        params = cfg.config.sup_levels

    p1_grid, p2_grid = np.meshgrid(params, params, indexing='ij')

    if player == "P1":
        x = p1_grid
    else:  # player == "P2"
        x = p2_grid


    if method == "Noise":
        utility_matricies.append({"real": real_matrix - weight * (1-x)})
        utility_matricies.append({"simulated": simulated_matrix - weight * (1-x)})
    else:      # method == "Suppression"
        utility_matricies.append({"real": real_matrix - weight * x/14})
        utility_matricies.append({"simulated": simulated_matrix - weight * x/14})

    return utility_matricies

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

def plot_heatmap(matrix, title, path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="RdYlGn", center=0)

    plt.title(title)            
    plt.savefig(path)
    plt.close()

if __name__ == "__main__":
    mode = "half"  # Change this to "full", "half" or "quarter" if needed
    games_path = "results/" + mode + "/games/average"

    path = f"results/{mode}/utilities"
    os.makedirs(path, exist_ok=True)

    players = [
        {"name": "Defender", "weight": 1.0},
        {"name": "Strategist1", "weight": 0.2},
        {"name": "Strategist2", "weight": 0.3},
        {"name": "Strategist3", "weight": 0.5},
        {"name": "Analyst", "weight": 0.0}
    ]
    
    games = list(itertools.product(players, repeat=2))

    bimatrix_results = load_games(games_path)

    for method in ['Noise', 'Suppression']: # Iterate over methods
        for player in players: # Iterate over player types
            os.makedirs(path + f"/{method}", exist_ok=True)

            print (f"\nProcessing utility matrices for {method} - {player['name']}, {player['weight']}...")
            # get the utility matrix for this game
            p1_utility_matricies = get_utility_matrix(player['weight'], bimatrix_results, method, 'P1')
            p2_utility_matricies = get_utility_matrix(player['weight'], bimatrix_results, method, 'P2')

            # save the utility matrices to path/method/P1_playername.npy and path/method/P2_playername.npy
            np.save(f"{path}/{method}/P1_{player['name']}_real.npy", p1_utility_matricies[0])
            np.save(f"{path}/{method}/P1_{player['name']}_simulated.npy", p1_utility_matricies[0])

            np.save(f"{path}/{method}/P2_{player['name']}_real.npy", p2_utility_matricies[1])
            np.save(f"{path}/{method}/P2_{player['name']}_simulated.npy", p2_utility_matricies[1])

            plot_heatmap(p1_utility_matricies[0]['real'], f"{method} - P1 {player['name']} Real Utility", f"{path}/{method}/P1_{player['name']}_real.png")
            plot_heatmap(p1_utility_matricies[1]['simulated'], f"{method} - P1 {player['name']} Simulated Utility", f"{path}/{method}/P1_{player['name']}_simulated.png")
            plot_heatmap(p2_utility_matricies[0]['real'], f"{method} - P2 {player['name']} Real Utility", f"{path}/{method}/P2_{player['name']}_real.png")
            plot_heatmap(p2_utility_matricies[1]['simulated'], f"{method} - P2 {player['name']} Simulated Utility", f"{path}/{method}/P2_{player['name']}_simulated.png")
            
        # # calculate the difference between real and simulated utility matrices for both players
        # p1_diff_matrix = p1_utility_matricies[0]['real'] - p1_utility_matricies[1]['simulated']
        # p2_diff_matrix = p2_utility_matricies[0]['real'] - p2_utility_matricies[1]['simulated']

        # np.save(f"{path}/{method}/P1_diff.npy", p1_diff_matrix)
        # np.save(f"{path}/{method}/P2_diff.npy", p2_diff_matrix)

        # plot_heatmap(p1_diff_matrix, f"{method} - P1 Real vs Simulated Utility Difference", f"{path}/{method}/P1_diff.png")
        # plot_heatmap(p2_diff_matrix, f"{method} - P2 Real vs Simulated Utility Difference", f"{path}/{method}/P2_diff.png")

