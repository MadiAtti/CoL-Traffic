import json
from itertools import product
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_json(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    with open(path, 'r') as f:
        return json.load(f)

def get_accuracy_from_local(data, sub_path=""):

        eval_m1 = data['models']['M1']['evaluation']
        eval_m2 = data['models']['M2']['evaluation']

        if sub_path == "P1/":
            # P1 szcenárió: p11 és p12
            p1_acc = eval_m1['p11']['accuracy']
            p2_acc = eval_m2['p12']['accuracy']
        elif sub_path == "P2/":
            # P2 szcenárió: p21 és p22
            p1_acc = eval_m1['p21']['accuracy']
            p2_acc = eval_m2['p22']['accuracy']
        else:
            # Full FL szcenárió: sima p1 és p2
            p1_acc = eval_m1['p1']['accuracy']
            p2_acc = eval_m2['p2']['accuracy']
            
        return p1_acc, p2_acc

def process_and_plot(seed):
    # Ezt a listát nem szabad felülírni bent!
    folders = [("", "Full_FL"), ("P1/", "P1_Subnet"), ("P2/", "P2_Subnet")]
    methods = [("2_suppression/", "Suppression"), ("3_noise/", "Noise")]

    for method_path, method_name in methods:
        for sub_path, sc_name in folders:
            loc_path = f"1_local_baseline/{sub_path}{seed}.json"
            loc_data = load_json(loc_path)
            if not loc_data: continue
            
            b_p1, b_p2 = get_accuracy_from_local(loc_data, sub_path)
            print(f"Local Baseline ({sc_name}): P1 Accuracy = {b_p1:.4f}, P2 Accuracy = {b_p2:.4f}")
            
            fed_path = f"{method_path}{sub_path}{seed}.json"
            fed_data = load_json(fed_path)
            if not fed_data: continue
            
            config_data = fed_data['parameters']['config']
            if method_name == "Noise":
                params = config_data['noise_levels']
                k1, k2 = 'noise_p1', 'noise_p2'
            else:                
                params = config_data['sup_levels']
                k1, k2 = 'features_p1', 'features_p2'

            experiments = fed_data['experiments']
            size = len(params)
            
            m1_diff = np.zeros((size, size))
            m2_diff = np.zeros((size, size))

            # Feltöltés az experimentek alapján
            for exp in experiments:
                val1 = exp[k1]
                val2 = exp[k2]
                
                try:
                    idx1 = params.index(val1)
                    idx2 = params.index(val2)
                except ValueError:
                    continue
                
                # A federáltban nagybetűs P1/P2
                p1_acc = exp['final_evaluation']["P1"]['accuracy']
                p2_acc = exp['final_evaluation']["P2"]['accuracy']

                # Accuracy Drop (Local - Fed): a pozitív szám jelzi a romlást
                m1_diff[idx1][idx2] = p1_acc - b_p1
                m2_diff[idx1][idx2] = p2_acc - b_p2

            # --- VIZUALIZÁCIÓ ---
            # A None-t lecseréljük 0.0-ra a tengelyen
            display_params = [0.0 if x is None else x for x in params]
            
            for p_tag, matrix in [("P1", m1_diff), ("P2", m2_diff)]:
                plt.figure(figsize=(10, 8))
                # YlOrRd skála: a sötétebb piros jelzi a nagyobb teljesítményvesztést
                sns.heatmap(matrix, annot=True, fmt=".3f", cmap="RdYlGn", center=0, 
                            xticklabels=display_params, yticklabels=display_params)
                
                plt.title(f"{method_name} ({sc_name}) - {p_tag} Accuracy Drop (Seed {seed})")
                plt.xlabel("Client 2 Params")
                plt.ylabel("Client 1 Params")
                
                os.makedirs(f'plots/seed{seed}', exist_ok=True)
                plt.savefig(f"plots/seed{seed}/{method_name}_{sc_name}_{p_tag}.png")
                plt.close()
                print(f"Kész: plots/seed{seed}/{method_name}_{sc_name}_{p_tag}.png")

if __name__ == "__main__":
    process_and_plot(3)