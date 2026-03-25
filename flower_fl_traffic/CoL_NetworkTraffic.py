import argparse
import random
import json
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import local_params
from federated.experiment_runner import run_dp

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Secure RNG turned off.*")
warnings.filterwarnings("ignore", message="Using a non-full backward hook.*")

def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Parameters:
        seed (int): The seed value to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the training script with a specified seed.")
    parser.add_argument("--seed", type=int, default=42, help="Seed value for reproducibility (default: 42)")
    args = parser.parse_args()

    set_seed(args.seed)
    (X_p1_train, X_p1_test, y_p1_train, y_p1_test,
    X_p2_train, X_p2_test, y_p2_train, y_p2_test,
    X_p11_train, X_p11_test, y_p11_train, y_p11_test,
    X_p12_train, X_p12_test, y_p12_train, y_p12_test,
    X_p21_train, X_p21_test, y_p21_train, y_p21_test,
    X_p22_train, X_p22_test, y_p22_train, y_p22_test) = DataLoader.prepare_data(args.seed, local_params)
    
    #print("Baseline ...")
    #set_seed(args.seed)
    #loc_res = run_local(X_p1_train, X_p1_test, y_p1_train, y_p1_test, X_p2_train, X_p2_test, y_p2_train, y_p2_test)
    #with open('1_local_baseline/' + str(args.seed) + '.json', 'w') as f:
    #    json.dump(loc_res, f, indent=4)

    #print("Federated ...")
    #set_seed(args.seed)
    #fl_res = run_fl(X_p1_train, X_p1_test, y_p1_train, y_p1_test, X_p2_train, X_p2_test, y_p2_train, y_p2_test)
    #with open('2_fl_baseline/' + str(args.seed) + '.json', 'w') as f:
    #    json.dump(fl_res, f, indent=4)

    #print("Suppression ...")
    #set_seed(args.seed)
    #sup_res = run_sup(X_p1_train, X_p1_test, y_p1_train, y_p1_test, X_p2_train, X_p2_test, y_p2_train, y_p2_test)
    #with open('3_suppression/' + str(args.seed) + '.json', 'w') as f:
    #    json.dump(sup_res, f, indent=4)

    print("Noise ...")
    set_seed(args.seed)
    dp_res = run_dp(X_p1_train, X_p1_test, y_p1_train, y_p1_test, X_p2_train, X_p2_test, y_p2_train, y_p2_test)
    with open('4_noise/' + str(args.seed) + '.json', 'w') as f:
        json.dump(dp_res, f, indent=4)

    #print("P1 Simulated Local ...")
    #set_seed(args.seed)
    #loc_res = run_local(X_p11_train, X_p11_test, y_p11_train, y_p11_test, X_p12_train, X_p12_test, y_p12_train, y_p12_test)
    #with open('1_local_baseline/P1/' + str(args.seed) + '.json', 'w') as f:
    #    json.dump(loc_res, f, indent=4)

    #print("P1 Simulated Federated ...")
    #set_seed(args.seed)
    #fl_res = run_fl(X_p11_train, X_p11_test, y_p11_train, y_p11_test, X_p12_train, X_p12_test, y_p12_train, y_p12_test)
    #with open('2_fl_baseline/P1/' + str(args.seed) + '.json', 'w') as f:
    #    json.dump(fl_res, f, indent=4)

    #print("P1 simulated Suppression ...")
    #set_seed(args.seed)
    #sup_res = run_sup(X_p11_train, X_p11_test, y_p11_train, y_p11_test, X_p12_train, X_p12_test, y_p12_train, y_p12_test)
    #with open('3_suppression/P1/' + str(args.seed) + '.json', 'w') as f:
    #    json.dump(sup_res, f, indent=4)

    print("P1 Simulated Noise ...")
    set_seed(args.seed)
    dp_res = run_dp(X_p11_train, X_p11_test, y_p11_train, y_p11_test, X_p12_train, X_p12_test, y_p12_train, y_p12_test)
    with open('4_noise/P1/' + str(args.seed) + '.json', 'w') as f:
        json.dump(dp_res, f, indent=4)

    #print("P2 Simulated Local ...")
    #set_seed(args.seed)
    #loc_res = run_local(X_p21_train, X_p21_test, y_p21_train, y_p21_test, X_p22_train, X_p22_test, y_p22_train, y_p22_test)
    #with open('1_local_baseline/P2/' + str(args.seed) + '.json', 'w') as f:
    #    json.dump(loc_res, f, indent=4)

    #print("P2 Simulated Federated ...")
    #set_seed(args.seed)
    #fl_res = run_fl(X_p21_train, X_p21_test, y_p21_train, y_p21_test, X_p22_train, X_p22_test, y_p22_train, y_p22_test)
    #with open('2_fl_baseline/P2/' + str(args.seed) + '.json', 'w') as f:
    #    json.dump(fl_res, f, indent=4)

    #print("P2 Simulated Suppression ...")
    #set_seed(args.seed)
    #sup_res = run_sup(X_p21_train, X_p21_test, y_p21_train, y_p21_test, X_p22_train, X_p22_test, y_p22_train, y_p22_test)
    #with open('3_suppression/P2/' + str(args.seed) + '.json', 'w') as f:
    #    json.dump(sup_res, f, indent=4)

    print("P2 simulated Noise ...")
    set_seed(args.seed)
    dp_res = run_dp(X_p21_train, X_p21_test, y_p21_train, y_p21_test, X_p22_train, X_p22_test, y_p22_train, y_p22_test)
    with open('4_noise/P2/' + str(args.seed) + '.json', 'w') as f:
        json.dump(dp_res, f, indent=4)

if __name__ == "__main__":
    main()
