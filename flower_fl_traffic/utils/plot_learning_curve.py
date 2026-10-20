import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for academic plotting
sns.set_theme(style="whitegrid", context="talk")

RESULTS_DIR = "results/baseline"


def generate_plots(csv_file="data_saturation_drug_discovery.csv",
                   output_path=os.path.join("results/final", "Data_Sat_NT.png")):
    # Load the data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return

    # Sort values by percentage ascending (5 -> 100)
    df = df.sort_values(by="percentage")
    # ==========================================
    # Plot: Accuracy only
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(df['percentage'], df['mean_accuracy'], color='tab:blue', marker='o', linewidth=2.5, label='Accuracy')
    plt.xlabel('Dataset Size (%)', fontweight='bold')
    plt.ylabel('Global Accuracy', fontweight='bold')
    plt.title('Model Capacity Saturation:\nAccuracy over Dataset Size', pad=20, fontweight='bold')
    plt.ylim(0.82, 0.95)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()



if __name__ == "__main__":
    generate_plots(os.path.join(RESULTS_DIR, "averaged_learning_curve.csv"))