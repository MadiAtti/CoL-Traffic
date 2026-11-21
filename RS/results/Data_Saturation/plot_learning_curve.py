import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for academic plotting
sns.set_theme(style="whitegrid", context="talk")

def generate_plots(csv_file=None):
    if csv_file is None:
        csv_file = Path(__file__).with_name("data_saturation_recommender_system.csv")

    # Load the data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return

    # Sort values by Data_Percentage ascending (5 -> 100)
    df = df.sort_values(by="Data_Percentage")
    # ==========================================
    # Plot: RMSE only
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(df['Data_Percentage'], df['Alone_RMSE'], color='tab:blue', marker='o', linewidth=2.5, label='RMSE')
    plt.xlabel('Dataset Size (%)', fontweight='bold')
    plt.ylabel('Global RMSE', fontweight='bold')
    plt.title('Model Capacity Saturation:\nRMSE over Dataset Size', pad=20, fontweight='bold')
    plt.ylim(0.68, 0.90)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_RMSE.png", dpi=300, bbox_inches='tight')
    print("Saved: plot_RMSE.png")
    plt.close()

if __name__ == "__main__":
    generate_plots()