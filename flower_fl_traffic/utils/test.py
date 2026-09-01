import matplotlib.pyplot as plt

for method in ["Noise", "Suppression"]:
    real = matrices["full"][method]["real"].ravel()
    pred = matrices["full"][method]["pred"].ravel()

    corr = np.corrcoef(pred, real)[0, 1]
    print(f"\n{method}")
    print(f"  korreláció: {corr:.4f}")
    print(f"  pred negatív: {(pred < 0).mean():.1%}")
    print(f"  real negatív: {(real < 0).mean():.1%}")

    plt.figure(figsize=(6, 6))
    plt.scatter(pred, real, alpha=0.5, s=20)
    plt.axline((0, 0), slope=1, color='r', linestyle='--', label='y=x')
    plt.xlabel("pred")
    plt.ylabel("real")
    plt.title(f"{method} – pred vs real")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"scatter_{method}.png")
    plt.close()
    print(f"  scatter mentve: scatter_{method}.png")