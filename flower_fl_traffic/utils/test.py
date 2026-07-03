import pandas as pd

dataset = pd.read_parquet("dataset/dataset.parquet")
# dataset_full = pd.read_parquet("dataset/full/p1.parquet")
# dataset_half = pd.read_parquet("dataset/half/p1.parquet")
# dataset_quarter = pd.read_parquet("dataset/quarter/p1.parquet")

print(dataset.shape[0])
# print(dataset_full.shape[0])
# print(dataset_half.shape[0])
# print(dataset_quarter.shape[0])

# Nézd meg a statisztikákat a különböző split-ekre
# print("Full Dataset statisztikák:\n", dataset_full.describe())

# # Adatsűrűség kiszámítása (nem nulla értékek aránya)
# density = dataset_full.astype(bool).sum().sum() / dataset_full.size
# print(f"Dataset sűrűsége: {density:.4f}")

# # Minimum és maximum értékek (látni fogod a -3 és +3 közötti tartományt)
# print("Min értékek:\n", dataset_full.min())
# print("Max értékek:\n", dataset_full.max())

# az összes label kilistázása
print("Unique labels in the dataset:", dataset['application_name'].unique())
