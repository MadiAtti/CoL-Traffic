import pandas as pd

dataset_full = pd.read_parquet("dataset/full/p1.parquet")
dataset_half = pd.read_parquet("dataset/half/p1.parquet")
dataset_quarter = pd.read_parquet("dataset/quarter/p1.parquet")


print(dataset_full.shape[0])
print(dataset_half.shape[0])
print(dataset_quarter.shape[0])