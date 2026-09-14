import pandas as pd


dataset = pd.read_parquet("dataset/dataset.parquet")
print(dataset.shape)
# print(dataset.head(1).to_dict())

classes = dataset["application_name"].unique()
print(classes)