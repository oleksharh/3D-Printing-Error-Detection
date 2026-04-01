import sys
import os
import torch
import pandas as pd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../clean_src"))
)

test_subset = torch.load(
    "C:/FYP/data/initial_layer_dataset_reduced/test_st1_reduced.pt"
)
validation_subset = torch.load(
    "C:/FYP/data/initial_layer_dataset_reduced/val_st1_reduced.pt"
)

full_reduced_df = pd.read_csv("C:/FYP/stages/reduced_datasets/full_dataset_reduced.csv")

test_subset_df = test_subset.dataset.dataframe.iloc[test_subset.indices]
validation_subset_df = validation_subset.dataset.dataframe.iloc[
    validation_subset.indices
]

seen_df = pd.concat([test_subset_df, validation_subset_df], ignore_index=True)

print(f"Seen dataset shape: {seen_df.shape}")


merged = full_reduced_df.merge(seen_df, how="left", indicator=True)

full_reduced_df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

print(f"Original shape: {len(merged)}")
print(f"Cleaned shape: {full_reduced_df.shape}")
print(full_reduced_df.reset_index(drop=True))

full_reduced_df.to_csv(
    "C:/FYP/stages/reduced_datasets/full_dataset_reduced_no_seen.csv", index=False
)
