import pandas as pd

TARGET_DIR = "C:/FYP/stages/reduced_datasets/"

# df = pd.read_csv("C:/FYP/stages/stage2/full_dataset.csv")

df = pd.read_csv("C:/FYP/stages/stage1/initial_layer_dataset.csv")

class_labels = ["flow_rate_class", "feed_rate_class", "z_offset_class", "hotend_class"]

change_mask = (df[class_labels].shift() != df[class_labels]).any(axis=1)
df['change'] = change_mask.cumsum()

print(df.head(190))

def filter_redundancy(change_group):
    
    indices = list(range(0, len(change_group), 10))

    if len(change_group)-1 not in indices:
        indices.append(len(change_group)-1)

    return change_group.iloc[indices]

reduced_df = df.groupby('change').apply(filter_redundancy).reset_index(drop=True)

print(reduced_df.shape)

# reduced_df.to_csv(TARGET_DIR + "full_dataset_reduced.csv", index=False)

reduced_df.to_csv(TARGET_DIR + "initial_layer_dataset_reduced.csv", index=False)