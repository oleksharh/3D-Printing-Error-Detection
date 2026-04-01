import pandas as pd
from math import inf
from itertools import product

TARGET_DIR = "C:/FYP/stages/stage3/"

df = pd.read_csv("C:/FYP/stages/stage2/full_dataset.csv")

# df = pd.read_csv("C:/FYP/stages/stage1/initial_layer_dataset.csv")

labels_ds = {
    "flow_rate_counts": [],
    "feed_rate_counts": [],
    "z_offset_counts": [],
    "hotend_counts": [],
}

for i in range(0, 3):
    labels_ds["flow_rate_counts"].append(int(df.flow_rate_class.value_counts()[i]))
    labels_ds["feed_rate_counts"].append(int(df.feed_rate_class.value_counts()[i]))
    labels_ds["z_offset_counts"].append(int(df.z_offset_class.value_counts()[i]))
    labels_ds["hotend_counts"].append(int(df.hotend_class.value_counts()[i]))


def get_lowest_count(counts):
    lowest_count = +inf
    key_validx_pair = None

    for key, val in labels_ds.items():
        if min(val) < lowest_count:
            lowest_count = min(val)
            key_validx_pair = (key, val.index(min(val)))

    return lowest_count, key_validx_pair


print(labels_ds)
print(get_lowest_count(labels_ds))

# Undersampling the dataset to balance the classes to 106647 each class
# or

# grouping key by 81 possible combo
df["combo"] = (
    df[["flow_rate_class", "feed_rate_class", "z_offset_class", "hotend_class"]]
    .astype(str)
    .agg("_".join, axis=1)
)

# finding the minimum count among the 81 combos
min_samples = df["combo"].value_counts().min()  # 429
print(f"Balancing all 81 classes to: {min_samples} samples each.")


combo_counts = df["combo"].value_counts().reset_index()
combo_counts.columns = ["combo", "count"]
combo_counts.to_csv("C:/FYP/report_metrics/" + "combo_counts.csv", index=False)

print(combo_counts)


# sampling 429 samples from each 81 combo possible
balanced_df = df.groupby("combo", group_keys=False).apply(
    lambda x: x.sample(n=min_samples, random_state=42)
)

print(balanced_df)

# Final count should be 81*429 = 34749
print(f"Final balanced dataset size: {len(balanced_df)}")

# balanced_df.to_csv(TARGET_DIR + "balanced_dataset.csv", index=False)
