# combo,count
# 2_2_2_1,47656
# 0_2_2_1,42769

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product

# combo_counts = pd.read_csv("C:/FYP/report_metrics/combo_counts.csv")

combo_counts = pd.read_csv("C:/FYP/report_metrics/reduced_combo_counts.csv")

labels_axis = [
    f"{''.join(map(str, part_combo))}" for part_combo in product(range(3), repeat=2)
]

combo_counts["combo"] = combo_counts["combo"].apply(lambda x: "".join(x.split("_")))

matrix = np.zeros((9, 9))


for idx, row in combo_counts.iterrows():
    combo = row["combo"]
    x_combo, y_combo = combo[:2], combo[2:]
    count = int(row["count"])
    x = labels_axis.index(x_combo)
    y = labels_axis.index(y_combo)
    matrix[x, y] = count

# print(combo_counts)
# print(matrix)
plt.figure(figsize=(10, 8))

plt.imshow(matrix, cmap="viridis", interpolation="nearest")
cbar = plt.colorbar(label="Count")


for i in range(len(labels_axis)):
    for j in range(len(labels_axis)):
        plt.text(
            j,
            i,
            f"{int(matrix[i, j])}",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )

plt.xticks(range(len(labels_axis)), labels_axis)
plt.yticks(range(len(labels_axis)), labels_axis)

plt.xlabel("Combo")
plt.ylabel("Combo")
plt.title("Combo Counts Heatmap")

plt.title(
    "Heatmap of 81 Classes \nRows: First 2 digits | Columns: Last 2 digits",
    fontsize=14,
    fontweight="bold",
)
plt.ylabel("First Two Digits (x0, x1)", fontsize=12)
plt.xlabel("Last Two Digits (x2, x3)", fontsize=12)
plt.tight_layout()
# plt.savefig(
#     "C:/FYP/report_metrics/combo_heatmap.svg", format="svg", bbox_inches="tight"
# )

plt.savefig(
    "C:/FYP/report_metrics/reduced_combo_heatmap.svg", format="svg", bbox_inches="tight"
)
plt.show()
