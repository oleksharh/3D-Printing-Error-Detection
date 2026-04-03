import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_scalar_events(path):
    ea = EventAccumulator(path)
    ea.Reload()

    epochs = [i for i in range(len(ea.Scalars("val_acc")))]
    val_acc = [e.value for e in ea.Scalars("val_acc")]
    train_acc = [e.value for e in ea.Scalars("train_acc_epoch")]

    min_len = min(len(epochs), len(val_acc), len(train_acc))

    df = pd.DataFrame(
        {
            "epoch": epochs[:min_len],
            "val_acc": val_acc[:min_len],
            "train_acc": train_acc[:min_len],
        }
    )

    print(df)

    return df


run1_path = r"C:/FYP/logs/01042026-4-3482/version_4/"
run2_path = r"C:/FYP/logs/01042026-4-3482/version_5/"
run3_path = r"C:/FYP/logs/01042026-4-3482/version_6/"



df1 = extract_scalar_events(run1_path)
df2 = extract_scalar_events(run2_path)
df3 = extract_scalar_events(run3_path)



df1 = extract_scalar_events(run1_path)
df1 = df1[df1["epoch"] <= 35]
df2 = extract_scalar_events(run2_path)
df2 = df2[df2["epoch"] <= 5]
df3 = extract_scalar_events(run3_path)


offset1 = df1["epoch"].max() + 1
print(offset1)
df2["epoch"] = df2["epoch"] + offset1
offset2 = df2["epoch"].max() + 1
print(offset2)
df3["epoch"] = df3["epoch"] + offset2
offset3 = df3["epoch"].max() + 1
print(offset3)



dfs = [df1, df2, df3]
full_data = pd.concat(dfs, ignore_index=True)


cols_to_fix = ["epoch", "val_acc", "train_acc"]
for col in cols_to_fix:
    full_data[col] = pd.to_numeric(full_data[col], errors="coerce")


print(f"Cleaned dataframe rows: {len(full_data)}")
print(full_data)

plt.figure(figsize=(12, 6), dpi=150)
plt.plot(
    full_data["epoch"],
    full_data["train_acc"],
    label="Training Accuracy",
    color="#5900ff",
    linewidth=2,
)
plt.plot(
    full_data["epoch"],
    full_data["val_acc"],
    label="Validation Accuracy",
    color="#00ff00",
    linewidth=2,
)

import matplotlib.ticker as ticker

# ax = plt.gca()
# ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # Grid line every 1 epoch
# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))  # Grid line every 0.05 acc
# ax.xaxis.set_major_locator(ticker.MultipleLocator(2)) # Grid line every 2 epochs


plt.axvline(x=36, color="black", linestyle=":", alpha=0.7)
plt.text(
    27.5,
    0.65,
    "Restarted run from last ckpt, with LR reduced to 1e-4",
    fontsize=10,
    fontweight="bold",
    wrap=True,
)

plt.title("Stage1: Sub-Sampled Dataset Training", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy Score", fontsize=12)
plt.ylim(0.5, 1.0)
plt.grid(True, which="both", linestyle="-", alpha=0.2)
plt.legend(loc="upper right", frameon=True, shadow=True)
plt.tight_layout()
plt.savefig("report_metrics/stage1_reduced_accuracy_plot.svg", format="svg", bbox_inches="tight")
plt.show()
