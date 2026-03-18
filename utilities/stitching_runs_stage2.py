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


run1_path = r"C:\FYP\logs\13032026-1-1234\version_5"
run2_path = r"C:\FYP\logs\13032026-1-1234\version_7"
run3_path = r"C:\FYP\logs\13032026-1-1234\version_8"
run4_path = r"C:\FYP\logs\14032026-1-1234\version_0"
run5_path = r"C:\FYP\logs\14032026-1-1234\version_4"
run6_path = r"C:\FYP\logs\16032026-1-1234\version_0"
run7_path = r"C:\FYP\logs\16032026-1-1234\version_1"
run8_path = r"C:\FYP\logs\16032026-1-1234\version_2"


df1 = extract_scalar_events(run1_path)
df2 = extract_scalar_events(run2_path)
df3 = extract_scalar_events(run3_path)
df4 = extract_scalar_events(run4_path)
df5 = extract_scalar_events(run5_path)
df6 = extract_scalar_events(run6_path)
df7 = extract_scalar_events(run7_path)
df8 = extract_scalar_events(run8_path)


offset1 = df1["epoch"].max() + 1
print(offset1)
df2["epoch"] = df2["epoch"] + offset1
offset2 = df2["epoch"].max() + 1
print(offset2)
df3["epoch"] = df3["epoch"] + offset2
offset3 = df3["epoch"].max() + 1
print(offset3)
df4["epoch"] = df4["epoch"] + offset3
offset4 = df4["epoch"].max() + 1
print(offset4)
df5["epoch"] = df5["epoch"] + offset4
offset5 = df5["epoch"].max() + 1
print(offset5)
df6["epoch"] = df6["epoch"] + offset5
offset6 = df6["epoch"].max() + 1
print(offset6)
df7["epoch"] = df7["epoch"] + offset6
offset7 = df7["epoch"].max() + 1
print(offset7)
df8["epoch"] = df8["epoch"] + offset7
print(df8["epoch"].max())


dfs = [df1, df2, df3, df4, df5, df6, df7, df8]
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

ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # Grid line every 1 epoch
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))  # Grid line every 0.05 acc
ax.xaxis.set_major_locator(ticker.MultipleLocator(2)) # Grid line every 2 epochs


plt.axvline(x=27, color="black", linestyle=":", alpha=0.7)
plt.text(
    27.5,
    0.65,
    "LR Reduction\n(0.001 -> 0.0001)",
    fontsize=10,
    fontweight="bold",
    wrap=True,
)

plt.title("Stage2: Full Dataset Training", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy Score", fontsize=12)
plt.ylim(0.5, 1.0)
plt.grid(True, which="both", linestyle="-", alpha=0.2)
plt.legend(loc="upper right", frameon=True, shadow=True)
plt.tight_layout()
plt.savefig("report_metrics/stage2_accuracy_plot.png")
plt.show()
