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


run1_path = r"C:\FYP\logs\logs-11032026\1234\lightning_logs\version_2"

df1 = extract_scalar_events(run1_path)

full_data = df1


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


plt.axvline(x=31, color="black", linestyle=":", alpha=0.7)
plt.text(
    27.5,
    0.65,
    "LR Reduction\n(0.001 -> 0.0001)",
    fontsize=10,
    fontweight="bold",
    wrap=True,
)


plt.title("Stage1: Initial Layer Dataset", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy Score", fontsize=12)
plt.ylim(0.5, 1.0)
plt.grid(True, which="both", linestyle="-", alpha=0.2)
plt.legend(loc="lower right", frameon=True, shadow=True)
plt.tight_layout()
plt.savefig("report_metrics/stage1_accuracy_plot.png")
plt.show()
