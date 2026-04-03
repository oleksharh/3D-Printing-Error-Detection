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


run1_path = r"C:/FYP/logs/01042026-5-555/version_1/"
run2_path = r"C:/FYP/logs/01042026-5-555/version_2/"


df1 = extract_scalar_events(run1_path)
df2 = extract_scalar_events(run2_path)


plt.figure(figsize=(12, 6), dpi=150)


plt.plot(
    df1["epoch"],
    df1["train_acc"],
    label="Training Accuracy (Run 1)",
    color="#5900ff",
    linewidth=2,
)
plt.plot(
    df1["epoch"],
    df1["val_acc"],
    label="Validation Accuracy (Run 1)",
    color="#00ff00",
    linewidth=2,
)


plt.plot(
    df2["epoch"],
    df2["train_acc"],
    label="Training Accuracy (Run 2)",
    color="#5900ff",
    linewidth=2,
    linestyle="--",
)
plt.plot(
    df2["epoch"],
    df2["val_acc"],
    label="Validation Accuracy (Run 2)",
    color="#00ff00",
    linewidth=2,
    linestyle="--",
)

plt.title("Stage2: Sub-Sampled Dataset Training", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy Score", fontsize=12)
plt.ylim(0.47, 0.6)
plt.grid(True, which="both", linestyle="-", alpha=0.2)
plt.legend(loc="upper right", frameon=True, shadow=True)
plt.tight_layout()
# plt.savefig("report_metrics/stage2_subsampled_accuracy_plot.svg", format="svg", bbox_inches="tight")
plt.show()
