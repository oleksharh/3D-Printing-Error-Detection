import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import numpy as np

head_names = ["Flow Rate", "Lat Speed", "Z-Offset", "Nozzle Temp"]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

# Stage 2 FULL DATASET NOT REDUCED
preds = (
    torch.load("test/final/preds_22-46_01-04-26.pt", weights_only=True).cpu().numpy()
)
targets = (
    torch.load("test/final/targets_22-46_01-04-26.pt", weights_only=True).cpu().numpy()
)



# REDUCED STAGE 1
# preds = (
#     torch.load("test/final/preds_22-35_01-04-26.pt", weights_only=True).cpu().numpy()
# )
# targets = (
#     torch.load("test/final/targets_22-35_01-04-26.pt", weights_only=True).cpu().numpy()
# )


# Stage 3 FULL DATASET NOT REDUCED BALANCED
preds = (
    torch.load("test/final/preds_23-08_01-04-26.pt", weights_only=True).cpu().numpy()
)
targets = (
    torch.load("test/final/targets_23-08_01-04-26.pt", weights_only=True).cpu().numpy()
)


for i in range(4):
    cm = confusion_matrix(targets[i], preds[i])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Low", "Good", "High"]
    )

    disp.plot(
        ax=axes[i],
        cmap="viridis",
        colorbar=False,
        values_format="d",
        text_kw={"color": "white"},
    )

    axes[i].set_title(head_names[i], fontsize=12, fontweight="bold")
    axes[i].set_xlabel("Predicted", fontsize=10)

    if i == 0:
        axes[i].set_ylabel("True Label", fontsize=10)
    else:
        axes[i].set_ylabel("")
        axes[i].set_yticklabels([])

    axes[i].grid(False)


cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
im_for_cbar = axes[0].images[0]
cbar = fig.colorbar(im_for_cbar, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Sample Count", fontsize=11)

plt.subplots_adjust(bottom=0.25, wspace=0.15)
plt.tight_layout()
plt.savefig(
    "C:/FYP/report_metrics/test_predicitons_stage3_full.svg",
    format="svg",
    bbox_inches="tight",
)
plt.show()
