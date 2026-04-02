import matplotlib.pyplot as plt

# (flow rate = 0.1348, lateral speed = 0.1273, z-offset = 0.1188, nozzle temperature = 0.2407)
# Stage I: Initial Layer Dataset
fl_rate = 0.1348
lat_speed = 0.1273
z_offset = 0.1188
nozzle_temp = 0.2407

# Stage II: Full Dataset
fl_rate = 0.4454
lat_speed = 0.4564
z_offset = 0.4318
nozzle_temp = 0.5554

# STAGE I REDUCED DATASET
fl_rate = 0.4000
lat_speed = 0.5787
z_offset = 0.4784
nozzle_temp = 0.5926

# STAGE III FULL DATASET NOT REDUCED BALANCED
fl_rate = 0.2748
lat_speed = 0.2696
z_offset = 0.2907
nozzle_temp = 0.371


labels = ["Flow Rate", "Lateral Speed", "Z-Offset", "Nozzle Temperature"]
values = [fl_rate, lat_speed, z_offset, nozzle_temp]
plt.figure(figsize=(8, 6))
colors = plt.cm.viridis([(v - min(values)) / (max(values) - min(values)) for v in values])
plt.bar(labels, values, color=colors)
plt.title("Validation Loss by Parameter", fontsize=14, fontweight="bold")
plt.ylabel("Loss Value", fontsize=12)
plt.ylim(0, max(values) + 0.05)
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=10)
plt.tight_layout()
# plt.savefig(
#     "C:/FYP/report_metrics/val_loss_by_param_stage1.svg", format="svg", bbox_inches="tight"
# )
plt.savefig(
    "C:/FYP/report_metrics/val_loss_by_param_stage3_full.svg", format="svg", bbox_inches="tight"
)
plt.show()