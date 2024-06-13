import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)


# ===================
# Part 2: Data Preparation
# ===================
data = {
    "Mean Voltage": np.random.rand(6, 6) * 0.5 + 3,
    "Median Voltage": np.random.rand(6, 6) * 0.5 + 2.5,
    "Voltage Variance": np.random.rand(6, 6) * 0.5 + 1.5,
    "Voltage Deviation": np.random.rand(6, 6) * 0.5 + 1,
}
colors = ["#d1af8e", "#d09dca", "#d48b4f", "#65b598", "#5894c2", "#deae57"]
colors = plt.get_cmap("Set3").colors
labels = [
    "Standard Operation",
    "Low Load",
    "High Load",
    "Maintenance",
    "Power Saving",
    "Overload",
]  # Updated scenario labels in English
# ===================
# Part 3: Plot Configuration and Rendering
# ===================
fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)

for i, (ax, (metric, values)) in enumerate(zip(axes.flatten(), data.items())):
    bplot = ax.boxplot(
        values,
        vert=True,
        patch_artist=True,
        showcaps=False,
        showfliers=False,
        whiskerprops=dict(color="#4f4f4f", linestyle="-", linewidth=0),
        medianprops={"color": "#4f4f4f"},
        boxprops=dict(linestyle="-", linewidth=0),
    )
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_title(metric)
    if i == 2 or i == 3:
        ax.set_xticklabels(labels, rotation=45)
    else:
        ax.set_xticks([])
    ax.yaxis.grid(True, alpha=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

# ===================
# Part 4: Saving Output
# ===================
plt.tight_layout()
plt.savefig('box_24.pdf', bbox_inches='tight')
