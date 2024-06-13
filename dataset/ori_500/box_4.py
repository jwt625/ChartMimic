# ===================
# Part 1: Importing Libraries
# ===================
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)


# ===================
# Part 2: Data Preparation
# ===================
# Sample data
methods = ["REM (ours)", "IRIS", "STORM", "TWM", "DreamerV3", "SimPLe"]
mean_scores = [0.8, 0.7, 0.9, 0.85, 0.75, 0.4]
median_scores = [0.35, 0.25, 0.4, 0.3, 0.2, 0.15]
iqr_mean_scores = [0.5, 0.4, 0.55, 0.45, 0.35, 0.2]
optimality_gap_scores = [0.6, 0.65, 0.55, 0.6, 0.7, 0.75]


# Generate random data for box plots
data1 = [np.random.normal(mean, 0.1, 100) for mean in mean_scores]
data2 = [np.random.normal(median, 0.1, 100) for median in median_scores]
data3 = [np.random.normal(iqr, 0.1, 100) for iqr in iqr_mean_scores]
data4 = [np.random.normal(opt_gap, 0.1, 100) for opt_gap in optimality_gap_scores]

xlabel = "Human Normalized Score"
titles = ["Mean (↑)", "Median (↑)", "Interquartile Mean (↑)", "Optimality Gap (↓)"]
axhline = 2.5
# ===================
# Part 3: Plot Configuration and Rendering
# ===================
# Set figure size to match the original image's dimensions
plt.figure(figsize=(12, 3))
# Colors for each method
colors = ["lightcoral", "bisque", "lightblue", "gold", "thistle", "yellowgreen"]

# Create subplots
ax1 = plt.subplot(141)
ax2 = plt.subplot(142)
ax3 = plt.subplot(143)
ax4 = plt.subplot(144)

# Plotting
boxplot1 = ax1.boxplot(
    data1,
    vert=False,
    patch_artist=True,
    medianprops={"color": "black"},
    whiskerprops=dict(color="black", linestyle="-", linewidth=0),
    showcaps=False,
    showfliers=False,
    boxprops=dict(linestyle="-", linewidth=0),
)
boxplot2 = ax2.boxplot(
    data2,
    vert=False,
    patch_artist=True,
    medianprops={"color": "black"},
    whiskerprops=dict(color="black", linestyle="-", linewidth=0),
    showcaps=False,
    showfliers=False,
    boxprops=dict(linestyle="-", linewidth=0),
)
boxplot3 = ax3.boxplot(
    data3,
    vert=False,
    patch_artist=True,
    medianprops={"color": "black"},
    whiskerprops=dict(color="black", linestyle="-", linewidth=0),
    showcaps=False,
    showfliers=False,
    boxprops=dict(linestyle="-", linewidth=0),
)
boxplot4 = ax4.boxplot(
    data4,
    vert=False,
    patch_artist=True,
    medianprops={"color": "black"},
    whiskerprops=dict(color="black", linestyle="-", linewidth=0),
    showcaps=False,
    showfliers=False,
    boxprops=dict(linestyle="-", linewidth=0),
)

for bplot in [boxplot1, boxplot2, boxplot3, boxplot4]:
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
# Set labels and titles
ax1.set_yticklabels(methods)
ax1.set_xlabel(xlabel)
ax1.set_title(titles[0])
ax1.spines["top"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.xaxis.grid(True, alpha=0.7)
ax1.invert_yaxis()
ax1.axhline(axhline, color="black", linewidth=1)

ax2.set_yticks([])
ax2.set_xlabel(xlabel)
ax2.set_title(titles[1])
ax2.spines["top"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.xaxis.grid(True, alpha=0.7)
ax2.invert_yaxis()
ax2.axhline(axhline, color="black", linewidth=1)

ax3.set_yticks([])
ax3.set_xlabel(xlabel)
ax3.set_title(titles[2])
ax3.spines["top"].set_visible(False)
ax3.spines["left"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.xaxis.grid(True, alpha=0.7)
ax3.invert_yaxis()
ax3.axhline(axhline, color="black", linewidth=1)

ax4.set_yticks([])
ax4.set_xlabel(xlabel)
ax4.set_title(titles[3])
ax4.spines["top"].set_visible(False)
ax4.spines["left"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.xaxis.grid(True, alpha=0.7)
ax4.invert_yaxis()
ax4.axhline(axhline, color="black", linewidth=1)

# ===================
# Part 4: Saving Output
# ===================
# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("box_4.pdf", bbox_inches="tight")
