# ===================
# Part 1: Importing Libraries
# ===================
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)


# ===================
# Part 2: Data Preparation
# ===================
# Sample data for the boxplots
data = {
    "BG": np.random.uniform(60, 95, 100),
    "IBP": np.random.uniform(70, 90, 100),
    "SSP": np.random.uniform(50, 85, 100),
    "NBP": np.random.uniform(40, 80, 100),
}

# Reverse the order of data for boxplots
data_values = list(data.values())[::-1]
data_keys = list(data.keys())[::-1]
xlabel = "Prediction Accuracy $\\nu_{D_1}$"
xlim = [0, 100]
xticks = np.arange(0, 101, 20)
title = "Methods"
# ===================
# Part 3: Plot Configuration and Rendering
# ===================
# Create a figure and axis with the specified size
fig, ax = plt.subplots(figsize=(9, 5))

# Create the boxplots with specific colors
boxprops = dict(linestyle="-", linewidth=2, color="black")
flierprops = dict(marker="o", color="black", markersize=5)
medianprops = dict(linestyle="-", linewidth=2, color="black")

bp = ax.boxplot(
    data_values,
    vert=False,
    patch_artist=True,
    boxprops=boxprops,
    flierprops=flierprops,
    medianprops=medianprops,
)

colors = ["#824920", "#377e22", "#0000f5", "#75147c"][::-1]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)

# Set the y-axis labels with reversed order
ax.set_yticklabels(data_keys)

# Set the x-axis label
ax.set_xlabel(xlabel)

# Set the x-axis limits and ticks
ax.set_xlim(xlim)
ax.set_xticks(xticks)
ax.set_xticklabels(["{}%".format(i) for i in xticks])

# Set the title of the plot
ax.set_title(title)

# ===================
# Part 4: Saving Output
# ===================
plt.tight_layout()
plt.savefig("box_21.pdf", bbox_inches="tight")
