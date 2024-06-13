# ===================
# Part 1: Importing Libraries
# ===================
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)


# ===================
# Part 2: Data Preparation
# ===================
# Sample data to mimic the boxplot in the picture
data = [
    np.random.normal(0.825, 0.02, 100),
    np.random.normal(0.850, 0.03, 100),
    np.random.normal(0.840, 0.025, 100),
    np.random.normal(0.860, 0.015, 100),
    np.random.normal(0.855, 0.02, 100),
]

labels = ["SQL-Only", "PoT", "IC-LP", "DAIL", "IC-LP+PoT"]
ylabel = "Execution Accuracy"
ylim = [0.725, 0.925]
yticks = np.arange(0.750, 0.901, 0.025)


# ===================
# Part 3: Plot Configuration and Rendering
# ===================
# Create the boxplot
fig, ax = plt.subplots(
    figsize=(6, 5)
)  # Adjusting figure size as per the dimensions provided
bp = ax.boxplot(
    data,
    labels=labels,
    patch_artist=True,
    boxprops=dict(facecolor="#549e9a", color="black"),
    medianprops=dict(color="black"),
    whiskerprops=dict(color="black", linestyle="-"),
    capprops=dict(color="black", linestyle="-"),
)

# Remove outliers
for flier in bp["fliers"]:
    flier.set(marker="", color="black")

# Set the y-axis range and tick labels
ax.set_ylim(ylim)
ax.set_yticks(yticks)
# Set the y-axis label
ax.set_ylabel(ylabel, fontsize=12)

# Set the tick label size
ax.tick_params(axis="both", which="major", labelsize=10)
plt.xticks(rotation=45)

# ===================
# Part 4: Saving Output
# ===================
# Displaying the plot with tight layout to minimize white space
plt.tight_layout()
plt.savefig("box_13.pdf", bbox_inches="tight")
