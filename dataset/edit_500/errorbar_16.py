import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)


# ===================
# Part 2: Data Preparation
# ===================
# Sample data (replace with actual values)
groups = [
    "Small Biz / 2022 / Q1",
    "Small Biz / 2022 / Q2",
    "Small Biz / 2023 / Q1",
    "Small Biz / 2023 / Q2",
    "Tech Startup / 2022 / Q1",
    "Tech Startup / 2022 / Q2",
    "Tech Startup / 2023 / Q1",
    "Tech Startup / 2023 / Q2",
    "Enterprise / 2022 / Q1",
    "Enterprise / 2022 / Q2",
    "Enterprise / 2023 / Q1",
    "Enterprise / 2023 / Q2",
    "Retail / 2022 / Q1",
    "Retail / 2022 / Q2"
]
solid_bar_values = np.random.rand(14) * 0.5
striped_bar_values = np.random.rand(14) * 0.5
error = np.random.rand(14) * 0.1 + 0.02
labels = ["Solid", "Striped"]
xlabel = "Growth Metric"
ylabel = "Business Categories and Periods"
title = "Business Growth Analysis"
xlim = [0, 1.1]
ylim = [-0.4, 13.4]

# ===================
# Part 3: Plot Configuration and Rendering
# ===================
# Set figure size to match the original image's dimensions
plt.figure(figsize=(8, 8))

# Create grouped bar chart with error bars
bar_width = 0.8
index = np.arange(len(groups))
plt.barh(
    index,
    solid_bar_values,
    bar_width,
    color="#bdbad7",
    xerr=error,
    label=labels[0],
    capsize=3,
    edgecolor="black",
)
plt.barh(
    index,
    striped_bar_values,
    bar_width,
    left=solid_bar_values,
    color="#f5cfe4",
    xerr=error,
    label=labels[1],
    capsize=3,
    edgecolor="black",
)

# Add labels and title
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.yticks(index, groups, rotation=0)
plt.gca().set_xlim(xlim)
plt.gca().set_ylim(ylim)

# ===================
# Part 4: Saving Output
# ===================
# Show plot with tight layout to minimize white space
plt.tight_layout()
plt.savefig('errorbar_16.pdf', bbox_inches='tight')
