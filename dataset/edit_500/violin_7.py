import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)

from matplotlib.lines import Line2D

# ===================
# Part 2: Data Preparation
# ===================
# Sample data for demonstration purposes
features = np.arange(1, 6)
financial_growth_data = np.random.rand(20, 100) * 450
economic_stability_data = np.random.rand(20, 100) * 150
titles=["% Increase in ROI vs. Baseline A","% Increase in ROI vs. Baseline B"]
ylims=[[0, 500],[0, 500]]
xlabel="Number of Features"
ytickslabels=[["0%", "100%", "200%", "300%", "400%", "500%", "600%", "700%"],["0%", "100%", "200%", "300%", "400%", "500%", "600%", "700%"]]
xticklabel=[4, 8, 12, 16, 20]
# ===================
# Part 3: Plot Configuration and Rendering
# ===================
# Create figure and subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
# Define the colors for the violin plots
color_group_l0 = "#4f86b8"  # Color for Group ℓ0
color_agis = "#ef8636"  # Color for AGIS

# Function to set the color of the violin plot
def set_violin_color(violin, color):
    for body in violin["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
    # Set color for the median line
    violin["cmedians"].set_color(color)


# Top subplot: FLAM-GL
for i, feature in enumerate(features):
    vl = axs[0].violinplot(
        financial_growth_data[i],
        positions=[feature - 0.2],
        showmedians=True,
        widths=0.3,
        showextrema=False,
    )
    set_violin_color(vl, color_group_l0)

    vl = axs[0].violinplot(
        economic_stability_data[i],
        positions=[feature + 0.2],
        showmedians=True,
        widths=0.3,
        showextrema=False,
    )
    set_violin_color(vl, color_agis)

    axs[0].text(
        feature - 0.35,
        np.median(financial_growth_data[i]),
        f"{int(np.median(financial_growth_data[i]))}%",
        ha="right",
        va="bottom",
        color=color_group_l0,
    )
    axs[0].text(
        feature + 0.35,
        np.median(economic_stability_data[i]),
        f"{int(np.median(economic_stability_data[i]))}%",
        ha="left",
        va="bottom",
        color=color_agis,
    )

axs[0].set_title(titles[0])
axs[0].set_xticks(features)
axs[0].set_ylim(ylims[0])
# remove x-axis label for the top subplot
axs[0].set_xticklabels([])
axs[0].set_yticklabels(ytickslabels[0])

# Bottom subplot: EBM-RS
for i, feature in enumerate(features):
    vl = axs[1].violinplot(
        financial_growth_data[i],
        positions=[feature - 0.2],
        showmedians=True,
        widths=0.3,
        showextrema=False,
    )
    set_violin_color(vl, color_group_l0)

    vl = axs[1].violinplot(
        economic_stability_data[i],
        positions=[feature + 0.2],
        showmedians=True,
        widths=0.3,
        showextrema=False,
    )
    set_violin_color(vl, color_agis)

    axs[1].text(
        feature - 0.35,
        np.median(financial_growth_data[i]),
        f"{int(np.median(financial_growth_data[i]))}%",
        ha="right",
        va="bottom",
        color=color_group_l0,
    )
    axs[1].text(
        feature + 0.35,
        np.median(economic_stability_data[i]),
        f"{int(np.median(economic_stability_data[i]))}%",
        ha="left",
        va="bottom",
        color=color_agis,
    )

axs[1].set_title(titles[1])
axs[1].set_xticks(features)
axs[1].set_ylim(ylims[1])
axs[1].set_xlabel(xlabel)
axs[1].set_xticklabels(xticklabel)
axs[1].set_yticklabels(ytickslabels[1])

# use line to create a custom legend
legend_elements = [
    Line2D([0], [0], color=color_group_l0, lw=2, label="Group ℓ${_0}$"),
    Line2D([0], [0], color=color_agis, lw=2, label="AGIS"),
]
axs[0].legend(handles=legend_elements, loc="upper right")
axs[1].legend(handles=legend_elements, loc="upper right")

# ===================
# Part 4: Saving Output
# ===================
# Adjust layout and save plot
plt.tight_layout()
plt.savefig('violin_7.pdf', bbox_inches='tight')
