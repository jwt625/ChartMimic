import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)


# ===================
# Part 2: Data Preparation
# ===================
# Placeholder data
categories = [
    "urban high",
    "urban low",
    "urban high",
    "urban low",
    "rural high",
    "rural low",
    "rural high",
    "rural low",
    "suburban high",
    "suburban low",
    "suburban high",
    "suburban low",
    "coastal high",
    "coastal low",
    "coastal high",
    "coastal low",
    "mountain",
    "mountain",
]

gender = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1]

title = "Geographical Distribution"
ylabel = "Region Type"
xlabel = "Population Density"

# Extract the first group of characters from each category
first_chars = [category.split(" ")[0] for category in categories]
# Find all unique first characters
unique_chars = list(set(first_chars))
# Create a margin for each unique first character
margins = {char: i for i, char in enumerate(unique_chars)}
data = [np.random.uniform(low=2, high=15, size=100) for _ in range(len(categories))]

# ===================
# Part 3: Plot Configuration and Rendering
# ===================
fig, ax = plt.subplots(figsize=(10, 8))
color = ["white", "gray"]
# Plot each category with a margin
positions = []
for i, (category, datum, g) in enumerate(zip(categories, data, gender)):

    position = i + margins[category.split(" ")[0]]
    if position in positions:
        position = position + 4
    positions.append(position)
    # print(i,position)
    bp = ax.boxplot(
        datum, positions=[position], vert=False, patch_artist=True, widths=0.6
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color[g])

ax.set_title(title)
ax.set_ylabel(ylabel)
ax.set_xlabel(xlabel)

plt.yticks(positions, labels=categories)

# Adding a vertical line for reference
ax.axvline(x=5, color="gray", linestyle="-", linewidth=1)

# ===================
# Part 4: Saving Output
# ===================
# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('box_1.pdf', bbox_inches='tight')
