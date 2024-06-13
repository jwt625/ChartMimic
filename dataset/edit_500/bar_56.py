# ===================
# Part 1: Importing Libraries
# ===================
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import numpy as np; np.random.seed(0)

# ===================
# Part 2: Data Preparation
# ===================
# Data
transport_modes = [
    "Scooter",
    "Walking",
    "Bicycle",
    "Train",
    "Bus",
    "Car",
]
number_of_passengers = [3000, 7000, 8000, 12000, 15000, 20000]
# Plot Configuration
xlabel = "Number of Passengers"
xlim_values = (0, 20000)
ylim_values = (-0.5, 5.5)
title_text = "Number of Passengers by Transport Mode"

basetick = [0, 1, 2, 3, 4, 5]
offsetticks = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

colors = [
    "lightskyblue",
    "turquoise",
    "lightgreen",
    "navajowhite",
    "lightsalmon",
    "lightcoral",
]
yticks_rotation = 45
# ===================
# Part 3: Plot Configuration and Rendering
# ===================
# Create horizontal bar chart
plt.figure(figsize=(12, 8))  # Adjust figure size to match original image's dimensions
plt.barh(
    transport_modes, number_of_passengers, color=colors, edgecolor="white"
)  # Change bar color to purple
plt.xlabel(xlabel)
plt.xlim(*xlim_values)
plt.ylim(*ylim_values)
plt.title(title_text)

plt.gca().yaxis.set_major_locator(ticker.FixedLocator(basetick))
plt.gca().yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f"{transport_modes[x-1]}")
)
plt.gca().yaxis.set_minor_locator(ticker.FixedLocator(offsetticks))
plt.gca().grid(True, which="minor", axis="y", color="gray", linestyle="--")
plt.gca().grid(True, which="major", axis="x", color="gray", linestyle="--")
plt.gca().set_axisbelow(True)
plt.tick_params(axis="both", which="major", length=0)
plt.tick_params(axis="y", which="minor", color="gray", length=3)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.yticks(rotation=yticks_rotation)

# ===================
# Part 4: Saving Output
# ===================
plt.tight_layout()
plt.savefig('bar_56.pdf', bbox_inches='tight')
