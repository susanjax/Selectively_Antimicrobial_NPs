import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# ... (your data loading code remains unchanged) ...

# Set a more professional color palette
colors = sns.color_palette("husl", 6)

# Adjust the figure size
plt.figure(figsize=(12, 8))

# Loop over each data set and fit the logarithmic curve
for i in range(len(labels1)):
    y = y_values[i]
    params, _ = curve_fit(log_func, x, y)
    sns.scatterplot(data=Population_data, x='generations', y=y, color=colors[i], alpha=0.3, label=labels1[i])
    plt.plot(x, log_func(x, *params), color=colors[i])

for i in range(len(labels2)):
    y = y_values[i+3]
    params, _ = curve_fit(log_func, x, y)
    sns.scatterplot(data=Population_data, x='generations', y=y, color=colors[i], alpha=0.3, label=labels2[i])
    plt.plot(x, log_func(x, *params), color=colors[i+3], linestyle='--')

# Set the title and axis labels
plt.title('Population Fitness Trends Over Generations', fontsize=16)
plt.xlabel('Generation Numbers', fontsize=14)
plt.ylabel('Fitness Score', fontsize=14)

# Add a legend with improved placement
plt.legend(fontsize=12, loc='upper left')

# Save the plot with higher resolution
plt.savefig('population.png', dpi=300, transparent=True)

# Show the plot
plt.show()
