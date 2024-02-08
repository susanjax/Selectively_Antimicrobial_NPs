import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# Set the font family to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

Mutation_data = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_MIC\output\Final_mutation.csv')
Population_data = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_MIC\output\Final_population.csv')

# Data for population
x = Population_data['generations']
y1 = Population_data['Avg_max_100']
y2 = Population_data['Avg_max_50']
y3 = Population_data['Avg_max_1-']
y4 = Population_data['Avg_mean_100']
y5 = Population_data['Avg_mean_50']
y6 = Population_data['Avg_mean_10']
y_values = [y1, y2, y3, y4, y5, y6]

# Define the logarithmic function to fit
def log_func(x, a, b, c):
    return a + b*np.log(x+c)

plt.figure(figsize=(8, 8))

colors = ['red', 'green', 'blue', 'red', 'green', 'blue']
labels1 = ['Max Pop. Size = 100', 'Max Pop. Size = 50', 'Max Pop. Size = 10']
labels2 = ['Mean Pop. Size = 100', 'Mean Pop. Size = 50', 'Mean Pop. Size = 10']
for i in range(len(labels1)):
    y = y_values[i]
    params, _ = curve_fit(log_func, x, y)
    sns.scatterplot(data=Population_data, x='generations', y=y, color=colors[i], alpha=0.3)
    plt.plot(x, log_func(x, *params), color=colors[i], label=labels1[i])

for i in range(len(labels2)):
    y = y_values[i+3]
    params, _ = curve_fit(log_func, x, y)
    sns.scatterplot(data=Population_data, x='generations', y=y, color=colors[i], alpha=0.3)
    plt.plot(x, log_func(x, *params), color=colors[i+3], label=labels2[i], linestyle='--')

# Set the title and axis labels
plt.title('Fitness Score of Different Populations', fontsize=16, color='black')
plt.xlabel('Generation Numbers', fontsize=14, color='black')
plt.ylabel('Fitness Score', fontsize=14, color='black')

# Add the legend to the plot
plt.legend(fontsize=12, loc='lower left')
plt.legend(fontsize=12, loc='lower right')

plt.savefig('population.png', transparent=True)
plt.show()

print(Mutation_data.columns)

plt.figure(figsize=(8, 8))

xm = Mutation_data['generations']
ym1 = Mutation_data['0.1% max']
ym2 = Mutation_data['1% max']
ym3 = Mutation_data['2% max']
ym4 = Mutation_data['5% max']
y_val = [ym1, ym2, ym3, ym4]

# Define the logarithmic function to fit
def log_func(xm, a, b, c):
    return a + b*np.log(xm+c)

colors1 = ['red', 'green', 'blue', 'orange']
labels = ['Mutation Rate = 0.1%', 'Mutation Rate = 1%', 'Mutation Rate = 2%', 'Mutation Rate = 5%']

for i in range(len(colors1)):
    ymm = y_val[i]
    params, _ = curve_fit(log_func, xm, ymm)
    sns.scatterplot(data=Mutation_data, x='generations', y=ymm, color=colors1[i], alpha=0.3)
    plt.plot(xm, log_func(xm, *params), color=colors1[i], label=labels[i])

# Set the title and axis labels
plt.title('Fitness Score of Different Populations', fontsize=16, color='black')
plt.xlabel('Generation Numbers', fontsize=14, color='black')
plt.ylabel('Fitness Score', fontsize=14, color='black')

plt.legend(loc='lower right', fontsize=12)
plt.savefig('mutation.png', transparent=True)
plt.show()
