import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

ZOI_data_raw = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_ZOI\data\raw\final_merged_df.csv')
# print(ZOI_data_raw.columns)

# Calculate the percentage of missing values in each column
# missing_percentage = ZOI_data_raw.isnull().mean() * 100
# missing_percentage.to_csv('missing_info_raw.csv', header=True) # subkindom has 92% and clade has 57% missing values
# remove these columns

ZOI_1 = ZOI_data_raw.drop(['combination','mdr', 'strain', 'id_bac','time_set','subkingdom', 'clade', 'species','doi', 'CID', 'Canonical_smiles', 'np_size_min (nm)', 'np_size_max (nm)'], axis=1)

def rename_method(value):
    if value.startswith('c'):
        return 'chemical_synthesis'
    elif value.startswith('g'):
        return 'green_synthesis'
    else:
        return value

ZOI_1['np_synthesis'] = ZOI_1['np_synthesis'].apply(lambda x: rename_method(x.lower()))

# print(ZOI_1.columns)
ZOI_2 = ZOI_1[ZOI_1['concentration'] > 0]
# print(ZOI_2['concentration'].quantile(0.99))
ZOI_3 = ZOI_2[ZOI_2['concentration'] < ZOI_2['concentration'].quantile(0.99)]
ZOI_3 =ZOI_3[ZOI_3['zoi_np'] > 5] # use this if it improve the model performance; right now the range is from 0 to 45
# print(ZOI_3['np_size_avg (nm)'].quantile(0.99))
ZOI_4 = ZOI_3[ZOI_3['np_size_avg (nm)'] < ZOI_3['np_size_avg (nm)'].quantile(0.99)]
# print(ZOI_3['avg_Incub_period, h'].quantile(0.99))
ZOI_5 = ZOI_4[ZOI_4['avg_Incub_period, h'] < ZOI_4['avg_Incub_period, h'].quantile(0.99)]
print('hh',max(ZOI_5['concentration']), max(ZOI_5['np_size_avg (nm)']),max(ZOI_5['avg_Incub_period, h']), min(ZOI_5['zoi_np']))
ZOI_5 = ZOI_5.drop_duplicates()
ZOI_5 = ZOI_5.reset_index(drop=True)
# print(ZOI_5.info())
# ZOI_5.to_csv('ZOI_5.csv')

numerical_columns = ZOI_5.select_dtypes(include=['float64', 'int64'])
categorical_columns = ZOI_5.select_dtypes(include=['object'])
# print('ZOI_5', ZOI_5.info())
# print('ss', numerical_columns)
'''heatmap before variance threshold and highly related column removal'''
# # Create a correlation matrix
correlation_matrix = numerical_columns.corr()

# # Generate the heatmap without displaying values
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap of Numerical Columns')
plt.show()

from sklearn.feature_selection import VarianceThreshold
# Fit VarianceThreshold with a threshold of 0.9
threshold = 0.95
selector = VarianceThreshold(threshold)
selector.fit(numerical_columns)

# Get the indices of columns to keep
columns_to_keep = numerical_columns.columns[selector.get_support()]

# Filter the DataFrame with columns having variance above the threshold
ZOI_data_filtered = numerical_columns[columns_to_keep]

# Create a correlation matrix after variance thresholding
correlation_matrix_variance = ZOI_data_filtered.corr()

# Remove highly correlated columns based on Pearson correlation coefficient threshold
high_corr_columns = set()  # To store highly correlated columns
corr_threshold = 0.95  # Pearson correlation coefficient threshold

for i in range(len(correlation_matrix_variance.columns)):
    for j in range(i):
        if abs(correlation_matrix_variance.iloc[i, j]) > corr_threshold:
            colname = correlation_matrix_variance.columns[i]
            high_corr_columns.add(colname)

# Create a DataFrame without highly correlated columns
ZOI_data_no_high_corr = ZOI_data_filtered.drop(columns=high_corr_columns)

# Generate the heatmap after removing highly correlated columns
plt.figure(figsize=(10, 8))
sns.heatmap(ZOI_data_no_high_corr.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap after Variance and Pearson Correlation Threshold Filtering')
plt.show()

final_ZOI = pd.concat([categorical_columns, ZOI_data_no_high_corr], axis=1)
final_ZOI.to_csv('preprocessed_ZOI_original9595_copy.csv')
final_ZOI = pd.concat([final_ZOI, ZOI_5['source']], axis=1)
final_ZOI = final_ZOI.drop_duplicates()
final_ZOI.reset_index(drop=True)
# final_ZOI.to_csv('preprocessed_ZOI_original9595.csv')
# print('final_ZOI', final_ZOI.info())
final_ZOI

# # Plotting the KDE plot for the 'concentration' column
# plt.figure(figsize=(8, 6))
# sns.kdeplot(data=ZOI_5['avg_Incub_period, h'], shade=True)
# plt.title('KDE Plot for size')
# plt.xlabel('size')
# plt.ylabel('Density')
# plt.show()
