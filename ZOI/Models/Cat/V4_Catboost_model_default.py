from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
from V4_ZOI.Models import V4_transform_ZOI
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ZOI_df = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_ZOI\data\preprocessed\final_preprocessed_ZOI.csv')
ZOI_df = ZOI_df.drop(['source','reference',], axis=1)
print(ZOI_df.columns)

# ZOI_train_df = ZOI_df[ZOI_df['source'] <= 4].drop(['source'], axis=1)
# ZOI_test_df = ZOI_df[ZOI_df['source'] == 3].drop(['source'], axis=1)
ZOI_train_df = ZOI_df.drop_duplicates()
ZOI_train_Y = ZOI_train_df[['zoi_np']].copy()
# ZOI_test_Y = ZOI_test_df[['zoi_np']].copy()

ZOI_train_df = ZOI_train_df.reset_index(drop=True)
# ZOI_test_df = ZOI_test_df.reset_index(drop=True)
ZOI_train_Y = ZOI_train_Y.reset_index(drop=True)
# ZOI_test_Y = ZOI_test_Y.reset_index(drop=True)

X_train_enc = V4_transform_ZOI.transform(ZOI_train_df)
# X_test_enc = V4_transform_ZOI.transform(ZOI_test_df)
Y_train_enc = ZOI_train_Y
# Y_test_enc = ZOI_test_Y

X_train, X_test, Y_train, Y_test = train_test_split(X_train_enc, Y_train_enc, test_size=0.2, random_state=42)
model = CatBoostRegressor()
model.fit(X_train, Y_train)
train_predictions = model.predict(X_train)
validation_predictions = model.predict(X_test)
# test_predictions = model.predict(X_test_enc)


# Evaluation metrics
train_r2 = r2_score(Y_train, train_predictions)
train_mae = mean_absolute_error(Y_train, train_predictions)
train_mse = mean_squared_error(Y_train, train_predictions)**0.5  # RMSE

valiation_r2 = r2_score(Y_test, validation_predictions)
valiation_mae = mean_squared_error(Y_test, validation_predictions)
validation_mse = mean_squared_error(Y_test, validation_predictions)

# test_r2 = r2_score(Y_test_enc, test_predictions)
# test_mae = mean_absolute_error(Y_test_enc, test_predictions)
# test_mse = mean_squared_error(Y_test_enc, test_predictions)**0.5  # RMSE

# Print metrics
print('Train')
print('Train R-square:', train_r2)
print('Mean Absolute Error:', train_mae)
print('Root Mean Squared Error:', train_mse)

print('Validation')
print('Validation R-square:', valiation_r2)
print('Mean Absolute Error:', valiation_mae)
print('Root Mean Squared Error:', validation_mse)

print('Test')
# print('Test R-square:', test_r2)
# print('Mean Absolute Error:', test_mae)
# print('Root Mean Squared Error:', test_mse)

'''visualization'''
import matplotlib.pyplot as plt
import seaborn as sns

# Set custom parameters for the plot
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.set(font_scale=2)

# Create a figure and axis
f, ax = plt.subplots(figsize=(13, 10))

# Scatter plot for train and validation data
plt.scatter(Y_train, train_predictions, color='#2d4d85', s=50, label='Train Data', alpha=0.7)
plt.scatter(Y_test, validation_predictions, color='#951d6d', s=50, label='Validation', alpha=0.7)
# plt.scatter(Y_test_enc, test_predictions, color='#f62d2d', s=50, label='test', alpha=0.7)

# Plotting the diagonal line (perfect prediction)
plt.plot(Y_train, Y_train, color='#444444', linewidth=2)

# Set labels and title
plt.title('cat Regressor - Predicted vs Actual')
plt.xlabel('Actual Data')
plt.ylabel('Predicted Data')
plt.legend()
plt.savefig('default_cat.png', transparent=True)
# Show the plot
plt.show()

