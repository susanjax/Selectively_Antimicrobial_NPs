import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from V4_ZOI.Models import V4_transform_ZOI
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ZOI_df = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_ZOI\data\preprocessed\final_preprocessed_ZOI.csv')

ZOI_train_df = ZOI_df[ZOI_df['source'] < 6].drop(['source'], axis=1)
ZOI_test_df = ZOI_df[ZOI_df['source'] >= 6].drop(['source'], axis=1)



# ZOI_test_ecoli = ZOI_test_df[ZOI_test_df['bacteria'] == 'Escherichia coli'] # 'Staphylococcus aureus' 'Enterococcus faecalis' 'Escherichia coli' 'Pseudomonas aeruginosa' 'Staphylococcus epidermidis' 'Bacillus subtilis'
# ZOI_test_df = ZOI_test_ecoli
# ZOI_test_ZnO = ZOI_test_df[ZOI_test_df['np'] == 'Cu'] #Ag, ZnO, CuO, Cu
# ZOI_test_df = ZOI_test_ZnO


ZOI_train_Y = ZOI_train_df[['zoi_np']].copy()
ZOI_test_Y = ZOI_test_df[['zoi_np']].copy()


ZOI_train_df = ZOI_train_df.reset_index(drop=True)
ZOI_test_df = ZOI_test_df.reset_index(drop=True)
ZOI_train_Y = ZOI_train_Y.reset_index(drop=True)
ZOI_test_Y = ZOI_test_Y.reset_index(drop=True)

X_train_enc = V4_transform_ZOI.transform(ZOI_train_df)
X_test_enc = V4_transform_ZOI.transform(ZOI_test_df)
Y_train_enc = ZOI_train_Y
Y_test_enc = ZOI_test_Y

n_best_hyperparameters = {
    'max_depth': 6,
    'learning_rate': 0.29778964486473575,
    'n_estimators': 950,
    'min_child_weight': 9,
    'gamma': 0.4730231011638504,
    'subsample': 0.9963430887980652,
    'colsample_bytree': 0.7623101361943776,
    'reg_lambda': 2.613320937293194,
    'reg_alpha': 1.0604193051539161,

}

cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2,
                            random_state=42)  # changing the n_split to 10 reduces the r2 score
cv_scores = np.empty(10)
for idx, (train_indices, test_indices) in enumerate(cv.split(X_train_enc, X_train_enc[['phylum']])): #phylum #gram #method
    X_train, X_test = X_train_enc.iloc[train_indices], X_train_enc.iloc[test_indices]
    Y_train, Y_test = Y_train_enc.iloc[train_indices], Y_train_enc.iloc[test_indices]

    model = XGBRegressor(**n_best_hyperparameters)
    model.fit(X_train, Y_train)
    train_predictions = model.predict(X_train)
    validation_predictions = model.predict(X_test)
    test_predictions = model.predict(X_test_enc)

# Evaluation metrics
train_r2 = r2_score(Y_train, train_predictions)
train_mae = mean_absolute_error(Y_train, train_predictions)
train_mse = mean_squared_error(Y_train, train_predictions)**0.5  # RMSE

validation_r2 = r2_score(Y_test, validation_predictions)
validation_mae = mean_absolute_error(Y_test, validation_predictions)
validation_mse = mean_squared_error(Y_test, validation_predictions)**0.5  # RMSE

test_r2 = r2_score(Y_test_enc, test_predictions)
test_mae = mean_absolute_error(Y_test_enc, test_predictions)
test_mse = mean_squared_error(Y_test_enc, test_predictions)**0.5  # RMSE

# Print metrics
print('Train')
print('Train R-square:', train_r2)
print('Mean Absolute Error:', train_mae)
print('Root Mean Squared Error:', train_mse)

print('validation')
print('validation R-square:', validation_r2)
print('Mean Absolute Error:', validation_mae)
print('Root Mean Squared Error:', validation_mse)

print('Test')
print('Test R-square:', test_r2)
print('Mean Absolute Error:', test_mae)
print('Root Mean Squared Error:', test_mse)

'''visualization'''
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# Set custom parameters for the plot
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.set(font_scale=2)

# Create a figure and axis
f, ax = plt.subplots(figsize=(10, 10))

# Scatter plot for train and validation data
plt.scatter(Y_train, train_predictions, color='#2d4d85', s=50, label='Train', alpha=0.3)
plt.scatter(Y_test, validation_predictions, color='#951d6d', s=50, label='Validation', alpha=0.3)
plt.scatter(Y_test_enc, test_predictions, color='#f62d2d', s=50, label='Test', alpha=1)
# plt.plot(Y_test, (Y_test - 2*np.sqrt(metrics.mean_squared_error(Y_test, test_predictions))), color='#444444', linewidth = 1)
# plt.plot(Y_test, (Y_test + 2*np.sqrt(metrics.mean_squared_error(Y_test, test_predictions))), color='#444444', linewidth = 1)
# Plotting the diagonal line (perfect prediction)
plt.plot(Y_train, Y_train, color='#444444', linewidth=2)

# Change the color and thickness of x and y axis
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['left'].set_linewidth(3)

# Set labels and title
plt.title('XGB Regressor')
plt.xlabel('Actual Data')
plt.ylabel('Predicted Data')
plt.legend()
plt.savefig('XGB_optimized_with_validation.png', transparent=True)
# Show the plot
plt.show()

