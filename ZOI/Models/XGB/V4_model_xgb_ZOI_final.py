import pandas as pd
import numpy as np
from V4_ZOI.Models import V4_transform_ZOI
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ZOI_df = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_ZOI\data\preprocessed\final_preprocessed_ZOI.csv')


ZOI_train_df = ZOI_df[ZOI_df['source'] < 6].drop(['source'], axis=1)
ZOI_test_df = ZOI_df[ZOI_df['source'] >= 6].drop(['source'], axis=1)
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


train_R2_metric_results = []
train_mse_metric_results= []
train_mae_metric_results = []
Validation_R2_metric_results = []
Validation_mse_metric_results= []
Validation_mae_metric_results = []

test_R2_metric_results = []
test_mse_metric_results = []
test_mae_metric_results = []

val_R2_metric_results = []
val_mse_metric_results = []
val_mae_metric_results = []

print('X_train_enc',X_train_enc.columns)
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)  # changing the n_split to 10 reduces the r2 score
cv_scores = np.empty(10)
for idx, (train_indices, test_indices) in enumerate(cv.split(X_train_enc, X_train_enc[['phylum']])): #phylum #gram #method
    x_train, x_test = X_train_enc.iloc[train_indices], X_train_enc.iloc[test_indices]
    y_train, y_test = Y_train_enc.iloc[train_indices], Y_train_enc.iloc[test_indices]

#
# cv = KFold(n_splits=10, shuffle=True, random_state=42)
#
# cv_scores = np.empty(10)
# for idx, (train_indices, test_indices) in enumerate(cv.split(X_train_enc, Y_train_enc)):
#     X_train, X_test = X_train_enc.iloc[train_indices], X_train_enc.iloc[test_indices]
#     Y_train, Y_test = Y_train_enc.iloc[train_indices], Y_train_enc.iloc[test_indices]
    model = XGBRegressor(**n_best_hyperparameters)
    model.fit(x_train, y_train)
    train = model.predict(x_train)
    validation = model.predict(x_test)
    test_predictions = model.predict(X_test_enc)

    train_R2_metric_results.append(r2_score(y_train, train))
    train_mse_metric_results.append(mean_squared_error(y_train, train))
    train_mae_metric_results.append(mean_absolute_error(y_train, train))

    val_R2_metric_results.append(r2_score(y_test, validation))
    val_mse_metric_results.append(mean_squared_error(y_test, validation))
    val_mae_metric_results.append(mean_absolute_error(y_test, validation))

    test_R2_metric_results.append(r2_score(Y_test_enc, test_predictions))
    test_mse_metric_results.append(mean_squared_error(Y_test_enc, test_predictions))
    test_mae_metric_results.append(mean_absolute_error(Y_test_enc, test_predictions))

print('Train')
print('Train R-square:', np.mean(train_R2_metric_results))
print('Mean Absolute Error:', np.mean(train_mae_metric_results))
print('Mean Squared Error:', (np.mean(train_mse_metric_results)))
print('Root Mean Squared Error:', (np.mean(train_mse_metric_results)**(1/2)))

print('validation')
print('one-out cross-validation (R-square):', r2_score(y_test, validation))
print('10-fold cross-validation result (R-square):', np.mean(val_R2_metric_results))
print('10-fold cross-validation result (MAE):', np.mean(val_mae_metric_results))
print('10-fold cross-validation result (MSE):', (np.mean(val_mse_metric_results)))
print('10-fold cross-validation result (RMSE):', (np.mean(val_mse_metric_results)**(1/2)))


print('testing')
print('one-out cross-validation (R-square):', r2_score(Y_test_enc, test_predictions))
print('10-fold cross-validation result (R-square):', np.mean(test_R2_metric_results))
print('10-fold cross-validation result (MAE):', np.mean(test_mae_metric_results))
print('10-fold cross-validation result (MSE):', (np.mean(test_mse_metric_results)))
print('10-fold cross-validation result (RMSE):', (np.mean(test_mse_metric_results)**(1/2)))

import pickle
with open('xgb_model_MIC_final_ZOI.pkl', 'wb') as file:
    pickle.dump(model, file)

"""#Visualization"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.set(font_scale=2)

f, ax = plt.subplots(figsize=(10, 10))
plt.scatter(y_train, train, color='#2d4d85', s=50, label='Train Data', alpha=0.3)
plt.scatter(y_test, validation, color='#951d6d', s=50, label='Validation', alpha=0.3)
plt.scatter(Y_test_enc, test_predictions, color='#f62d2d', s=50, label='test', alpha=1)
plt.plot(y_test, (y_test - 2*np.sqrt(metrics.mean_squared_error(y_test, validation))), color='#444444', linewidth = 1)
plt.plot(y_test, (y_test + 2*np.sqrt(metrics.mean_squared_error(y_test, validation))), color='#444444', linewidth = 1)
# Plotting the diagonal line (perfect prediction)
plt.plot(y_train, y_train, color='#444444', linewidth=2)
plt.xlim(0,50)
plt.ylim(0, 50)
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

# Save the figure with transparency
plt.savefig('model_xgb_ZOI_optimized_CV_with_validation_error.png', transparent=True)
plt.show()

# Getting feature importance from the model
feature_importance = model.feature_importances_
feature_names = x_train.columns
cols =['concentration', 'np_size_avg (nm)',
       'min_Incub_period, h', 'avg_Incub_period, h', 'growth_temp, C',
       'Valance_electron', 'amw', 'NumHeteroatoms', 'kappa1', 'kappa2',
       'kappa3', 'Phi']


# Creating a DataFrame to organize feature importance
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Define a custom color palette as a gradient from red to blue
custom_palette = sns.color_palette("RdBu", n_colors=len(feature_importance_df))

# Filter feature importance for selected columns
selected_feature_importance = feature_importance_df[feature_importance_df['Feature'].isin(cols)]

# Plotting feature importance with custom color gradient for selected columns
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=selected_feature_importance, palette=custom_palette)
plt.title('XGB Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()

#
# # Save the plot
plt.savefig('xgb_feature_importance_optimized.png', transparent=True)
#
# # Show the plot if needed
plt.show()

import shap
import matplotlib.pyplot as plt

X_importance = x_train
selected_X_importance = X_importance[cols]
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_importance)
# Filter SHAP values for selected columns
selected_shap_values = shap_values[:, X_importance.columns.isin(cols)]

# Create the SHAP summary plot with selected columns
# shap.summary_plot(shap_values, X_importance, show=False)
shap.summary_plot(selected_shap_values, selected_X_importance, show=False)
plt.tight_layout()  # Ensures plots are properly arranged

# Save the plot before showing it
plt.savefig('important_features_xgb_ZOI.png', transparent=True)

# Show the plot if needed
plt.show()