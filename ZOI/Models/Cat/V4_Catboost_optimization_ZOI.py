import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostRegressor
from V4_ZOI.Models import V4_transform_ZOI
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings('ignore')

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

X = X_train_enc
Y = Y_train_enc

def objective(trial, x, y):
    param = {
        'depth': trial.suggest_int('depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000, step=1),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 8),
        'border_count': trial.suggest_int('border_count', 1, 255),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0)
    }

    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)  # changing the n_split to 10 reduces the r2 score
    cv_scores = np.empty(10)
    for idx, (train_indices, test_indices) in enumerate(cv.split(X_train_enc, X_train_enc[['phylum']])):
        x_train, x_test = X_train_enc.iloc[train_indices], X_train_enc.iloc[test_indices]
        y_train, y_test = Y_train_enc.iloc[train_indices], Y_train_enc.iloc[test_indices]
        model = CatBoostRegressor(**param)
        model.fit(x_train, y_train)
        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)
        print('training_r2', r2_score(y_train, pred_train))
        print('mean_sq_error', mean_squared_error(y_train, pred_train))
        cv_scores[idx] = r2_score(y_test, pred_test)
        print('r2 score', idx, cv_scores)

    return np.mean(cv_scores)
study = optuna.create_study(direction="maximize", study_name="CatBoost_Regressor")
func = lambda trial: objective(trial, X, Y)
study.optimize(func, n_trials=50)

print(f"\tBest value (r2 score): {study.best_value:.5f}")
print(f"\tBest params:")

for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")