import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyRegressor
from V4_ZOI.Models import V4_transform_ZOI
from sklearn.model_selection import train_test_split


raw = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_ZOI\data\raw\final_merged_df.csv')
print(raw.columns)
# raw = raw.reset_index(drop=True)
raw = raw.drop(['source','reference',], axis=1)
raw = raw.drop_duplicates()
raw = raw.reset_index(drop=True)
raw_X = raw.drop(['zoi_np'], axis=1)
raw_Y = raw[['zoi_np']].copy()

rX = V4_transform_ZOI.first_transform(raw_X)

X_train, X_test, Y_train, Y_test = train_test_split(rX, raw_Y, test_size=0.2, random_state = 42)
clf = LazyRegressor(verbose=3,ignore_warnings=True, custom_metric=None)
train,test = clf.fit(X_train, X_test, Y_train, Y_test)

train_mod = train.iloc[:-0, :]
train.to_csv('Model_comparision_raw_ZOI.csv')
