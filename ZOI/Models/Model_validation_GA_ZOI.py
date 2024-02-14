import pickle
import pandas as pd
import numpy as np
from V4_ZOI.Models import V4_transform_ZOI
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ZOI_data = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_ZOI\data\preprocessed\final_preprocessed_ZOI.csv')
# MIC_data=MIC_data[MIC_data['concentration']>0]
ZOI_df = ZOI_data.drop(['Unnamed: 0','np','reference',], axis=1)

# df = MIC_data[MIC_data['source'].isin([1, 6])]
df = ZOI_data[ZOI_data['source'] >= 6]
df = df.reset_index(drop=True)
print(df)
df_enc = V4_transform_ZOI.transform(df)

Cat_model_path = r'C:\Users\user\Desktop\Valya\V4_ZOI\Models\Cat\cat_model_final_ZOI.pkl'

def model_load(path):
    with open(path, 'rb') as f:
        cat_model = pickle.load(f)
        return cat_model
mod = model_load(Cat_model_path)

def cat_predict(input_data):
    x = input_data
    predict = mod.predict(x)
    return predict

predicted_df_y = cat_predict(df_enc)
print(predicted_df_y)

df['predicted'] = predicted_df_y
df.to_csv('validaiton_ZOI.csv')