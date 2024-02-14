import pandas as pd
import pickle
import warnings


warnings.filterwarnings('ignore')

xgb_model_path = r'C:\Users\user\Desktop\Valya\V4_ZOI\Models\XGB\xgb_model_MIC_final_ZOI.pkl'
cat_model_path = r'C:\Users\user\Desktop\Valya\V4_ZOI\Models\Cat\cat_model_final_ZOI.pkl'

def model_load(path):
    with open(path, 'rb') as f:
        xgb_model = pickle.load(f)
        return xgb_model
mod = model_load(xgb_model_path)

def xgb_predict(input_data):
    x = input_data
    predict = mod.predict(x)
    return predict

def cat_load(path):
    with open(path, 'rb') as f:
        cat_model = pickle.load(f)
        return cat_model
cat_mod = model_load(cat_model_path)

def cat_predict(input_data):
    x = input_data
    predict = cat_mod.predict(x)
    return predict
