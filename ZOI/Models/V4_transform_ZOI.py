import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# df_ZOI = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_ZOI\data\preprocessed\preprocessed_ZOI_original9595_MOD.csv')
# print(df_ZOI.info())
# df_model = df_ZOI.drop(['np', 'source','zoi_np','reference'], axis=1)

processed_data = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_ZOI\data\preprocessed\final_preprocessed_ZOI.csv')
X = processed_data.drop(['Unnamed: 0','source','zoi_np','reference',], axis=1)

all = X.drop_duplicates()
all = all.reset_index(drop=True)

# print(all.info())
# Extract numerical and categorical columns
numerical = all.select_dtypes(include=['int', 'float64'])
categorical = all.select_dtypes(include=['object'])

cat_col = categorical.columns
num_col = numerical.columns
# print('ca', cat_col, 'nc', num_col)

# cat_col = ['bacteria', 'bac_type', 'np_synthesis', 'method', 'shape', 'reference',
#        'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'gram',
#        'isolated_from']
# num_col = ['mol_weight (g/mol)', 'np_size_avg (nm)', 'time_set',
#        'min_Incub_period, h', 'growth_temp, C', 'Valance_electron',
#        'labuteASA', 'tpsa', 'chi0v']

# Standard scaler and label encoding
def transform(data):
    le = LabelEncoder()
    le.fit(categorical.values.flatten())  # Fit the encoder on all categorical data
    Xc_all = categorical.apply(le.transform)
    Xct = data[cat_col].apply(le.transform)

    sc = StandardScaler()
    X_all = sc.fit_transform(numerical)
    X_ss = sc.transform(data[num_col])
    X_sc = pd.DataFrame(X_ss, columns=num_col)
    join = pd.concat([Xct, X_sc], axis=1)
    return join

# print(all, transform(all))

def first_transform(data):
    le = LabelEncoder()
    le.fit(data.select_dtypes(include=['object']).values.flatten())  # Fit the encoder on all categorical data
    Xct = data.select_dtypes(include=['object']).apply(le.transform)
    # Xct = data[cat_col].apply(le.transform)
    Xct.reset_index(drop=True)

    sc = StandardScaler()
    X_all = sc.fit_transform(data.select_dtypes(include=['int', 'float64']))
    X_ss = sc.transform(data[data.select_dtypes(include=['int', 'float64']).columns])
    X_sc = pd.DataFrame(X_ss, columns=data.select_dtypes(include=['int', 'float64']).columns)
    join = pd.concat([Xct, X_sc], axis=1)
    return join