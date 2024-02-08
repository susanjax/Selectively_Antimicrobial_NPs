import time

import pandas as pd
import V4_models
import random
from  V4_MIC.Models import  V4_transform_MIC_trial

population_size = 100
df_MIC = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_MIC\data\preprocessed\preprocessed_MIC_mod.csv')
df_MIC =df_MIC[df_MIC['np_synthesis'].isin(['green_synthesis','chemical_synthesis'])]
# print(df_MIC['np_synthesis'].unique())
X = df_MIC.drop(['Unnamed: 0','reference','source', 'concentration'], axis=1) # no need for concentration, zoi or gi as all of these parameters will be predicted
# print(X.columns)
# it might be better not to choose random generation from unique set, we can just choose from all data ( higher number of data have higher chance to pick, this will imporve the predictibility of the material as model predict best for those who have higher number of data)
''' generate dataframe with unique bacteria'''
uniq_bacteria_data = X.drop_duplicates('bacteria')
"""uniq value datasets"""
uniq = [] # stores all the unique characters available in the dataset, it helps to make a new population with random parameters
for a in range(len(X.columns)):
  uni = pd.unique(X.iloc[:, a])
  uniq.append(uni)

"""create individual with values that are picked from the uniq array above"""

def individuals():
  indv = []
  for a in range(len(X.columns)):
    uniqas = random.choice(uniq[a])
    indv.append(uniqas)
  return indv

# print(individuals())
"""generate population with specific population size"""
#population with specific material descriptors were generated but cell line were still random
def population(size):
  pops = []
  for indv in range(2*size):
    single = individuals()
    pops.append(single)
  new_one = pd.DataFrame(data=pops, columns=X.columns)
  #control the range of any column/parameter from here
  # neww = new_one[(new_one['concentration (ug/ml)'] > 5) & (new_one['Hydrodynamic diameter (nm)']> 8)]
  neww = new_one[(new_one['np_size_avg (nm)'] > 5)]
  new = neww.head(size)
  new = new.reset_index(drop=True)
  #material and bacterial descriptor created here have random samples taken from the original data, later we use part of it to replace in randomly generated df so that we can keep the same NP with its descriptor and same bacteria with descriptor
  material_descriptor = X.iloc[[random.randrange(0, len(X)) for _ in range(len(new))]]
  material_descriptor = material_descriptor.reset_index(drop=True)
  # print(material_descriptor)
  new[['np', 'mol_weight (g/mol)', 'Valance_electron','labuteASA', 'tpsa', 'CrippenMR', 'chi0v']] = material_descriptor[['np', 'mol_weight (g/mol)', 'Valance_electron', 'labuteASA', 'tpsa', 'CrippenMR', 'chi0v']]
  return new


dff = population(population_size)
# print('here', dff)
# print(dff.loc[3])

"""change bacteria type into pathogenic and non pathogenic"""
def bacteria_type(population_df):
  single_bacteria_pathogen = uniq_bacteria_data.loc[uniq_bacteria_data['bacteria'] == 'Klebsiella pneumoniae'] # Escherichia coli, Staphylococcus aureus, Acinetobacter baumannii, Salmonella typhimurium
  single_bacteria_nonpathogen = uniq_bacteria_data.loc[uniq_bacteria_data['bacteria'] == 'Bacillus subtilis']
  pop_non_pathogen =pd.concat([single_bacteria_nonpathogen]*len(population_df), ignore_index=True)
  pop_pathogen = pd.concat([single_bacteria_pathogen] * len(population_df), ignore_index=True)
  df_pathogen= population_df.copy()
  df_non_pathogen = population_df.copy()
  # print(population_df)
  df_non_pathogen[['bacteria', 'bac_type', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'gram', 'min_Incub_period, h', 'growth_temp, C','isolated_from']] = pop_non_pathogen[['bacteria', 'bac_type', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'gram', 'min_Incub_period, h', 'growth_temp, C','isolated_from']]
  df_pathogen[['bacteria', 'bac_type', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'gram', 'min_Incub_period, h', 'growth_temp, C','isolated_from']] = pop_pathogen[['bacteria', 'bac_type', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'gram', 'min_Incub_period, h', 'growth_temp, C','isolated_from']]
  return df_non_pathogen, df_pathogen

# print('here', bacteria_type(dff))

def fitness(df):
  # start_time = time.time()
  n_path, path_gen = bacteria_type(df)
  np = V4_transform_MIC_trial.transform(n_path)
  p = V4_transform_MIC_trial.transform(path_gen)
  normal_b = V4_models.cat_predict(np)
  pathogen_b = V4_models.cat_predict(p)
  # end_time = time.time()
  # print('total time for model to predict 100 sample in milisec: ', (end_time-start_time)*1000, 'length of df', len(df))
  fitness = []
  norm_v = []
  path_v = []
  for a in range(len(normal_b)):
    n = normal_b[a]
    c = pathogen_b[a]
    #for MIC, higher MIC means lower toxicity and lower MIC means higher toxicity; so we are searching for higher MIC of non pathogenic bacteria and lower MIC of pathogenic bacteria
    fit = n - c # higher MIC for normal bacteria and lower MIC for pathogenic bacteria
    fitnn = fit.tolist()
    norm_v.append(n)
    path_v.append(c)
    fitness.append(fitnn)
  copy = n_path.assign(pred_MIC_norm=norm_v)
  copy1 = copy.assign(pathogenic_bacteria = path_gen['bacteria'].tolist())
  copy2 = copy1.assign(pred_MIC_pathogen=path_v)
  copy3 = copy2.assign(Fitness = fitness)
  copy3 = copy3.sort_values('Fitness', ascending=False)
  copy3['pred_norm_MIC_original'] = 10** copy3['pred_MIC_norm']
  copy3['pred_path_MIC_original'] = 10** copy3['pred_MIC_pathogen']
  return copy3

# print(fitness(dff))
# fitness(dff)

