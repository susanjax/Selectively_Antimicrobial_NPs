import pandas as pd
import V4_models_ZOI
import random
from  V4_ZOI.Models import  V4_transform_ZOI

population_size = 100
df_ZOI = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_ZOI\data\preprocessed\final_preprocessed_ZOI.csv')
df_ZOI =df_ZOI[df_ZOI['np_synthesis'].isin(['green_synthesis','chemical_synthesis'])]
# print(df_MIC['np_synthesis'].unique())
X = df_ZOI.drop(['Unnamed: 0','reference','source', 'zoi_np'], axis=1) # no need for concentration, zoi or gi as all of these parameters will be predicted
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
  # new_one = new_one[(new_one['concentration (ug/ml)'] > 5) & (new_one['Hydrodynamic diameter (nm)']> 8)]
  new = new_one.head(size)
  new = new.reset_index(drop=True)
  #material and bacterial descriptor created here have random samples taken from the original data, later we use part of it to replace in randomly generated df so that we can keep the same NP with its descriptor and same bacteria with descriptor
  material_descriptor = X.iloc[[random.randrange(0, len(X)) for _ in range(len(new))]]
  material_descriptor = material_descriptor.reset_index(drop=True)
  # print(material_descriptor)
  new[['np', 'Valance_electron', 'amw', 'NumHeteroatoms', 'kappa1', 'kappa2', 'kappa3', 'Phi']] = material_descriptor[['np', 'Valance_electron', 'amw', 'NumHeteroatoms', 'kappa1', 'kappa2', 'kappa3', 'Phi']]
  return new


dff = population(population_size)
# print('here', dff)
# print(dff.loc[3])

"""change bacteria type into pathogenic and non pathogenic"""
def bacteria_type(population_df):
  single_bacteria_pathogen = uniq_bacteria_data.loc[uniq_bacteria_data['bacteria'] == 'Escherichia coli'] # Escherichia coli, Klebsiella pneumoniae Staphylococcus aureus,Acinetobacter baumannii, Salmonella typhimurium
  single_bacteria_nonpathogen = uniq_bacteria_data.loc[uniq_bacteria_data['bacteria'] == 'Bacillus subtilis']
  pop_non_pathogen =pd.concat([single_bacteria_nonpathogen]*len(population_df), ignore_index=True)
  pop_pathogen = pd.concat([single_bacteria_pathogen] * len(population_df), ignore_index=True)
  df_pathogen= population_df.copy()
  df_non_pathogen = population_df.copy()
  # print(population_df)
  df_non_pathogen[['bacteria', 'bac_type', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'gram', 'min_Incub_period, h','avg_Incub_period, h', 'growth_temp, C','isolated_from']] = pop_non_pathogen[['bacteria', 'bac_type', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'gram', 'min_Incub_period, h', 'avg_Incub_period, h', 'growth_temp, C','isolated_from']]
  df_pathogen[['bacteria', 'bac_type', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'gram', 'min_Incub_period, h', 'avg_Incub_period, h', 'growth_temp, C','isolated_from']] = pop_pathogen[['bacteria', 'bac_type', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'gram', 'min_Incub_period, h', 'avg_Incub_period, h', 'growth_temp, C','isolated_from']]
  return df_non_pathogen, df_pathogen

# print('here', bacteria_type(dff))

def fitness(df):
  n_path, path_gen = bacteria_type(df)
  np = V4_transform_ZOI.transform(n_path)
  p = V4_transform_ZOI.transform(path_gen)
  normal_b = V4_models_ZOI.cat_predict(np)
  pathogen_b = V4_models_ZOI.cat_predict(p)
  fitness = []
  norm_v = []
  path_v = []
  for a in range(len(normal_b)):
    n = normal_b[a]
    c = pathogen_b[a]
    #for MIC, higher MIC means lower toxicity and lower MIC means higher toxicity; so we are searching for higher MIC of non pathogenic bacteria and lower MIC of pathogenic bacteria
    fit = c - n # higher MIC for normal bacteria and lower MIC for pathogenic bacteria
    fitnn = fit.tolist()
    norm_v.append(n)
    path_v.append(c)
    fitness.append(fitnn)
  copy = n_path.assign(pred_ZOI_norm=norm_v)
  copy1 = copy.assign(pathogenic_bacteria = path_gen['bacteria'].tolist())
  copy2 = copy1.assign(pred_ZOI_pathogen=path_v)
  copy3 = copy2.assign(Fitness = fitness)
  copy3 = copy3.sort_values('Fitness', ascending=False)
  # copy3['pred_norm_MIC_original'] = 10** copy3['pred_MIC_norm']
  # copy3['pred_path_MIC_original'] = 10** copy3['pred_MIC_pathogen']
  return copy3

# print(fitness(dff))


