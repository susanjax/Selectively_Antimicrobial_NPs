# SelTox: Discovering the Capacity of Selectively Antimicrobial Nanoparticles for Targeted Eradication of Pathogenic Bacteria

**Abstract**

For years, researchers have searched for novel antibiotics to combat pathogenic infections. However, antibiotics lack specificity, harm beneficial microbes and cause emergence of antibiotic-resistant strains. This study proposes an innovative approach to selectively eradicate pathogenic bacteria with a minimal effect on non-pathogenic ones by discovering selectively antimicrobial nanoparticles. To achieve this, we first compiled a comprehensive database to characterize nanoparticles and their antibacterial activity. Then, we trained CatBoost regression models for predicting minimal concentration (MC) and zone of inhibition (ZOI). The models achieved a 10-fold cross validation (CV) R2 score of 0.82 and 0.84 with root mean square error (RMSE) of 0.46 and 2.41, respectively. Finally, we developed a machine learning (ML) reinforced genetic algorithm (GA) to identify the best performing selective antibacterial NPs. As a proof of concept, a selectively antibacterial nanoparticle, CuO, was identified for targeted eradication of a pneumoniae causing pathogen *Klebsiella pneumoniae*. The difference in minimal bactericidal concentration (MBC) of 392.85 µg/ml was achieved, compared to *Bacillus subtilis*. These findings significantly contribute to the emerging research domain of selectively toxic (SelTox) nanoparticles and open the door for future exploration of synergetic interactions of SelTox nanoparticles with drugs.

**Guidelines**

Two repositories have been created: one for Minimal Inhibitory Concentration (MIC) to identify selectively antimicrobial NPs based on predicted minimal concentration, and another for Zone of Inhibition (ZOI) to identify selectively antimicrobial NPs based on predicted diameter of inhibition zone. Each repository contains code for the genetic algorithm, as well as separate folders for data, models, and output of the genetic algorithm. The data folder includes subfolders for raw data, preprocessed data, and data visualization. Similarly, the model folder contains code for model selection, optimization, validation, and visualization of model performance. Finally, the output folder stores results generated by the genetic algorithm regarding selectively antimicrobial NPs, along with information about the optimization of population size, generation number, and mutation and crossover rate.

**Data**

Preprocessing steps were conducted as outlined in the  V4_preprocessing_MIC.py file for MIC and V4_preprocessing_ZOI.py file for ZOI. Data distribution was visualized before and after preprocessing using pie charts, kernel density estimate (KDE) plots, and violin plots.

**Machine Learning**

**Model Selection:** We evaluated performance of various regression models on raw and preprocessed data for prediction of MC and ZOI using Python scripts in the LP folder. The results were stored in respective CSV files, and model performance was visualized using bar plots in the Visualization folder. CatBoost regressor and XGB regressor models emerged as the top performers and were subsequently selected for further optimization.

**Model Optimization:** Hyperparameter tuning was performed to identify the best parameters for the CatBoost regressor and XGB regressor models. The CatBoost regressor exhibited slightly superior performance and was used for predicting MC and ZOI.

**Model Validation:** Ten-fold cross-validation was conducted before assessing model performance on the test dataset. Additionally, the performance of the model on specific NPs was evaluated and visualized using scatter plots.

**Genetic Algorithm**

To screen selectively antimicrobial NPs, a genetic algorithm was implemented, with all necessary files stored in the MIC and ZOI folders.

**Unique NP generation:** Parameters for unique metal and metal oxide NPs were generated using the V4_ga_compd_generation.py file. The antimicrobial activity of these NPs was assessed against pneumonia-causing pathogenic bacteria on *Staphylococcus aureus* and *Klebsiella pneumoniae*, with predictions made using the optimized CatBoost regression model.

**NP mutation and crossover:** The V4_crossing_mutation.py and V4_cross_modified.py files were used for the mutation, crossover, and evolution of nanoparticle parameters to enhance selective antimicrobial properties.

**Selective antimicrobial NP screening:** The main file, V4_ga_main.py, orchestrated population generation and evolution up to a user-defined generation number. The fitness score of each individual NP was calculated by comparing the logarithm of MC or the difference in ZOI.

**Results:** The top candidates of selectively toxic NPs generated by the algorithm were stored in the output folder for further analysis and consideration.
