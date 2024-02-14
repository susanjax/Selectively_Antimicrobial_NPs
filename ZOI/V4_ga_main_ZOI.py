import pandas as pd
import V4_ga_compd_generation_ZOI
import V4_crossing_mutation_ZOI
import time


mutation_rate = 0.1
cross_over_rate = 0.1


#import warnings
#warnings.filterwarnings("ignore")
# df = ga_compd_generation_new.fitness(ga_compd_generation_new.population(population_size)).sort_values('Fitness', ascending=False)

def new_generations(Gen, population_size):
    half = int((population_size * 0.5)+1)
    selected = Gen.iloc[:half,:]
    new = [selected, V4_ga_compd_generation_ZOI.fitness(V4_ga_compd_generation_ZOI.population(half))]
    new_generation_input = pd.concat(new)
    new_generation_input.reset_index(drop=True, inplace=True)
    new_gen = V4_crossing_mutation_ZOI.evolve_crossing(new_generation_input, cross_over_rate, mutation_rate)
    new_gen.reset_index(drop=True, inplace=True)
    return new_gen


# print('original', df, 'new', new_generations(df, population_size))
means = []
maxs = []
def Genetic_Algorithm(generation_number, population_size):
    Generation1 = V4_ga_compd_generation_ZOI.fitness(V4_ga_compd_generation_ZOI.population(population_size)).sort_values('Fitness', ascending=False)
    mean1 = Generation1['Fitness'].mean()
    max1 = Generation1['Fitness'].max()
    Generation1.to_csv('output/B_sub_vs_s_aureus/pop_size_' + str(population_size) + '_Generation_1.csv')
    Generation2 = V4_crossing_mutation_ZOI.evolve_crossing(Generation1, cross_over_rate, mutation_rate)
    mean2 = Generation2['Fitness'].mean()
    max2 = Generation2['Fitness'].max()
    Generation2.to_csv('output/B_sub_vs_s_aureus/pop_size_' + str(population_size)+ '_Generation_2.csv')
    Generation_next = Generation2
    means = [ mean1, mean2]
    maxs = [max1, max2]
    g = 3
    while g in range(generation_number + 1):
        Generation_next = new_generations(Generation_next, population_size)
        # i = Generation_next.iloc[0][0]
        mean = Generation_next['Fitness'].mean()
        max = Generation_next['Fitness'].max()
        Generation_next.to_csv('output/B_sub_vs_s_aureus/pop_size_' + str(population_size) + '_Generation_' + str(g) + '.csv')
        means.append(mean)
        maxs.append(max)

        g += 1

    genn = generation_number + 1
    gens = list(range(1,genn))
    summary = pd.DataFrame( list(zip( gens, means, maxs)), columns= ['generations','mean', 'max'] )
    print(summary)
    #summary.to_csv('output/results/pop_size_50/t2/summary_pop_size_' + str(population_size) + '.csv')
    summary.to_csv('output/B_sub_vs_s_aureus/summary_pop_size_' + str(population_size) +'_gen_' + str(generation_number)+'.csv')
    return Generation_next


def final_loop():
    pop_col = []
    time_all = []
    gen_col = []
    gen = 100
    while gen <= 100:
        population_size = 50
        while population_size <= 80:
            st = time.time()
            Genetic_Algorithm(gen, population_size)
                #population_size += 10
            gen_col.append(gen)
                # gen += 10
            escape_time = time.time() - st
            time_all.append(escape_time)
            pop_col.append(population_size)
            print('Escape time:', escape_time)
            population_size += 10
        gen +=10
        et = pd.DataFrame(list(zip(pop_col, gen_col, time_all)), columns=['population_size','Generation number', 'Time'])
        #et.to_csv('output/results/pop_size_50/t2/Time_' + str(population_size-10) + '.csv')
        et.to_csv('output/B_sub_vs_s_aureus/Time_' + str(population_size) + '.csv')

final_loop()