import random
import V4_ga_compd_generation_ZOI

# in1 = ['Ag', 'ZnO', 'Bacillus subtilis', 'non-pathogenic', 'None', 'None', 'Chemical_synthesis using silver nitrate and zinc nitrate', 'MIC', 'nanorods', 'Bacteria', nan, 'Bacillota', 'Bacilli', 'Bacillales', 'Bacillaceae', 'Bacillus', 'Bacillus subtilis group', 'p', 'soil', 10.4, 4, 7.08, 30, 1, 11, 107.868, 0, 0, 23.00188061, -0.0025, 1.78376517, 0.0, 0.0, 0.74025974, 1.74025974, 2.143768512, -267.0895082514542, 'Escherichia coli', -148.93512773844262, -118.1543805130116]

in1 = ['Cu', 'Bacillus subtilis', 'non-pathogenic', 'chemical_synthesis', 'disk_diffusion', 'rod-shaped', 'Bacteria', 'Bacillota', 'Bacilli', 'Bacillales', 'Bacillaceae', 'Bacillus', 'p', 'soil', 7.6, 44.45, 0.16, 7.08, 30, 11, 63.546, 1, 1.519480519, 0.444480519, 4.219480519, 0.675379491, 15.570224, 'Staphylococcus aureus', 15.375084, -0.19513988494873047]
in2 = ['TiO2', 'Bacillus subtilis', 'non-pathogenic', 'green_synthesis', 'disc_diffusion', 'cubic', 'Bacteria', 'Bacillota', 'Bacilli', 'Bacillales', 'Bacillaceae', 'Bacillus', 'p', 'soil', 7.95454, 34.0, 0.16, 7.08, 30, 16, 79.865, 3, 3.314285714, 2.314285714, 2.314285714, 2.556734694, 3.2289157, 'Staphylococcus aureus', 7.373609, 4.144693374633789]


indv2_list = V4_ga_compd_generation_ZOI.fitness(V4_ga_compd_generation_ZOI.population(size=50))
# print(indv2_list.columns)
# print(indv2_list.loc[2].values.tolist())
cross_over_frequency = 0.2
mutation_rate = 0.2


#following code with mutate each point 1, 9, 10, 11 etc with the prbability of cross_over frequency; while remaining other feature will have one probability to be changed keeping all parameter intact because these feature are inter-related.
def to_crossover(indv1, indv2, cross_over_frequency):
    a = random.random()

    for each in range(1,len(indv1)):
        if (each == 3) or (each == 4) or (each == 5) or (each == 14) or (each == 15):
            # 3=np_synthesis, 4 = method, 5 = shape, 14- concentration, 15 - np size
            if random.random()< cross_over_frequency:
                indv1[each] = indv2[each]
            continue
        if a < cross_over_frequency:
            indv1[each] = indv2[each]

    return indv1

# print((to_crossover(in1, in2, cross_over_frequency=0.5)),'\n')

def to_mutation(individual1, mutation_rate):
    individual2 = indv2_list.iloc[random.randrange(20)].values.tolist()
    # print(individual2)
    mut = to_crossover(individual1, individual2, mutation_rate)
    return mut
# print(len(in1),to_mutation(in1, mutation_rate=0.1))

# print(to_mutation(in1, mutation_rate))