# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:25:59 2020
online source: https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6
@author: TiberEmi
"""

import numpy as np
import ga
import split


sol_per_pop = 1

nr_bit_num = 22

genes = 2


def split(word): 
    return [char for char in word]
minimo = 6500000-3197833
minimo_bin = bin(minimo)
splitted = split(minimo_bin)
splitted = splitted[2:]
splitted_int = np.zeros([1,2*np.size(splitted)])
for i in range(np.size(splitted)):
    splitted_int[0,i] = int(splitted[i])

splitted_int[0,np.size(splitted):] = splitted_int[0,0:np.size(splitted)]
splitted_int = splitted_int.astype(np.int64)

   

fitness = ga.foxholes(splitted_int,genes,22)

# Defining the population size.

pop_size = (sol_per_pop,genes*nr_bit_num) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.

#Creating the initial population.

new_population = np.random.randint(low=0, high=2, size=pop_size)

num_generations = 4000

num_parents_mating = 4

for generation in range(num_generations):
     # Measuring the fitness of each chromosome in the population.
     fitness = ga.foxholes(new_population,genes,nr_bit_num)
    # Selecting the best parents in the population for mating.
     parents = ga.select_mating_pool(new_population, fitness, 
                                       num_parents_mating)
 
     # Generating next generation using crossover.
     offspring_crossover = ga.crossover(parents,
                                        offspring_size=(pop_size[0]-parents.shape[0], genes*nr_bit_num))
 
     # Adding some variations to the offsrping using mutation.
     offspring_mutation = ga.mutation(offspring_crossover)
# Creating the new population based on the parents and offspring.
     new_population[0:parents.shape[0], :] = parents
     new_population[parents.shape[0]:, :] = offspring_mutation
         # The best result in the current iteration.



# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = ga.foxholes(new_population,genes,nr_bit_num)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.where(fitness == np.max(fitness))
conversion = np.zeros([np.size(new_population,0),genes])
for j in range(np.size(new_population,0)):
     conversion[j,:] = np.array([[int("".join(str(x) for x in new_population[j,0:nr_bit_num]), 2), int("".join(str(x) for x in new_population[j,nr_bit_num+1:nr_bit_num*2+1]), 2)]])
best = conversion[best_match_idx,:]/(10**6)-65                 
print("Best solution : ", best)
print("Best solution fitness : ", fitness[best_match_idx])