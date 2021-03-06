# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:25:59 2020
online source: https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6
@author: TiberEmi
"""

import numpy as np
import ga
import matplotlib.pyplot as plt
from pylive import live_plotter_xy


sol_per_pop = 100

nr_bit_num = 27

genes = 2

#def split(word): 
#    return [char for char in word]
#minimo = 6500000-3197833
#minimo_bin = bin(minimo)
#splitted = split(minimo_bin)
#splitted = splitted[2:]
#splitted_int = np.zeros([1,2*np.size(splitted)])
#for i in range(np.size(splitted)):
#    splitted_int[0,i] = int(splitted[i])
#
#splitted_int[0,np.size(splitted):] = splitted_int[0,0:np.size(splitted)]
#splitted_int = splitted_int.astype(np.int64)
#
#   
#
#fitness = ga.foxholes(splitted_int,genes,22)

# Defining the population size.

pop_size = (sol_per_pop,genes*nr_bit_num) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.

#Creating the initial population.

new_population = np.random.randint(low=0, high=2, size=pop_size)
old_fitness = np.zeros(np.size(new_population,0))
n_cross_point = 8
n_mut_point = 10

num_generations = 2000

num_parents_mating = int(pop_size[0]*2*0.9)
line1 = []
fun = []
gen = []
old_population = []
for generation in range(num_generations):
     # Measuring the fitness of each chromosome in the population.
     fitness = ga.foxholes(new_population,genes,nr_bit_num)
     print(np.min(fitness))
     fun = np.append(fun,np.min(fitness))
     gen = np.append(gen,generation)
     line1 = live_plotter_xy(gen,fun,line1)
     if generation>1:
        fit =  np.concatenate([old_fitness,fitness])
        index_selection = sorted(range(len(fit)), key=lambda k: fit[k])
        #index_selection = np.argsort(np.concatenate(fitness,old_fitness))
        pop = np.concatenate((old_population,new_population))
        new_population = pop[index_selection[0:sol_per_pop],:]
     if generation%1000==0:
        print("generation",generation, "with value=",np.min(fitness))
     old_fitness[0:] = fitness[0:]
    # Selecting the best parents in the population for mating.
     parents = ga.select_mating_pool(new_population, fitness, 
                                       num_parents_mating)
     # Generating next generation using crossover.
     offspring_crossover = ga.crossover(parents,n_cross_point,
                                        offspring_size=(int(parents.shape[0]/2), genes*nr_bit_num))
 
     # Adding some variations to the offspring using mutation.
     offspring_mutation = ga.mutation(pop_size[0]-offspring_crossover.shape[0],new_population,n_mut_point)
# Creating the new population based on the parents and offspring.
     old_population = new_population
     new_population[0:int(parents.shape[0]/2), :] = offspring_crossover
     new_population[int(parents.shape[0]/2):, :] = offspring_mutation
         # The best result in the current iteration.



# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = ga.foxholes(new_population,genes,nr_bit_num)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.where(fitness == np.min(fitness))
best_match = best_match_idx[0];
bestid = best_match[0];
conversion = np.zeros([np.size(new_population,0),genes])
for j in range(np.size(new_population,0)):
     conversion[j,:] = np.array([[int("".join(str(x) for x in new_population[j,0:nr_bit_num]), 2), int("".join(str(x) for x in new_population[j,nr_bit_num+1:nr_bit_num*2+1]), 2)]])
best = conversion[bestid,:]/(10**6)-65
print("Best solution : ", best)
print("Best solution fitness : ", fitness[bestid])
