# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:25:59 2020

@author: TiberEmi
"""

import numpy
import ga
# Inputs of the equation.
equation_inputs = [4,-2,3.5,5,-11,-4.7]
 # Number of the weights we are looking to optimize.
num_weights = 6



sol_per_pop = 12
# Defining the population size.

pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.

#Creating the initial population.

new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)



num_generations = 700000

num_parents_mating = 4

for generation in range(num_generations):
     # Measuring the fitness of each chromosome in the population.
     fitness = ga.cal_pop_fitness(equation_inputs, new_population)
    # Selecting the best parents in the population for mating.
     parents = ga.select_mating_pool(new_population, fitness, 
                                       num_parents_mating)
 
     # Generating next generation using crossover.
     offspring_crossover = ga.crossover(parents,
                                        offspring_size=(pop_size[0]-parents.shape[0], num_weights))
 
     # Adding some variations to the offsrping using mutation.
     offspring_mutation = ga.mutation(offspring_crossover)
# Creating the new population based on the parents and offspring.
     new_population[0:parents.shape[0], :] = parents
     new_population[parents.shape[0]:, :] = offspring_mutation
         # The best result in the current iteration.

print("Best result : ", numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))

# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = ga.cal_pop_fitness(equation_inputs, new_population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])