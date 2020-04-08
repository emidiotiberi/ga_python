
import numpy as np

def cal_pop_fitness(equation_inputs, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    fitness = np.sum(pop*equation_inputs, axis=1)
    return fitness

def foxholes(pop,genes,nr_bit_num):
#    x1min=-65.536
#    x1max=65.536
#    x2min=-65.536
#    x2max=65.536
#    R=1500 # steps resolution
#    x1=np.linspace(x1min,x1max,R)
#    x2=np.linspace(x2min,x2max,R)
        
    a=np.array([[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],[-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]]) 
        
        # initializing list  
        #pop = [[1, 0, 0, 1, 1, 0],
        #       [0, 0, 1, 0, 1, 0]]
        
          
        # using join() + list comprehension 
        # converting binary list to integer
    X = np.zeros([np.size(pop,0),genes])
    for j in range(np.size(pop,0)):
        X[j,:] = np.array([[int("".join(str(x) for x in pop[j,0:nr_bit_num]), 2), int("".join(str(x) for x in pop[j,nr_bit_num:nr_bit_num*2]), 2)]])
     
                 
    X = X*(10**(-6))-65.0
          
    Fk = np.zeros([np.size(pop,0),25])  
    Fs = np.zeros(np.size(pop,0))
    f = np.zeros(np.size(pop,0))
        
    for i in range(np.size(pop,0)):
                             
        for k in range(25):
            Fk[i,k] = (k+1+((X[i,0])-a[0,k])**6+((X[i,1])-a[1,k])**6)**(-1)
     
       
        Fs[i]=np.sum(Fk[i,:],axis=0)
        
        f[i]=((500**(-1))+Fs[i])**(-1)
        
    
    
    return f



def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents,n_cross_point, offspring_size):
    offspring_cross = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually it is at the center.
#    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        
        breakpoints = np.random.permutation(parents.shape[1]-1)
        breakpoints = np.sort(breakpoints[:n_cross_point])

        offspring_cross[k,:] = np.concatenate((parents[parent1_idx,:breakpoints[0]], parents[parent2_idx,breakpoints[0]:]),axis=0)
        for j in np.arange(1,n_cross_point):
            if j%2:
                offspring_cross[k,:] = np.concatenate((offspring_cross[k,:breakpoints[j]], parents[parent2_idx,breakpoints[j]:]),axis=0)
            else:
                offspring_cross[k,:] = np.concatenate((offspring_cross[k,:breakpoints[j]], parents[parent1_idx,breakpoints[j]:]),axis=0)
        
    
#        # The new offspring will have its first half of its genes taken from the first parent.
#        offspring_cross[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
#        # The new offspring will have its second half of its genes taken from the second parent.
#        offspring_cross[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring_cross

def mutation(offspring_mutation_num,new_population):
    # Mutation changes a single gene in each offspring randomly.
    offspring_mutation = new_population[new_population.shape[0]-offspring_mutation_num:,:]
    for idx in range(offspring_mutation.shape[0]):
        # The random value to be added to the gene.
        random_value = np.random.randint(0, 2, 1)
        random_position = np.random.randint(0, offspring_mutation.shape[1], 1)
        offspring_mutation[idx, random_position] = random_value
    return offspring_mutation
