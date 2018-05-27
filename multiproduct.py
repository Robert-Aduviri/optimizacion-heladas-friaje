from functools import partial
from itertools import product
import copy
import random
import multiprocessing

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from deap import creator, base, tools, algorithms
sns.set()

############################## DATASET ##############################

def random_partition(n, k, rand_generator, min_val=0):
    ''' Generate k values that sum to n '''
    p = []
    for i in range(k):
        v = max(min_val, rand_generator())
        p.append(min(n, v) if i < k-1 else n)
        n -= p[-1]
    safe_idxs = [i for i,v in enumerate(p) if v - min_val >= min_val]
    bad_idxs = [i for i,v in enumerate(p) if v < min_val]
    for i in bad_idxs:
        p[i] += min_val
        p[safe_idxs[0]] -= min_val
        del safe_idxs[0]
    return p
        
def generate_n_nodes(nodes, levels):    
#     n_nodes = [nodes // levels] * (levels - 1) + \
#               [nodes % levels + nodes // levels]
    offset = int(0.25 * (nodes // levels) + 1)
    n_nodes = random_partition(nodes, levels, 
                    partial(np.random.randint, low=max(1,nodes // levels - offset),
                                              high=nodes // levels + offset))
    assert sum(n_nodes) == nodes # partition
    return n_nodes
    
def generate_supply_demand(total_items, nodes, factor=1):
    res = []
    for total_item in total_items: 
        reduced_total = total_item // factor
        res.append([factor * x for x in random_partition(
                reduced_total, nodes, 
                partial(np.random.poisson, lam=reduced_total // nodes), min_val=1)])
        assert sum(res[-1]) == total_item # partition
    return res
    
def generate_dataset(nodes, levels, total_supplies, total_demands, 
                     transp_costs, random_state=None):
    '''
    nodes: int
    levels: int
    total_supplies: [n_items]
    total_demands: [n_items]
    transp_costs: (low, high)
    '''
    if random_state:
        np.random.seed(random_state)
        
    # Generate number of nodes per level (n_levels)
    n_nodes = generate_n_nodes(nodes, levels)
    
    # Generate supplies for each item (n_items, n_nodes)
    supplies = generate_supply_demand(total_supplies, n_nodes[0], factor=10)
    
    # Generate demands for each item (n_items, n_nodes)
    demands = generate_supply_demand(total_demands, n_nodes[-1], factor=10)
    
    # Generate costs (n_levels-1, n_nodes[i], n_nodes[i+1])
    costs = [np.random.randint(*transp_costs, (n_nodes[i], n_nodes[i+1])) \
                for i in range(levels-1)]
    
    # Generate capacities (n_levels, n_nodes[i])
    max_flow = max(sum(total_supplies), sum(total_demands))
    min_flow = min(sum(total_supplies), sum(total_demands))
    capacities = [list(np.sum(supplies, axis=0))] + \
                 [random_partition(max_flow + np.random.poisson(int(0.1 * max_flow)), n, 
                     partial(np.random.poisson, lam=max_flow // n)) \
                     for n in n_nodes[1:-1]] + \
                 [list(np.sum(demands, axis=0))]
    for cap in capacities:
        assert min_flow <= sum(cap)
    
    dummy_supplies = [max(0, d - s) for s,d in zip(total_supplies, total_demands)]
    dummy_demands = [max(0, s - d) for s,d in zip(total_supplies, total_demands)]
    
    # Supply-demand + dummies (supply-demand) (n_items, n_nodes+1)
    supplies = [sup + [dum] for sup,dum in zip(supplies, dummy_supplies)]
    demands = [dem + [dum] for dem,dum in zip(demands, dummy_demands)]

    return n_nodes, supplies, demands, costs, capacities

############################## AMPL ##############################

def print_items(n_items, fout):
    items = ' '.join([f'I{i+1}' for i in range(n_items)])
    print(f'set I := {items};', file=fout)        
    
def print_nodes(n_nodes, fout):
    supply = ' '.join([f'S{i+1}' for i in range(n_nodes[0])])
    trans = ' '.join([f'T{i+1}{j+1}' for i in range(len(n_nodes[1:-1])) \
                                     for j in range(n_nodes[1:-1][i])])
    demand = ' '.join([f'D{i+1}' for i in range(n_nodes[-1])])
    print(f'set ST := {supply} {trans};', file=fout)
    print(f'set D := {demand};', file=fout)
    print(f'set DU := dummy_sup dummy_dem;\n', file=fout)

def print_edges(set_name, n_nodes, fout, dummies=False):
    print(f'set {set_name} := ', file=fout)
    supply_trans = ' '.join([f'(S{j+1},T1{k+1})' \
                             for j in range(n_nodes[0]) \
                             for k in range(n_nodes[1])])
    trans_trans = ' '.join([f'(T{i+1}{j+1},T{i+2}{k+1})' \
                             for i in range(len(n_nodes[1:-1])-1) \
                             for j in range(n_nodes[1:-1][i]) \
                             for k in range(n_nodes[1:-1][i+1])])
    trans_demand = ' '.join([f'(T{len(n_nodes[1:-1])}{j+1},D{k+1})' \
                             for j in range(n_nodes[-2]) \
                             for k in range(n_nodes[-1])])
    print(f'   {supply_trans}', file=fout)
    print(f'   {trans_trans}', file=fout)
    print(f'   {trans_demand}', file=fout)
    
    if dummies:
        dummy_supply = ' '.join(f'(S{j+1},dummy_dem)' \
                                 for j in range(n_nodes[0]))
        dummy_demand = ' '.join(f'(dummy_sup,D{j+1})' \
                                 for j in range(n_nodes[-1]))
        print(f'   {dummy_supply}', file=fout)
        print(f'   {dummy_demand}', file=fout)
        
    print(';\n', file=fout)
    
def print_transp_cost(n_nodes, costs, fout):
    supply_trans = ' '.join([f'S{j+1} T1{k+1} {costs[0][j][k]}' \
                             for j in range(n_nodes[0]) \
                             for k in range(n_nodes[1])])
    trans_trans = ' '.join([f'T{i+1}{j+1} T{i+2}{k+1} {costs[i+1][j][k]}' \
                             for i in range(len(n_nodes[1:-1])-1) \
                             for j in range(n_nodes[1:-1][i]) \
                             for k in range(n_nodes[1:-1][i+1])])
    trans_demand = ' '.join([f'T{len(n_nodes[1:-1])}{j+1} D{k+1} {costs[-1][j][k]}' \
                             for j in range(n_nodes[-2]) \
                             for k in range(n_nodes[-1])])    
    print(f'param transp_cost := {supply_trans} {trans_trans} {trans_demand};\n', 
                             file=fout)
    
def print_supply_demand(n_nodes, supplies, demands, fout):
    supply = ' '.join([f'S{i+1}' for i in range(n_nodes[0])])
    trans = ' '.join([f'T{i+1}{j+1}' for i in range(len(n_nodes[1:-1])) \
                                     for j in range(n_nodes[1:-1][i])])
    demand = ' '.join([f'D{i+1}' for i in range(n_nodes[-1])])
    print(f'param supply_demand (tr): {supply} {trans} {demand} '
           'dummy_sup dummy_dem := ', file=fout)
    for i in range(len(supplies)):
        supply_demand = ' '.join([str(x) for x in supplies[i][:-1] + \
                                 [0]*sum(n_nodes[1:-1]) + \
                                 [-x for x in demands[i][:-1]] + \
                                 [supplies[i][-1]] + [-demands[i][-1]]])
        print(f'    I{i+1} {supply_demand}', file=fout)
    print(';\n', file=fout)
    
def print_capacities(n_nodes, capacities, fout):
    supply = ' '.join([f'S{i+1} {capacities[0][i]}' \
                              for i in range(n_nodes[0])])
    trans = ' '.join([f'T{i+1}{j+1} {capacities[i+1][j]}' 
                              for i in range(len(n_nodes[1:-1])) \
                              for j in range(n_nodes[1:-1][i])])
    demand = ' '.join([f'D{i+1} {capacities[-1][i]}' 
                              for i in range(n_nodes[-1])])
    print(f'param node_capacity := {supply} {trans} {demand};\n', 
                              file=fout)
    
def generate_ampl(n_nodes, supplies, demands, costs, capacities, 
                  output_file='data/model_data.mod'):
    with open('data/model.mod', 'r') as f_model, \
         open(output_file, 'w') as fout:
        print(f_model.read(), file=fout)
        print('data;', file=fout)
        print_items(len(supplies), fout)
        print_nodes(n_nodes, fout)        
        print_edges('E', n_nodes, fout)
        print_edges('EDU', n_nodes, fout, dummies=True)
        print_transp_cost(n_nodes, costs, fout)
        print_supply_demand(n_nodes, supplies, demands, fout)
        print_capacities(n_nodes, capacities, fout)
        
############################## CHROMOSOME ##############################

def transport(left, right, left_cap, right_cap, X, 
              item, j, k, demand_limit=None):
    '''
    left: [I, J]
    right: [I, K]
    left_cap: [I, J] or [J]
    right_cap: [I, K]
    X: [I, J, K]
    demand_limit: [I,K]
    '''
#     print('caps:', left_cap, right_cap)
    item_cap = left_cap[item][j] if len(np.array(left_cap).shape)==2 \
                                 else left_cap[j]
    assert item_cap >= left[item][j] 
    assert right_cap[item][k] >= right[item][k] 
    available = min(item_cap - left[item][j], 
                    right_cap[item][k] - right[item][k])
    if demand_limit is not None: 
        available = min(available, 
                        int(demand_limit[item][k] * right_cap[item][k]))
    left[item][j] += available
    right[item][k] += available
    X[item][j][k] += available
    return available

def transportation_stage(I, J, K, left_cap, right_cap, X,
                         chrom, costs, chrom_demands=None):
    '''
    I, J, K: int
    left_cap: [I, J] or [J]
    right_cap: [I, K] (no dummies)
    costs: [J, K]
    chrom_demands: [I,K]
    '''
    left = np.zeros((I, J), dtype=int)
    right = np.zeros((I, K), dtype=int)
    chrom = np.reshape(chrom, (I, K))  
    if chrom_demands is not None: 
        chrom_demands = np.reshape(chrom_demands, (I,K))
    
    num_iter, max_iter = 0, 10
    to_transport = np.array(right_cap).sum()
    while to_transport > 0: 
        
        for i, k in sorted(product(range(I), range(K)), 
                          key=lambda x: chrom[x],
                          reverse=True):
            for j in sorted(range(J), 
                          key=lambda x: costs[x,k]):
                to_transport -= transport(left, right, left_cap, right_cap, X, i, j, k,
                          chrom_demands)
                
        if num_iter >= max_iter:
            chrom_demands = None # just fill partially for max_iter
        num_iter += 1
    return left

def dummy_transportation(I, K, left_cap, right_cap, chrom, chrom_demands=None):
    '''
    I, K: int
    left_cap: [I]
    right_cap: [I,K]
    chrom: [K]
    chrom_demands: [I,K]
    '''
    chrom = np.reshape(chrom, (I,K))
    if chrom_demands is not None: 
        chrom_demands = np.reshape(chrom_demands, (I,K))
    
    num_iter, max_iter = 0, 10
    to_transport = sum(left_cap)
    while to_transport > 0:
        
        for i, k in sorted(product(range(I), range(K)),
                           key=lambda x: chrom[x],
                           reverse=True):
            available = min(left_cap[i], right_cap[i][k])
            if chrom_demands is not None: 
                available = min(available, 
                                int(chrom_demands[i][k] * right_cap[i][k]))
            left_cap[i] -= available
            right_cap[i][k] -= available
            to_transport -= available
            
        if num_iter >= max_iter:
            chrom_demands = None # just fill partially for max_iter
        num_iter += 1
    return right_cap

def transportation_tree(chromosome, n_nodes, supplies, demands, costs, capacities,
                        chromosome_demands=None):
    n_items = len(demands)
    X = [np.zeros((n_items,
               n_nodes[i], 
               n_nodes[i+1])) \
         for i in range(len(n_nodes)-1)]
    # Divide chromosome in stages
    chrom_idxs = [0] + list(np.cumsum([n_items * x for x in n_nodes][1:]))
    chrom_stages = [chromosome[chrom_idxs[i]:chrom_idxs[i+1]] \
                    for i in range(len(n_nodes)-1)]
    # Dummy demand
    right_cap = [x[:-1] for x in demands]
    right_cap = dummy_transportation(n_items, n_nodes[-1],
            [x[-1] for x in supplies], right_cap, chrom_stages[-1], 
            chromosome_demands)
    for i in range(len(n_nodes)-1):
        stage = len(n_nodes) - i - 2
        left_cap = capacities[stage] if stage > 0 else supplies
        right_cap = transportation_stage(n_items, n_nodes[stage],
                            n_nodes[stage+1], left_cap, right_cap, 
                            X[stage], chrom_stages[stage], costs[stage],
                            chromosome_demands if i == 0 else None)
    return X

def chromosome_fitness(chromosome, n_nodes, supplies, demands,
                       costs, capacities, variable_demands=False, 
                       multiobjective=False):
    chrom_demands = None
    if variable_demands:
        n_items = len(supplies)
        permutation_size = sum([n_items * sum(n_nodes[1:])])
        chrom_demands = chromosome[permutation_size:]
        chromosome = chromosome[:permutation_size]
        
    X = transportation_tree(chromosome, n_nodes, supplies, demands, 
                        costs, capacities, chrom_demands)
    transportation_cost = sum(np.multiply(x.sum(0), c).sum() \
                               for c, x in zip(costs, X))
    fairness = (np.divide(X[-1].sum(1), np.array(demands)[:,:-1]) ** 2).sum()
    if multiobjective:
        return transportation_cost, fairness
    return transportation_cost + 10000 * fairness,

############################## GENETIC ALGORITHM ##############################

def create_chromosome(permutation_size, demand_size):
    perm = np.random.permutation(permutation_size).astype(int)
    demand = np.random.uniform(size=demand_size)
    return list(perm) + list(demand)

def custom_crossover(ind1, ind2, permutation_size):
    perm1 = ind1[:permutation_size]
    perm2 = ind2[:permutation_size]
    demand1 = ind1[permutation_size:]
    demand2 = ind2[permutation_size:]
    perm_child1, perm_child2 = tools.cxPartialyMatched(perm1, perm2)
    demand_child1, demand_child2 = tools.cxSimulatedBinaryBounded(
                                        demand1, demand2, 20, 0, 1)
    child1 = list(perm_child1) + list(demand_child1)
    child2 = list(perm_child2) + list(demand_child2)
    for i, x in enumerate(child1):
        ind1[i] = child1[i]
        ind2[i] = child2[i]
    return ind1, ind2

def custom_mutation(ind, permutation_size):
    perm = ind[:permutation_size]
    demand = ind[permutation_size:]
    perm_mut = tools.mutShuffleIndexes(perm, 2.0/len(perm))
    demand_mut = tools.mutPolynomialBounded(demand, 20, 0, 1, 1.0/len(demand))
    mut = list(perm_mut[0]) + list(demand_mut[0])
    for i, x in enumerate(mut):
        ind[i] = mut[i]
    return ind,

##### MULTI OBJECTIVE

def init_multiobjective_GA(n_nodes, supplies, demands, costs, 
                           capacities):
    try:
        del creator.FitnessMin
        del creator.Individual
    except:
        pass
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,-1.0))
    creator.create('Individual', list, fitness=creator.FitnessMin)
    
    n_items = len(supplies)
    permutation_size = sum([n_items * sum(n_nodes[1:])])
    demand_size = n_items * n_nodes[-1]
    
    toolbox = base.Toolbox()
    toolbox.register('chromosome', create_chromosome, 
                    permutation_size=permutation_size,
                    demand_size=demand_size)
    toolbox.register('individual', tools.initIterate, creator.Individual,
                    toolbox.chromosome)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    
    ga_fitness = partial(chromosome_fitness, 
                         n_nodes=n_nodes, supplies=supplies, 
                         demands=demands, costs=costs, 
                         capacities=capacities,
                         variable_demands=True,
                         multiobjective=True)
    
    toolbox.register('evaluate', ga_fitness)
    toolbox.register('mate', custom_crossover, 
                             permutation_size=permutation_size)
    toolbox.register('mutate', custom_mutation,
                             permutation_size=permutation_size)
    toolbox.register('select', tools.selNSGA2)
    
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)  
    
    stats = tools.Statistics()
    stats.register("pop", copy.deepcopy)
    
    return toolbox, stats

def run_multiobjective_GA(n_nodes, supplies, demands, costs, 
                          capacities, pop_size=100, n_generations=30, 
                          n_solutions=20, crossover_p=0.5, mutation_p=0.2):
    toolbox, stats = init_multiobjective_GA(n_nodes, supplies, 
                                            demands, costs, capacities)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(n_solutions)
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, 
                             mu=pop_size,
                             lambda_=pop_size,
                             cxpb=crossover_p, mutpb=mutation_p, ngen=n_generations, 
                             stats=stats, halloffame=hof, verbose=False)
    return pop, hof, log, toolbox

##### SINGLE OBJECTIVE



def init_singleobjective_GA(n_nodes, supplies, demands, costs, 
                           capacities):
    try:
        del creator.FitnessMin
        del creator.Individual
    except:
        pass
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)
    
    n_items = len(supplies)
    permutation_size = sum([n_items * sum(n_nodes[1:])])
    demand_size = n_items * n_nodes[-1]
    
    toolbox = base.Toolbox()
    toolbox.register('chromosome', create_chromosome, 
                    permutation_size=permutation_size,
                    demand_size=demand_size)
    toolbox.register('individual', tools.initIterate, creator.Individual,
                    toolbox.chromosome)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    
    ga_fitness = partial(chromosome_fitness, 
                         n_nodes=n_nodes, supplies=supplies, 
                         demands=demands, costs=costs, 
                         capacities=capacities,
                         variable_demands=True,
                         multiobjective=False)
    
    toolbox.register('evaluate', ga_fitness)
    toolbox.register('mate', custom_crossover, 
                             permutation_size=permutation_size)
    toolbox.register('mutate', custom_mutation,
                             permutation_size=permutation_size)
    toolbox.register('select', tools.selTournament, tournsize=20)
    
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)  
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('Avg', np.mean)
    stats.register('Std', np.std)
    stats.register('Min', np.min)
    stats.register('Max', np.max)
    
    return toolbox, stats, pool

def run_singleobjective_GA(n_nodes, supplies, demands, costs, 
                          capacities, pop_size=100, n_generations=30, 
                          n_solutions=20, crossover_p=0.5, mutation_p=0.2,
                          early_stopping_rounds=30, verbose=True):
    toolbox, stats, pool = init_singleobjective_GA(n_nodes, supplies, 
                                            demands, costs, capacities)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(n_solutions)
    pop, log = eaMuPlusLambdaEarlyStopping(pop, toolbox, 
                             mu=pop_size,
                             lambda_=pop_size,
                             cxpb=crossover_p, mutpb=mutation_p,
                             ngen=n_generations, stats=stats, 
                             halloffame=hof, verbose=verbose,
                             early_stopping_rounds=early_stopping_rounds)
    pool.close()
    return pop, hof, log

def eaMuPlusLambdaEarlyStopping(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__,
                   early_stopping_rounds=30):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    
    patience = early_stopping_rounds
    
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
            
        if gen > 1:
            if logbook[-1]['Avg'] < logbook[-2]['Avg']:
                patience = early_stopping_rounds
            else:
                patience -= 1
        if patience == 0:
            if verbose:
                print('Early stopping...')
            break

    return population, logbook

############################## PLOTTING ##############################

def plot_fronts(fronts, toolbox):
    plot_colors = sns.color_palette("Set1", n_colors=10)
    fig, ax = plt.subplots(1, figsize=(6,6))
    for i,inds in enumerate(fronts):
        par = [toolbox.evaluate(ind) for ind in inds]
        df = pd.DataFrame(par)
        df.plot(ax=ax, kind='scatter', label='Front ' + str(i+1), 
                     x=df.columns[0], y=df.columns[1], 
                     color=plot_colors[i % len(plot_colors)])
    plt.xlabel('$f_1(\mathbf{x})$');plt.ylabel('$f_2(\mathbf{x})$')

def animate(frame_index, logbook, toolbox, ax):
    plot_colors = sns.color_palette("Set1", n_colors=10)
    'Updates all plots to match frame _i_ of the animation.'
    ax.clear()    
    fronts = tools.emo.sortLogNondominated(logbook.select('pop')[frame_index], 
                                           len(logbook.select('pop')[frame_index]))
    for i,inds in enumerate(fronts):
        par = [toolbox.evaluate(ind) for ind in inds]
        df = pd.DataFrame(par)
        df.plot(ax=ax, kind='scatter', label='Front ' + str(i+1), 
                 x=df.columns[0], y=df.columns[1], alpha=0.47,
                 color=plot_colors[i % len(plot_colors)])
        
    ax.set_title('$t=$' + str(frame_index))
    ax.set_xlabel('$f_1(\mathbf{x})$');ax.set_ylabel('$f_2(\mathbf{x})$')
    return []

def get_animation(log, toolbox):
    fig = plt.figure(figsize=(6,6))
    ax = fig.gca()
    anim = animation.FuncAnimation(fig, lambda i: animate(i, log, toolbox, ax), 
                                   frames=len(log), interval=1000, 
                                   blit=True)
    plt.close()
    return anim