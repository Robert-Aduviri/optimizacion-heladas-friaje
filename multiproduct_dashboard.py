from functools import partial
from itertools import product
from collections import namedtuple
import copy
import random
import multiprocessing

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
import networkx as nx
from deap import creator, base, tools, algorithms
sns.set()

import ipywidgets as widgets
from IPython.display import display, clear_output
from multiproduct import generate_ampl

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
    len_demand = max(1, len(demand))
    demand_mut = tools.mutPolynomialBounded(demand, 20, 0, 1, 1.0/len_demand)        
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
                         variable_demands=True, # True
                         multiobjective=True)
    
    toolbox.register('evaluate', ga_fitness)
    toolbox.register('mate', custom_crossover, 
                             permutation_size=permutation_size)
    toolbox.register('mutate', custom_mutation,
                             permutation_size=permutation_size)
    toolbox.register('select', tools.selNSGA2)
    
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)  
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('Avg', np.mean, axis=0)
    stats.register('Std', np.std, axis=0)
    stats.register('Min', np.min, axis=0)
    stats.register('Max', np.max, axis=0)
    
    # stats = tools.Statistics()
    # stats.register("pop", copy.deepcopy)
    
    return toolbox, stats, pool

def run_multiobjective_GA(n_nodes, supplies, demands, costs, 
                          capacities, pop_size=100, n_generations=30, 
                          n_solutions=20, crossover_p=0.5, mutation_p=0.2, 
                          early_stopping_rounds=None,
                          print_log=False, plot_pop=False, plot_log=False,
                          plot_fairness=False, log_output=None, 
                          pop_output=None, fairness_output=None):
    toolbox, stats, pool = init_multiobjective_GA(n_nodes, supplies, 
                                            demands, costs, capacities)
    
    Params = namedtuple('Params', ['nodes', 'supplies', 'demands', 'costs', 
                                   'capacities'])
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(n_solutions)
    pop, log = eaMuPlusLambdaEarlyStopping(pop, toolbox, 
                             mu=pop_size,
                             lambda_=pop_size,
                             cxpb=crossover_p, mutpb=mutation_p, ngen=n_generations, 
                             stats=stats, halloffame=hof, verbose=False,
                             plot_pop=plot_pop, print_log=print_log,
                             plot_log=plot_log, plot_fairness=plot_fairness,
                             early_stopping_rounds=early_stopping_rounds,
                             params=Params(n_nodes, supplies, demands, costs,
                                           capacities),
                             log_output=log_output, pop_output=pop_output, 
                             fairness_output=fairness_output)
    pool.close()
    pool.join()
    return pop, hof, log, toolbox

def plot_logbook(log, fig, ax, log_output, min_max=True, optimal_value=False):
    ax[0].clear()
    ax[1].clear()
    ax[0].plot([x['gen'] for x in log], [x['Avg'][0] for x in log], 
               label='f1 objective value')
    ax[1].plot([x['gen'] for x in log], [x['Avg'][1] for x in log],
               label='f2 objective value')
    
    if min_max:
        ax[0].fill_between([x['gen'] for x in log], [x['Min'][0] for x in log],
                           [x['Max'][0] for x in log], alpha=0.3)
        ax[1].fill_between([x['gen'] for x in log], [x['Min'][1] for x in log],
                           [x['Max'][1] for x in log], alpha=0.3)
    else:
        ax[0].fill_between([x['gen'] for x in log],
                           [x['Avg'][0] - x['Std'][0] for x in log],
                           [x['Avg'][0] + x['Std'][0] for x in log],
                           alpha=0.3)
        ax[1].fill_between([x['gen'] for x in log],
                           [x['Avg'][1] - x['Std'][1] for x in log],
                           [x['Avg'][1] + x['Std'][1] for x in log],
                           alpha=0.3)
        
    if optimal_value:
        ax.plot([x['gen'] for x in log],
                [optimal_value for x in log], dashes=[6,2],
                label='GLPK solver')
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel('# Generations')
    ax[0].set_ylabel('$f_1(\mathbf{x})$')
    ax[1].set_xlabel('# Generations')
    ax[1].set_ylabel('$f_2(\mathbf{x})$')   
    
    with log_output:
        clear_output(wait=True)
        display(fig)
        
def plot_fronts(pop, fig, ax, pop_output, gen):
    ax.clear()
    fronts = tools.emo.sortLogNondominated(pop, len(pop))
    plot_colors = sns.color_palette("Set1", n_colors=10)
    for i, inds in enumerate(fronts):
        par = [ind.fitness.values for ind in inds]
        df = pd.DataFrame(par)        
        label = f'Gen {gen} - Front {i+1}' if gen > 0 else None
        df.plot(ax=ax, kind='scatter', label=label, 
                     x=df.columns[1], y=df.columns[0], 
                     color=plot_colors[i % len(plot_colors)])
    ax.set_xlabel('$f_2(\mathbf{x})$')
    ax.set_ylabel('$f_1(\mathbf{x})$')
    ax.set_title(f'Population at generation {gen}')
    
    with pop_output:
        clear_output(wait=True)
        display(fig)
        
        
def eaMuPlusLambdaEarlyStopping(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__,
                   early_stopping_rounds=None, plot_pop=False, plot_log=False,
                   plot_fairness=False, print_log=False, params=None,
                   log_output=None, pop_output=None, fairness_output=None):
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
    
    ############ WIDGETS #############
    
    gen = 0
        
    if plot_log:
        plot_log_every = 1
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
        # log_output = widgets.Output()
        # display(log_output)
        plot_logbook(logbook, fig, ax, log_output)
    
    if plot_pop:
        plot_pop_every = 1
        fig_pop, ax_pop = plt.subplots(figsize=(5,5))
        # pop_output = widgets.Output()
        # display(pop_output)
        plot_fronts(population, fig_pop, ax_pop, pop_output, gen)
    
    if plot_fairness:
        assert params is not None
        plot_fairness_every = 1
        fig_fairness, ax_fairness = plt.subplots(figsize=(5,4))
        # fairness_output = widgets.Output()
        # display(fairness_output)
        plot_demands(population, params.nodes, params.supplies, params.demands,
                      params.costs, params.capacities, fig_fairness, ax_fairness,
                      fairness_output)
    
    # Main Plot Box
    if False and plot_pop and plot_fairness:
        hbox_layout = widgets.Layout(display='flex',
                    flex_flow='row',
                    align_items='center',
                    justify_content='space-around',
                    width='100%',
                    max_width='800px',
                    margin='0 auto')
        vbox_layout = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_items='center',
                    justify_content='space-around',
                    width='100%',
                    max_width='800px',
                    margin='0 auto')
        
        pop_fairness_box = widgets.HBox([pop_output, fairness_output], 
                                        layout=hbox_layout)
        plots_box = widgets.VBox([log_output, 
                                  pop_fairness_box], 
                                 layout=vbox_layout)
        display(plots_box)
    
    if print_log:
        print(f'Gen: {gen:>3} | Min f1(x): {record["Min"][0]:.2f} | Min f2(x): {record["Min"][1]:.2f}')
        
    
    ########## ITERATIONS ############
    
    if early_stopping_rounds is None:
        early_stopping_rounds = ngen
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
        
        if print_log:
            print(f'Gen: {gen:>3} | Min f1(x): {record["Min"][0]:.3f} | Min f2(x): {record["Min"][1]:.3f}')
            
        if plot_log and gen % plot_log_every == 0:
            plot_logbook(logbook, fig, ax, log_output)
            
        if plot_pop and gen % plot_pop_every == 0:
            plot_fronts(population, fig_pop, ax_pop, pop_output, gen)
        
        if plot_fairness and gen % plot_fairness_every == 0:
            assert params is not None
            current_sol = logbook[-1]['Min']
            last_sol = logbook[-2]['Min']
            if current_sol[1] < last_sol[1]:
                plot_demands(population, params.nodes, params.supplies, params.demands,
                          params.costs, params.capacities, fig_fairness, ax_fairness,
                          fairness_output)
        
        if gen > 1:
            current_sol = logbook[-1]['Min']
            last_sol = logbook[-2]['Min']
            if current_sol[0] < last_sol[0] or current_sol[1] < last_sol[1]:
                patience = early_stopping_rounds
            else:
                patience -= 1
        if patience == 0:
            if print_log or verbose:
                print('Early stopping...')
            break
    
    plt.close(fig)
    plt.close(fig_pop)
    return population, logbook

def plot_graph(X, figsize=(10,5)):
    # X: [stages, #items, source, destination]
    G = nx.DiGraph()
    plt.figure(figsize=figsize)
    plt.xticks([])
    plt.yticks([])
    v_space = max(s for x in X for s in x.shape[1:])

    x0 = np.transpose(X[0], (1,2,0)) # (src,dest,items)
    spacing = v_space / (x0.shape[0] + 1)
    for i in range(x0.shape[0]):         
        G.add_node(f'A{i+1}', pos=(1, v_space - spacing * (i+1)))
    
    for idx, x in enumerate(X): # for each stage
        x = np.transpose(x, (1,2,0)) # (src, dest, items)
        spacing = v_space / (x.shape[1] + 1)
        for i in range(x.shape[1]):
            s = chr(ord('B') + idx)
            G.add_node(f'{s}{i+1}', pos=(idx+2, v_space - spacing * (i+1)))

    for idx, x in enumerate(X):
        x = np.transpose(x, (1,2,0)) # (src, dest, items)
        s1 = chr(ord('A') + idx)
        s2 = chr(ord('A') + idx + 1)
        for i in range(len(x)):
            for j in range(len(x[i])):
                if sum(x[i][j]) > 0: # [items]
                    G.add_edge(f'{s1}{i+1}', f'{s2}{j+1}', transp=x[i][j])

    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = nx.get_edge_attributes(G, 'transp')
    nx.draw_networkx_nodes(G, pos, node_size=2000, edgecolors='black', 
                           node_color='white')
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                 label_pos=0.5)
    
def best_individual(pop, nodes, supplies, demands, costs, capacities, obj=1):
    assert obj in [1,2]
    best = pop[min(range(len(pop)), key=lambda i: pop[i].fitness.values[obj-1])]
    n_items = len(supplies)
    permutation_size = sum([n_items * sum(nodes[1:])])
    Solution = namedtuple('Solution', ['X', 'chromosome', 'fitness'])
    return Solution(transportation_tree(best[:permutation_size], nodes, 
                       supplies, demands, costs, capacities, 
                       best[permutation_size:]), best, best.fitness.values)    
    
def plot_demands(pop, nodes, supplies, demands, costs, capacities, fig, ax, 
                  fairness_output, obj=2):
    # obj 1: best cost | obj 2: best fairness
    ax.clear()
    best_fairness = best_individual(pop, nodes, supplies, demands, 
                                    costs, capacities, obj=obj)
    satisfied_demand = best_fairness.X[-1].sum(1)
    pending_demand = np.array(demands)[:,:-1]
    relative_satisfied_demand = np.divide(satisfied_demand,
                                          pending_demand).transpose(1,0).reshape(-1)
    plot_colors = sns.color_palette("Set1", n_colors=10)
    colors = [plot_colors[i%len(plot_colors)] for i in range(pending_demand.shape[1]) \
                                              for j in range(pending_demand.shape[0])]
    ax.bar(x=range(len(relative_satisfied_demand)),
            height=relative_satisfied_demand, color=colors)
    ax.set_xlabel('Item demand for each location')
    ax.set_ylabel('% satisfied demand')
    ax.set_title(f'Best individual ($f_2(x)$)')
    
    with fairness_output:
        clear_output(wait=True)
        display(fig)
        
###### EXPORT / LOAD DATASETS ######

def export_dataset(dataset, file_name):
    with open(f'datasets/evaluation/{file_name}', 'w') as f:
#         f = sys.stdout
        print('# ITEMS:', file=f)
        print('  ', len(dataset.supplies), file=f)
        print(file=f)
        
        print('# NODES:', file=f)
        print('  ', ' '.join([str(n) for n in dataset.nodes]), file=f)
        print(file=f)
        
        print('# SUPPLIES:', file=f)
        supplies = np.array(dataset.supplies).transpose(1,0)
        for i, sup in enumerate(supplies[:-1]): # without dummy
            print(f'  # A{i+1}:', file=f)
            print( '    ', ' '.join([str(s) for s in sup]), file=f)
        print(file=f)
        
        print('# DEMANDS:', file=f)
        demands = np.array(dataset.demands).transpose(1,0)
        for i, dem in enumerate(demands[:-1]): # without dummy
            ch = chr(ord('A') + len(dataset.nodes) - 1)
            print(f'  # {ch}{i+1}:', file=f)
            print( '    ', ' '.join([str(d) for d in dem]), file=f)
        print(file=f)
        
        print('# COSTS:', file=f)
        for i, cost in enumerate(dataset.costs):
            ch1 = chr(ord('A') + i)
            ch2 = chr(ord('A') + i + 1)
            print(f'  # {ch1} => {ch2}:', file=f)
            for j_cost in cost:
                print('    ', ' '.join([str(c) for c in j_cost]), file=f)
        print(file=f)
        
        print('# CAPACITIES:', file=f)
        for i, cap in enumerate(dataset.capacities):
            ch = chr(ord('A') + i)
            print(f'  # {ch}:', file=f)
            print( '    ', ' '.join([str(c) for c in cap]), file=f)
        print(file=f)
        
def read_dataset(file_name):
    with open(f'datasets/evaluation/{file_name}', 'r') as f:
        lines = [x.strip() for x in f.readlines() \
                           if '#' not in x and x.strip() != '']
        items = int(lines[0])
        nodes = [int(n) for n in lines[1].split()]
        supplies = [[int(s) for s in lines[i].split()] \
                            for i in range(2, 2 + nodes[0])]
        idx = 2 + nodes[0]
        demands = [[int(d) for d in lines[i].split()] \
                           for i in range(idx, idx + nodes[-1])]
        idx = 2 + nodes[0] + nodes[-1]
        costs = [np.array([[float(c) for c in lines[i].split()] \
                            for i in range(idx + sum(nodes[:stage]), 
                                           idx + sum(nodes[:stage]) + nodes[stage])]) \
                            for stage in range(len(nodes)-1)]
        idx = 2 + nodes[0] + nodes[-1] + sum(nodes[:len(nodes)-1])
        capacities = [[int(c) for c in lines[i].split()] \
                              for i in range(idx, idx + len(nodes))]
        
        # DUMMIES
        
        supplies = np.array(supplies).transpose(1,0)
        demands = np.array(demands).transpose(1,0)
        
        total_supplies = np.sum(supplies, 1)
        total_demands = np.sum(demands, 1)
        
        dummy_supplies = [max(0, d - s) for s,d in zip(total_supplies, total_demands)]
        dummy_demands = [max(0, s - d) for s,d in zip(total_supplies, total_demands)]
        
        supplies = [list(sup) + [dum] for sup,dum in zip(supplies, dummy_supplies)]
        demands = [list(dem) + [dum] for dem,dum in zip(demands, dummy_demands)]
       
    Dataset = namedtuple('Dataset', ['nodes', 'supplies', 'demands', 'costs', 'capacities'])
    return Dataset(nodes, supplies, demands, costs, capacities)