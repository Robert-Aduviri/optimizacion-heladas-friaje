import re
import random
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from deap import creator, base, tools, algorithms

######################## CHROMOSOME ########################

def transport(d, W, q, i, j, split_delivery):
    items = min(d[i], W[j])
    if split_delivery or not items < d[i]:
        W[j] -= items
        d[i] -= items
        q[j,i] += items
        # if items > 0:
        #     print(f'Assign {items} from Plant {j+1} to Customer {i+1}')

def transportation_tree(d, W, c, v, split_delivery=False, 
                                      priority='left'):
    '''
    d: customers demand [i]
    W: DCs capacity [j]
    c: DCs-customers transportation $ [j,i]
    v: chromosome substring [j]
    '''
    J = len(W) # num DCs
    I = len(d) # num customers
    
    q = np.zeros((J,I)) # DC to customer transportation units
    z = np.zeros(J) # used DCs flag
    if priority == 'left':
        for j in sorted(range(J), key=lambda x: v[x], reverse=True):
            for i in sorted(range(I), key=lambda x: c[j,x]):
                transport(d, W, q, i, j, split_delivery)
    else:
        for i in sorted(range(I), key=lambda x: v[x], reverse=True):
            for j in sorted(range(J), key=lambda x: c[x,i]):
                transport(d, W, q, i, j, split_delivery)  
    z = np.array(q.sum(1) > 0, dtype='int')
    return q, z

def decode_chromosome(ch, E, D, W, d, t, a, c):
    '''
    E: suppliers capacity [s]
    D: plants capacity [k]
    W: DCs capacity [j]
    d: customers demand [i]
    t: suppliers-plants transportation + purchasing $ [s,k]
    a: plants-DCs transportation $ [k,j]
    c: DCs-customers transportation $ [j,i]
    '''
    k = len(D)
    q, z = transportation_tree(    d[:], W[:], c[:], ch[k:])
    f, p = transportation_tree(q.sum(1), D[:], a[:], ch[:k], split_delivery=True)
    b, s = transportation_tree(f.sum(1), E[:], t[:], ch[:k], split_delivery=True, priority='right')
    return q, z, f, p, b, s

def evaluate_decoding(b, t, p, g, f, a, z, v, q, c):
    return np.multiply(b, t).sum() + \
           np.multiply(p, g).sum() + \
           np.multiply(f, a).sum() + \
           np.multiply(z, v).sum() + \
           np.multiply(q, c).sum()    

def chromosome_fitness(ch, E, D, W, d, t, a, c, g, v):
    q, z, f, p, b, s = decode_chromosome(ch, E, D, W, d, t, a, c)
    return evaluate_decoding(b, t, p, g, f, a, z, v, q, c)


######################## GENETIC ALGORITHM ########################

def init_genetic_algorithm(chromosome_fitness, E, D, W, d, g, v, t, a, c):
    try:
        del creator.FitnessMin
        del creator.Individual
    except:
        pass
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)
    N_NODES = len(D) + len(W) # num plants + num DCs
    toolbox = base.Toolbox()
    toolbox.register('permutation', random.sample, range(N_NODES), N_NODES)
    toolbox.register('individual', tools.initIterate, creator.Individual,
                     toolbox.permutation)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    ga_fitness = lambda x: (partial(chromosome_fitness, E=E, 
                            D=D, W=W, d=d, t=t, a=a, c=c, g=g, v=v)(x),)
    toolbox.register('evaluate', ga_fitness)
    toolbox.register('mate', tools.cxPartialyMatched)
    toolbox.register('mutate', tools.mutShuffleIndexes, indpb=2.0/N_NODES)
    toolbox.register('select', tools.selTournament, tournsize=3)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('Avg', np.mean)
    stats.register('Std', np.std)
    stats.register('Min', np.min)
    stats.register('Max', np.max)
    return toolbox, stats

def run_genetic_algorithm(E, D, W, d, g, v, t, a, c, 
                          pop_size=100, n_generations=30, n_solutions=5,
                          crossover_p=0.5, mutation_p=0.2):
    toolbox, stats = init_genetic_algorithm(chromosome_fitness, E, D, W, d, 
                                            g, v, t, a, c)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(n_solutions)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=crossover_p, mutpb=mutation_p,
                              ngen=n_generations, stats=stats, halloffame=hof,
                              verbose=True)
    return pop, hof, log
    
######################## LOAD AND GENERATE FILES ########################

def load_file(input_file):
    '''
    Outputs: [E, D, W, d, g, v, t, a, c]
    '''
    with open(input_file, 'r') as f:
        lines = f.read().strip().split('\n')
        var_order = 'EDWdgvtac'
        params = {}
        for i in range(6):
            params[var_order[i]] = [int(n) for n in lines[i].split()[1:]]
        idx = 6
        for i in range(3):
            n_rows = len(params[var_order[i]])
            params[lines[idx]] = np.array(
                            [line.split() for line in \
                            lines[idx+1 : idx+1+n_rows]],
                            dtype=float)
            idx += 1 + n_rows
    return [params[var] for var in var_order]

def generate_file(output_file, 
                  n_S, S_cap, n_P, P_cap, 
                  n_D, D_cap, n_C, C_dem, 
                  P_cost, D_cost, 
                  SP_cost, PD_cost, DC_cost, random_state=None):
    '''
    output_file: path to the output file
    n_S: number of suppliers
    S_cap: capacity range of suppliers (S_cap_min, S_cap_max)
    n_P: number of plants
    P_cap: capacity range of plants (P_cap_min, P_cap_max)
    n_D: number of DCs
    D_cap: capacity range of DCs (D_cap_min, D_cap_max)
    n_C: number of customers
    C_dem: demand range of customers (C_dem_min, D_dem_max)
    P_cost: operation cost range of plants (P_cost_min, P_cost_max)
    D_cost: operation cost range of DCs (D_cost_min, D_cost_max)
    SP_cost: transportation cost from suppliers to plants (SP_cost_min, SP_cost_max)
    PD_cost: transportation cost from plants to DCs (PD_cost_min, PD_cost_max)
    DC_cost: transportation cost from DCs to customers (DC_cost_min, DC_cost_max)
    random_state: initial random seed
    '''
    if random_state:
        np.random.seed(random_state)
    with open(output_file, 'w') as f:
        var_order = 'EDWdgvtac'
        vals_order = [(n_S, S_cap), (n_P, P_cap), (n_D, D_cap), 
                      (n_C, C_dem), (n_P, P_cost), (n_D, D_cost),
                      SP_cost, PD_cost, DC_cost]
        for i in range(6):
            print(' '.join([var_order[i]] + [str(n) for n in \
                10 * np.random.randint(vals_order[i][1][0]//10, 
                                       vals_order[i][1][1]//10, 
                                       vals_order[i][0])]), file=f)
        for i in range(3):
            print(var_order[6+i], file=f)
            for j in range(vals_order[i][0]):
                print(' '.join([str(n) for n in \
                        np.random.randint(vals_order[6+i][0],
                                          vals_order[6+i][1],
                                          vals_order[i+1][0])]), file=f)

def generate_ampl(data_file, output_file, ampl_model='modelos/priority_encoding.mod'):
    with open(ampl_model, 'r') as f_model, \
         open(output_file, 'w') as f_output:
        lines = f_model.read().strip().split('\n')
        idx = [i for i, val in enumerate(lines) if 'data;' in val][0]
        for line in lines[:idx+2]:
            print(line, file=f_output)
        E, D, W, d, g, v, t, a, c = load_file(data_file)
        values = [E, D, W, d, g, v, t, a, c]
        sets = 'SKJI'
        params = 'EDWd'
        for i in range(4):
            elements = [f'{sets[i]}{j+1}' for j in range(len(values[i]))]
            print(f'set {sets[i]} := {" ".join(elements)};', file=f_output)
        print(file=f_output)
        
        for i in range(4):
            elements = [f'{sets[i]}{j+1} {values[i][j]}' for j in range(len(values[i]))]
            print(f'param {params[i]} := {" ".join(elements)};', file=f_output)
        print(file=f_output)
        
        params = 'gv'
        for i in range(2):
            elements = [f'{sets[i+1]}{j+1} {values[i+4][j]}' for j in range(len(values[i+1]))]
            print(f'param {params[i]} := {" ".join(elements)};', file=f_output)
        print(file=f_output)
        
        params = 'tac'
        for i in range(3):
            elements = [f'{sets[i+1]}{j+1}' for j in range(len(values[i+1]))]
            print(f'param {params[i]}: {" ".join(elements)} :=', file=f_output)
            for j in range(len(values[i])):
                elements = [str(n) for n in values[i+6][j]]
                print(f'      {sets[i]}{j+1} {" ".join(elements)}', file=f_output)
            print(';\n', file=f_output)
        
        idx = [i for i, val in enumerate(lines) if 'P_max :=' in val][0]
        for line in lines[idx:]:
            line = re.sub('P_max := \d', f'P_max := {len(D)}', line)
            line = re.sub('W_max := \d', f'W_max := {len(W)}', line)
            print(line, file=f_output)
                
######################## PLOT FUNCTIONS ########################

def plot_graph(b, f, q, figsize=(10,5)):
    G = nx.DiGraph()
    plt.figure(figsize=figsize)

    v_space = max(b.shape + f.shape + q.shape)

    for i in range(b.shape[0]):
        G.add_node(f'Sup{i+1}', pos=(1, v_space - (v_space-b.shape[0]+1)/2 - i))

    for i in range(b.shape[1]):
        G.add_node(f'Pla{i+1}', pos=(2, v_space - (v_space-b.shape[1]+1)/2 - i))

    for i in range(f.shape[1]):
        G.add_node(f'DC{i+1}' , pos=(3, v_space - (v_space-f.shape[1]+1)/2 - i))

    for i in range(q.shape[1]):
        G.add_node(f'Cus{i+1}', pos=(4, v_space - (v_space-q.shape[1]+1)/2 - i))

    for i in range(len(b)):
        for j in range(len(b[i])):
            if b[i][j] > 0:
                G.add_edge(f'Sup{i+1}', f'Pla{j+1}', transp=b[i][j])

    for i in range(len(f)):
        for j in range(len(f[i])):
            if f[i][j] > 0:
                G.add_edge(f'Pla{i+1}', f'DC{j+1}', transp=f[i][j])

    for i in range(len(q)):
        for j in range(len(q[i])):
            if q[i][j] > 0:
                G.add_edge(f'DC{i+1}', f'Cus{j+1}', transp=q[i][j])

    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = nx.get_edge_attributes(G, 'transp')
    nx.draw_networkx_nodes(G, pos, node_size=2000, edgecolors='black', 
                           node_color='white')
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                 label_pos=0.5)
    
def plot_log(log, optimal_value=None, min_max=False):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot([x['gen'] for x in log],
            [x['Avg'] for x in log], label='Genetic Algorithm')
    if min_max:
        ax.fill_between([x['gen'] for x in log],
                     [x['Min'] for x in log],
                     [x['Max'] for x in log],
                     alpha=0.3)
    else:
        ax.fill_between([x['gen'] for x in log],
                         [x['Avg'] - x['Std'] for x in log],
                         [x['Avg'] + x['Std'] for x in log],
                         alpha=0.3)
    if optimal_value:
        ax.plot([x['gen'] for x in log],
                [optimal_value for x in log], dashes=[6,2],
                label='GLPK solver')
    ax.legend()
    ax.set_xlabel('# Generations')
    ax.set_ylabel('Objective value')
    return ax