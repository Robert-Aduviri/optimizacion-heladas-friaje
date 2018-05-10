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

def init_genetic_algorithm(chromosome_fitness, E, D, W, d, t, a, c, g, v):
    del creator.FitnessMin
    del creator.Individual
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

######################## PLOT FUNCTIONS ########################

def plot_graph(b, f, q):
    G = nx.DiGraph()
    plt.figure(figsize=(10,5))

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
    
def plot_log(log):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot([x['gen'] for x in log],
            [x['Avg'] for x in log])
    ax.fill_between([x['gen'] for x in log],
                     [x['Avg'] - x['Std'] for x in log],
                     [x['Avg'] + x['Std'] for x in log],
                     alpha=0.3)