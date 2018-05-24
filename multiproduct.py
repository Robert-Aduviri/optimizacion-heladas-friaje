from functools import partial
from itertools import product
import numpy as np

############################## DATASET ##############################

def random_partition(n, k, rand_generator):
    ''' Generate k values that sum to n '''
    p = []
    for i in range(k):
        v = rand_generator()
        p.append(min(n, v) if i < k-1 else n)
        n -= p[-1]
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
                partial(np.random.poisson, lam=reduced_total // nodes))])
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
    demand_limit: [K]
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
                         chrom, costs):
    '''
    I, J, K: int
    left_cap: [I, J] or [J]
    right_cap: [I, K] (no dummies)
    costs: [J, K]
    '''
    left = np.zeros((I, J), dtype=int)
    right = np.zeros((I, K), dtype=int)
    chrom = np.reshape(chrom, (I, K))    
    for i, k in sorted(product(range(I), range(K)), 
                      key=lambda x: chrom[x],
                      reverse=True):
        for j in sorted(range(J), 
                      key=lambda x: costs[x,k]):
            transport(left, right, left_cap, right_cap, X, i, j, k)
    return left

def dummy_transportation(I, K, left_cap, right_cap, chrom):
    '''
    I, K: int
    left_cap: [I]
    right_cap: [I,K]
    chrom: [K]
    '''
    chrom = np.reshape(chrom, (I,K))
    for i, k in sorted(product(range(I), range(K)),
                       key=lambda x: chrom[x],
                       reverse=True):
        available = min(left_cap[i], right_cap[i][k])
        left_cap[i] -= available
        right_cap[i][k] -= available
    return right_cap

def transportation_tree(chromosome, n_nodes, supplies, demands, costs, capacities):
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
            [x[-1] for x in supplies], right_cap, chrom_stages[-1])
    for i in range(len(n_nodes)-1):
        stage = len(n_nodes) - i - 2
        left_cap = capacities[stage] if stage > 0 else supplies
        right_cap = transportation_stage(n_items, n_nodes[stage],
                            n_nodes[stage+1], left_cap, right_cap, 
                            X[stage], chrom_stages[stage], costs[stage])
    return X