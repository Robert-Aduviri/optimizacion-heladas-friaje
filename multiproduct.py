from functools import partial
import numpy as np

def random_partition(n, k, rand_generator):
    ''' Generate k values that sum to n '''
    p = []
    for i in range(k):
        v = rand_generator()
        p.append(min(n, v) if i < k-1 else n)
        n -= p[-1]
    return p
        
def generate_dataset(nodes, levels, total_supplies, total_demands, 
                     transp_costs, random_state=None):
    # Generate number of nodes per level
#     n_nodes = [nodes // levels] * (levels - 1) + \
#               [nodes % levels + nodes // levels]
    offset = int(0.25 * (nodes // levels) + 1)
    n_nodes = random_partition(nodes, levels, 
                    partial(np.random.randint, low=max(1,nodes // levels - offset),
                                              high=nodes // levels + offset))
    assert sum(n_nodes) == nodes # partition
    if random_state:
        np.random.seed(random_state)
    
    # Generate supplies for each item
    supplies = []
    for total_supply in total_supplies: 
        supplies.append(random_partition(
                total_supply, n_nodes[0], 
                partial(np.random.poisson, lam=total_supply // n_nodes[0])))
        assert sum(supplies[-1]) == total_supply # partition
    
    # Generate demands for each item
    demands = []
    for total_demand in total_demands:
        demands.append(random_partition(
                total_demand, n_nodes[-1],
                partial(np.random.poisson, lam=total_demand // n_nodes[-1])))
        assert sum(demands[-1]) == total_demand # partition
        
    # Generate costs
    costs = [np.random.randint(*transp_costs, (n_nodes[i], n_nodes[i+1])) \
                for i in range(levels-1)]
    
    # Capacities
    flow = max(sum(total_supplies), sum(total_demands))
    capacities = [list(np.sum(supplies, axis=0))] + \
                 [random_partition(flow + np.random.poisson(int(0.1 * flow)), n, 
                     partial(np.random.poisson, lam=flow // n)) \
                     for n in n_nodes[1:-1]] + \
                 [list(np.sum(demands, axis=0))]
    
    dummy_supplies = [max(0, d - s) for s,d in zip(total_supplies, total_demands)]
    dummy_demands = [max(0, s - d) for s,d in zip(total_supplies, total_demands)]
    
    # Supply-demand + dummies (supply-demand)
    supplies = [sup + [dum] for sup,dum in zip(supplies, dummy_supplies) ]
    demands = [dem + [dum] for dem,dum in zip(demands, dummy_demands)]

    return n_nodes, supplies, demands, costs, capacities