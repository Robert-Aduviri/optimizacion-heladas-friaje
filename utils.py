import numpy as np

def transport(d, W, q, i, j, split_delivery):
    items = min(d[i], W[j])
    if split_delivery or not items < d[i]:
        W[j] -= items
        d[i] -= items
        q[j,i] += items
        if items > 0:
            print(f'Assign {items} from Plant {j+1} to Customer {i+1}')

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