set I; # items

set ST; # supply and transshipment nodes
set D; # demand nodes
set K := ST union D; # all nodes 
set E within {K, K}; # all edges

set DU; # dummy nodes
set KDU := K union DU; # all nodes + dummies
set EDU within {KDU,KDU}; # all edges + dummies

# param item_priority {I};
param transp_cost {E};
param supply_demand {KDU, I};
param node_capacity {K};

var X {EDU, I} >= 0 integer; # items to be transported

minimize Objectives: # item_priority[i] 
	(sum {(k,j) in E, i in I} transp_cost[k,j] * X[k,j,i]) + 
	10000*(sum {k in D, i in I} ((sum {(j,k) in E} X[j,k,i]) / -supply_demand[k,i]) ^ 2); # fairness
	
subject to Transportation_Balance {k in KDU, i in I}:
	sum {(k,j) in EDU} X[k,j,i] - sum {(j,k) in EDU} X[j,k,i] = supply_demand[k,i];

subject to Inbound_Capacity {k in K}:
	sum {(j,k) in E, i in I} X[j,k,i] <= node_capacity[k];
	
subject to Outbound_Capacity {k in K}:
	sum {(k,j) in E, i in I} X[k,j,i] <= node_capacity[k];
	
data;
set I := I1 I2;
set ST := S1 T11 T12 T21;
set D := D1 D2 D3 D4 D5 D6;
set DU := dummy_sup dummy_dem;

set E := 
   (S1,T11) (S1,T12)
   (T11,T21) (T12,T21)
   (T21,D1) (T21,D2) (T21,D3) (T21,D4) (T21,D5) (T21,D6)
;

set EDU := 
   (S1,T11) (S1,T12)
   (T11,T21) (T12,T21)
   (T21,D1) (T21,D2) (T21,D3) (T21,D4) (T21,D5) (T21,D6)
   (S1,dummy_dem)
   (dummy_sup,D1) (dummy_sup,D2) (dummy_sup,D3) (dummy_sup,D4) (dummy_sup,D5) (dummy_sup,D6)
;

param transp_cost := S1 T11 37 S1 T12 37 T11 T21 37 T12 T21 22 T21 D1 18 T21 D2 38 T21 D3 24 T21 D4 22 T21 D5 10 T21 D6 34;

param supply_demand (tr): S1 T11 T12 T21 D1 D2 D3 D4 D5 D6 dummy_supply dummy_demand := 
    I1 25 0 0 0 -4 -4 -4 -2 -5 -9 3 0
    I2 30 0 0 0 -5 -3 -5 -7 -1 -4 0 -5
;

param node_capacity := S1 55 T11 32 T12 26 T21 57 D1 9 D2 7 D3 9 D4 9 D5 6 D6 13;

