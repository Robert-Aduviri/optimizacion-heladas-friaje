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
set I := I1 I2 I3 I4 I5 I6 I7;
set ST := S1 T11;
set D := D1 D2 D3;
set DU := dummy_sup dummy_dem;

set E := 
   (S1,T11)
   
   (T11,D1) (T11,D2) (T11,D3)
;

set EDU := 
   (S1,T11)
   
   (T11,D1) (T11,D2) (T11,D3)
   (S1,dummy_dem)
   (dummy_sup,D1) (dummy_sup,D2) (dummy_sup,D3)
;

param transp_cost := S1 T11 388  T11 D1 478 T11 D2 872 T11 D3 589;

param supply_demand (tr): S1 T11 D1 D2 D3 dummy_sup dummy_dem := 
    I1 400 0 -150 -170 -180 100 0
    I2 400 0 -150 -120 -230 100 0
    I3 400 0 -170 -130 -200 100 0
    I4 400 0 -100 -130 -270 100 0
    I5 400 0 -160 -90 -250 100 0
    I6 400 0 -150 -200 -150 100 0
    I7 400 0 -110 -190 -200 100 0
;

param node_capacity := S1 2800 T11 3842 D1 990 D2 1030 D3 1480;

