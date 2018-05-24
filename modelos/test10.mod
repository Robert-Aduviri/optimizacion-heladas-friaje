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
set I := I1 I2 I3;
set ST := S1 S2 T11 T12 T13;
set D := D1 D2 D3 D4 D5;
set DU := dummy_sup dummy_dem;

set E := 
   (S1,T11) (S1,T12) (S1,T13) (S2,T11) (S2,T12) (S2,T13)
   
   (T11,D1) (T11,D2) (T11,D3) (T11,D4) (T11,D5) (T12,D1) (T12,D2) (T12,D3) (T12,D4) (T12,D5) (T13,D1) (T13,D2) (T13,D3) (T13,D4) (T13,D5)
;

set EDU := 
   (S1,T11) (S1,T12) (S1,T13) (S2,T11) (S2,T12) (S2,T13)
   
   (T11,D1) (T11,D2) (T11,D3) (T11,D4) (T11,D5) (T12,D1) (T12,D2) (T12,D3) (T12,D4) (T12,D5) (T13,D1) (T13,D2) (T13,D3) (T13,D4) (T13,D5)
   (S1,dummy_dem) (S2,dummy_dem)
   (dummy_sup,D1) (dummy_sup,D2) (dummy_sup,D3) (dummy_sup,D4) (dummy_sup,D5)
;

param transp_cost := S1 T11 24 S1 T12 52 S1 T13 38 S2 T11 45 S2 T12 22 S2 T13 41  T11 D1 80 T11 D2 68 T11 D3 95 T11 D4 37 T11 D5 75 T12 D1 51 T12 D2 54 T12 D3 71 T12 D4 66 T12 D5 15 T13 D1 37 T13 D2 37 T13 D3 53 T13 D4 93 T13 D5 39;

param supply_demand (tr): S1 S2 T11 T12 T13 D1 D2 D3 D4 D5 dummy_sup dummy_dem := 
    I1 200 300 0 0 0 -10 -50 -90 -40 -210 0 -100
    I2 130 270 0 0 0 -110 -180 -100 -120 -90 200 0
    I3 170 130 0 0 0 -30 -40 -130 -70 -130 100 0
;

param node_capacity := S1 500 S2 700 T11 437 T12 500 T13 619 D1 150 D2 270 D3 320 D4 230 D5 430;

