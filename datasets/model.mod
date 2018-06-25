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
set ST := S1 S2 T11 T12 T13 T21 T22;
set D := D1 D2 D3 D4 D5 D6 D7 D8;
set DU := dummy_sup dummy_dem;

set E := 
   (S1,T11) (S1,T12) (S1,T13) (S2,T11) (S2,T12) (S2,T13)
   (T11,T21) (T11,T22) (T12,T21) (T12,T22) (T13,T21) (T13,T22)
   (T21,D1) (T21,D2) (T21,D3) (T21,D4) (T21,D5) (T21,D6) (T21,D7) (T21,D8) (T22,D1) (T22,D2) (T22,D3) (T22,D4) (T22,D5) (T22,D6) (T22,D7) (T22,D8)
;

set EDU := 
   (S1,T11) (S1,T12) (S1,T13) (S2,T11) (S2,T12) (S2,T13)
   (T11,T21) (T11,T22) (T12,T21) (T12,T22) (T13,T21) (T13,T22)
   (T21,D1) (T21,D2) (T21,D3) (T21,D4) (T21,D5) (T21,D6) (T21,D7) (T21,D8) (T22,D1) (T22,D2) (T22,D3) (T22,D4) (T22,D5) (T22,D6) (T22,D7) (T22,D8)
   (S1,dummy_dem) (S2,dummy_dem)
   (dummy_sup,D1) (dummy_sup,D2) (dummy_sup,D3) (dummy_sup,D4) (dummy_sup,D5) (dummy_sup,D6) (dummy_sup,D7) (dummy_sup,D8)
;

param transp_cost := S1 T11 963.0 S1 T12 810.0 S1 T13 919.0 S2 T11 588.0 S2 T12 739.0 S2 T13 650.0 T11 T21 437.0 T11 T22 971.0 T12 T21 740.0 T12 T22 878.0 T13 T21 572.0 T13 T22 250.0 T21 D1 514.0 T21 D2 397.0 T21 D3 710.0 T21 D4 362.0 T21 D5 863.0 T21 D6 243.0 T21 D7 445.0 T21 D8 723.0 T22 D1 671.0 T22 D2 980.0 T22 D3 101.0 T22 D4 996.0 T22 D5 403.0 T22 D6 353.0 T22 D7 751.0 T22 D8 552.0;

param supply_demand (tr): S1 S2 T11 T12 T13 T21 T22 D1 D2 D3 D4 D5 D6 D7 D8 dummy_sup dummy_dem := 
    I1 230 170 0 0 0 0 0 -50 -50 -60 -40 -40 -90 -30 -140 100 0
    I2 210 190 0 0 0 0 0 -50 -60 -50 -100 -20 -50 -40 -130 100 0
    I3 150 250 0 0 0 0 0 -50 -40 -90 -40 -60 -80 -30 -110 100 0
;

param node_capacity := S1 590 S2 610 T11 489 T12 515 T13 665 T21 719 T22 936 D1 150 D2 150 D3 200 D4 180 D5 120 D6 220 D7 100 D8 380;

