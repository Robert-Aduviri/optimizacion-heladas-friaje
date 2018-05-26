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

param transp_cost := S1 T11 316 S1 T12 863 S1 T13 287 S2 T11 479 S2 T12 592 S2 T13 140  T11 D1 256 T11 D2 114 T11 D3 912 T11 D4 164 T11 D5 956 T12 D1 938 T12 D2 620 T12 D3 443 T12 D4 228 T12 D5 747 T13 D1 571 T13 D2 162 T13 D3 238 T13 D4 598 T13 D5 692;

param supply_demand (tr): S1 S2 T11 T12 T13 D1 D2 D3 D4 D5 dummy_sup dummy_dem := 
    I1 150 250 0 0 0 -100 -70 -110 -90 -130 100 0
    I2 130 270 0 0 0 -90 -70 -120 -140 -80 100 0
    I3 220 180 0 0 0 -130 -70 -90 -80 -130 100 0
;

param node_capacity := S1 500 S2 700 T11 507 T12 510 T13 641 D1 320 D2 210 D3 320 D4 310 D5 340;

