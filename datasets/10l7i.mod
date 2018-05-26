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

param transp_cost := S1 T11 771 S1 T12 707 S1 T13 571 S2 T11 332 S2 T12 791 S2 T13 212  T11 D1 929 T11 D2 596 T11 D3 541 T11 D4 663 T11 D5 367 T12 D1 609 T12 D2 906 T12 D3 485 T12 D4 486 T12 D5 212 T13 D1 712 T13 D2 724 T13 D3 180 T13 D4 798 T13 D5 212;

param supply_demand (tr): S1 S2 T11 T12 T13 D1 D2 D3 D4 D5 dummy_sup dummy_dem := 
    I1 150 250 0 0 0 -140 -90 -130 -70 -70 100 0
    I2 130 270 0 0 0 -80 -100 -180 -80 -60 100 0
    I3 220 180 0 0 0 -90 -80 -70 -130 -130 100 0
    I4 200 200 0 0 0 -90 -120 -80 -70 -140 100 0
    I5 220 180 0 0 0 -100 -120 -110 -110 -60 100 0
    I6 290 110 0 0 0 -80 -100 -90 -80 -150 100 0
    I7 160 240 0 0 0 -80 -80 -100 -120 -120 100 0
;

param node_capacity := S1 1370 S2 1430 T11 1169 T12 1171 T13 1517 D1 660 D2 690 D3 760 D4 660 D5 730;

