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
set I := I1;
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

param transp_cost := S1 T11 575 S1 T12 799 S1 T13 882 S2 T11 289 S2 T12 786 S2 T13 662  T11 D1 975 T11 D2 666 T11 D3 343 T11 D4 931 T11 D5 604 T12 D1 230 T12 D2 584 T12 D3 918 T12 D4 746 T12 D5 120 T13 D1 940 T13 D2 266 T13 D3 373 T13 D4 487 T13 D5 700;

param supply_demand (tr): S1 S2 T11 T12 T13 D1 D2 D3 D4 D5 dummy_sup dummy_dem := 
    I1 150 250 0 0 0 -50 -80 -110 -110 -150 100 0
;

param node_capacity := S1 150 S2 250 T11 173 T12 148 T13 225 D1 50 D2 80 D3 110 D4 110 D5 150;

