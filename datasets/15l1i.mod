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
set ST := S1 S2 S3 S4 S5 T11 T12 T13 T14 T15 T16;
set D := D1 D2 D3 D4;
set DU := dummy_sup dummy_dem;

set E := 
   (S1,T11) (S1,T12) (S1,T13) (S1,T14) (S1,T15) (S1,T16) (S2,T11) (S2,T12) (S2,T13) (S2,T14) (S2,T15) (S2,T16) (S3,T11) (S3,T12) (S3,T13) (S3,T14) (S3,T15) (S3,T16) (S4,T11) (S4,T12) (S4,T13) (S4,T14) (S4,T15) (S4,T16) (S5,T11) (S5,T12) (S5,T13) (S5,T14) (S5,T15) (S5,T16)
   
   (T11,D1) (T11,D2) (T11,D3) (T11,D4) (T12,D1) (T12,D2) (T12,D3) (T12,D4) (T13,D1) (T13,D2) (T13,D3) (T13,D4) (T14,D1) (T14,D2) (T14,D3) (T14,D4) (T15,D1) (T15,D2) (T15,D3) (T15,D4) (T16,D1) (T16,D2) (T16,D3) (T16,D4)
;

set EDU := 
   (S1,T11) (S1,T12) (S1,T13) (S1,T14) (S1,T15) (S1,T16) (S2,T11) (S2,T12) (S2,T13) (S2,T14) (S2,T15) (S2,T16) (S3,T11) (S3,T12) (S3,T13) (S3,T14) (S3,T15) (S3,T16) (S4,T11) (S4,T12) (S4,T13) (S4,T14) (S4,T15) (S4,T16) (S5,T11) (S5,T12) (S5,T13) (S5,T14) (S5,T15) (S5,T16)
   
   (T11,D1) (T11,D2) (T11,D3) (T11,D4) (T12,D1) (T12,D2) (T12,D3) (T12,D4) (T13,D1) (T13,D2) (T13,D3) (T13,D4) (T14,D1) (T14,D2) (T14,D3) (T14,D4) (T15,D1) (T15,D2) (T15,D3) (T15,D4) (T16,D1) (T16,D2) (T16,D3) (T16,D4)
   (S1,dummy_dem) (S2,dummy_dem) (S3,dummy_dem) (S4,dummy_dem) (S5,dummy_dem)
   (dummy_sup,D1) (dummy_sup,D2) (dummy_sup,D3) (dummy_sup,D4)
;

param transp_cost := S1 T11 180 S1 T12 661 S1 T13 971 S1 T14 487 S1 T15 101 S1 T16 489 S2 T11 665 S2 T12 205 S2 T13 871 S2 T14 921 S2 T15 576 S2 T16 802 S3 T11 501 S3 T12 829 S3 T13 655 S3 T14 261 S3 T15 301 S3 T16 369 S4 T11 962 S4 T12 915 S4 T13 370 S4 T14 555 S4 T15 561 S4 T16 826 S5 T11 351 S5 T12 801 S5 T13 395 S5 T14 824 S5 T15 819 S5 T16 848  T11 D1 437 T11 D2 978 T11 D3 152 T11 D4 891 T12 D1 316 T12 D2 863 T12 D3 287 T12 D4 479 T13 D1 592 T13 D2 140 T13 D3 256 T13 D4 114 T14 D1 912 T14 D2 164 T14 D3 956 T14 D4 938 T15 D1 620 T15 D2 443 T15 D3 228 T15 D4 747 T16 D1 571 T16 D2 162 T16 D3 238 T16 D4 598;

param supply_demand (tr): S1 S2 S3 S4 S5 T11 T12 T13 T14 T15 T16 D1 D2 D3 D4 dummy_sup dummy_dem := 
    I1 70 40 40 50 200 0 0 0 0 0 0 -110 -90 -140 -160 100 0
;

param node_capacity := S1 70 S2 40 S3 40 S4 50 S5 200 T11 91 T12 79 T13 94 T14 78 T15 78 T16 134 D1 110 D2 90 D3 140 D4 160;

