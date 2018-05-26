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

param transp_cost := S1 T11 809 S1 T12 515 S1 T13 346 S1 T14 935 S1 T15 538 S1 T16 302 S2 T11 283 S2 T12 222 S2 T13 500 S2 T14 866 S2 T15 393 S2 T16 379 S3 T11 936 S3 T12 983 S3 T13 709 S3 T14 297 S3 T15 610 S3 T16 851 S4 T11 243 S4 T12 708 S4 T13 300 S4 T14 223 S4 T15 286 S4 T16 425 S5 T11 563 S5 T12 448 S5 T13 870 S5 T14 759 S5 T15 863 S5 T16 502  T11 D1 445 T11 D2 610 T11 D3 246 T11 D4 247 T12 D1 963 T12 D2 810 T12 D3 919 T12 D4 588 T13 D1 739 T13 D2 650 T13 D3 437 T13 D4 971 T14 D1 740 T14 D2 878 T14 D3 572 T14 D4 250 T15 D1 514 T15 D2 397 T15 D3 710 T15 D4 362 T16 D1 863 T16 D2 243 T16 D3 445 T16 D4 723;

param supply_demand (tr): S1 S2 S3 S4 S5 T11 T12 T13 T14 T15 T16 D1 D2 D3 D4 dummy_sup dummy_dem := 
    I1 70 40 40 50 200 0 0 0 0 0 0 -140 -100 -80 -180 100 0
    I2 40 80 80 100 100 0 0 0 0 0 0 -90 -80 -130 -200 100 0
    I3 130 70 80 100 20 0 0 0 0 0 0 -90 -140 -130 -140 100 0
;

param node_capacity := S1 240 S2 190 S3 200 S4 250 S5 320 T11 237 T12 256 T13 243 T14 274 T15 264 T16 378 D1 320 D2 320 D3 340 D4 520;

