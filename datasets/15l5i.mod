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
set I := I1 I2 I3 I4 I5;
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

param transp_cost := S1 T11 203 S1 T12 492 S1 T13 910 S1 T14 345 S1 T15 275 S1 T16 138 S2 T11 576 S2 T12 781 S2 T13 858 S2 T14 637 S2 T15 966 S2 T16 917 S3 T11 507 S3 T12 624 S3 T13 927 S3 T14 605 S3 T15 924 S3 T16 135 S4 T11 784 S4 T12 119 S4 T13 420 S4 T14 875 S4 T15 611 S4 T16 499 S5 T11 753 S5 T12 982 S5 T13 570 S5 T14 242 S5 T15 191 S5 T16 453  T11 D1 933 T11 D2 899 T11 D3 826 T11 D4 953 T12 D1 150 T12 D2 764 T12 D3 797 T12 D4 674 T13 D1 289 T13 D2 224 T13 D3 249 T13 D4 413 T14 D1 669 T14 D2 441 T14 D3 404 T14 D4 791 T15 D1 781 T15 D2 937 T15 D3 882 T15 D4 153 T16 D1 543 T16 D2 712 T16 D3 363 T16 D4 152;

param supply_demand (tr): S1 S2 S3 S4 S5 T11 T12 T13 T14 T15 T16 D1 D2 D3 D4 dummy_sup dummy_dem := 
    I1 70 40 40 50 200 0 0 0 0 0 0 -110 -140 -120 -130 100 0
    I2 40 80 80 100 100 0 0 0 0 0 0 -120 -110 -130 -140 100 0
    I3 130 70 80 100 20 0 0 0 0 0 0 -120 -150 -100 -130 100 0
    I4 20 70 50 70 190 0 0 0 0 0 0 -60 -80 -120 -240 100 0
    I5 100 100 110 80 10 0 0 0 0 0 0 -180 -90 -100 -130 100 0
;

param node_capacity := S1 360 S2 360 S3 360 S4 400 S5 520 T11 396 T12 428 T13 429 T14 410 T15 431 T16 654 D1 590 D2 570 D3 570 D4 770;

