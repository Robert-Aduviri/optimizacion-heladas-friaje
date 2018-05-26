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

param transp_cost := S1 T11 725 S1 T12 592 S1 T13 174 S1 T14 512 S1 T15 475 S1 T16 519 S2 T11 828 S2 T12 376 S2 T13 860 S2 T14 775 S2 T15 493 S2 T16 968 S3 T11 556 S3 T12 291 S3 T13 838 S3 T14 788 S3 T15 198 S3 T16 647 S4 T11 195 S4 T12 763 S4 T13 762 S4 T14 289 S4 T15 835 S4 T16 136 S5 T11 879 S5 T12 468 S5 T13 794 S5 T14 624 S5 T15 378 S5 T16 316  T11 D1 966 T11 D2 972 T11 D3 897 T11 D4 372 T12 D1 980 T12 D2 161 T12 D3 695 T12 D4 979 T13 D1 828 T13 D2 441 T13 D3 496 T13 D4 798 T14 D1 118 T14 D2 276 T14 D3 711 T14 D4 495 T15 D1 544 T15 D2 332 T15 D3 175 T15 D4 364 T16 D1 554 T16 D2 895 T16 D3 817 T16 D4 834;

param supply_demand (tr): S1 S2 S3 S4 S5 T11 T12 T13 T14 T15 T16 D1 D2 D3 D4 dummy_sup dummy_dem := 
    I1 70 40 40 50 200 0 0 0 0 0 0 -110 -90 -60 -240 100 0
    I2 40 80 80 100 100 0 0 0 0 0 0 -180 -90 -130 -100 100 0
    I3 130 70 80 100 20 0 0 0 0 0 0 -140 -130 -70 -160 100 0
    I4 20 70 50 70 190 0 0 0 0 0 0 -80 -100 -140 -180 100 0
    I5 100 100 110 80 10 0 0 0 0 0 0 -90 -180 -120 -110 100 0
    I6 120 80 110 70 20 0 0 0 0 0 0 -140 -160 -90 -110 100 0
    I7 70 40 60 80 150 0 0 0 0 0 0 -100 -110 -100 -190 100 0
;

param node_capacity := S1 550 S2 480 S3 530 S4 550 S5 690 T11 592 T12 568 T13 595 T14 605 T15 544 T16 988 D1 840 D2 860 D3 710 D4 1090;

