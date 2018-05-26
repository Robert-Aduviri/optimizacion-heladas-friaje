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
set ST := S1 S2 S3 S4 S5 T11 T12 T13 T14 T15 T16 T21 T22 T23;
set D := D1 D2 D3 D4 D5 D6;
set DU := dummy_sup dummy_dem;

set E := 
   (S1,T11) (S1,T12) (S1,T13) (S1,T14) (S1,T15) (S1,T16) (S2,T11) (S2,T12) (S2,T13) (S2,T14) (S2,T15) (S2,T16) (S3,T11) (S3,T12) (S3,T13) (S3,T14) (S3,T15) (S3,T16) (S4,T11) (S4,T12) (S4,T13) (S4,T14) (S4,T15) (S4,T16) (S5,T11) (S5,T12) (S5,T13) (S5,T14) (S5,T15) (S5,T16)
   (T11,T21) (T11,T22) (T11,T23) (T12,T21) (T12,T22) (T12,T23) (T13,T21) (T13,T22) (T13,T23) (T14,T21) (T14,T22) (T14,T23) (T15,T21) (T15,T22) (T15,T23) (T16,T21) (T16,T22) (T16,T23)
   (T21,D1) (T21,D2) (T21,D3) (T21,D4) (T21,D5) (T21,D6) (T22,D1) (T22,D2) (T22,D3) (T22,D4) (T22,D5) (T22,D6) (T23,D1) (T23,D2) (T23,D3) (T23,D4) (T23,D5) (T23,D6)
;

set EDU := 
   (S1,T11) (S1,T12) (S1,T13) (S1,T14) (S1,T15) (S1,T16) (S2,T11) (S2,T12) (S2,T13) (S2,T14) (S2,T15) (S2,T16) (S3,T11) (S3,T12) (S3,T13) (S3,T14) (S3,T15) (S3,T16) (S4,T11) (S4,T12) (S4,T13) (S4,T14) (S4,T15) (S4,T16) (S5,T11) (S5,T12) (S5,T13) (S5,T14) (S5,T15) (S5,T16)
   (T11,T21) (T11,T22) (T11,T23) (T12,T21) (T12,T22) (T12,T23) (T13,T21) (T13,T22) (T13,T23) (T14,T21) (T14,T22) (T14,T23) (T15,T21) (T15,T22) (T15,T23) (T16,T21) (T16,T22) (T16,T23)
   (T21,D1) (T21,D2) (T21,D3) (T21,D4) (T21,D5) (T21,D6) (T22,D1) (T22,D2) (T22,D3) (T22,D4) (T22,D5) (T22,D6) (T23,D1) (T23,D2) (T23,D3) (T23,D4) (T23,D5) (T23,D6)
   (S1,dummy_dem) (S2,dummy_dem) (S3,dummy_dem) (S4,dummy_dem) (S5,dummy_dem)
   (dummy_sup,D1) (dummy_sup,D2) (dummy_sup,D3) (dummy_sup,D4) (dummy_sup,D5) (dummy_sup,D6)
;

param transp_cost := S1 T11 942 S1 T12 545 S1 T13 929 S1 T14 449 S1 T15 706 S1 T16 763 S2 T11 922 S2 T12 236 S2 T13 858 S2 T14 998 S2 T15 342 S2 T16 642 S3 T11 139 S3 T12 328 S3 T13 647 S3 T14 322 S3 T15 105 S3 T16 421 S4 T11 567 S4 T12 831 S4 T13 174 S4 T14 103 S4 T15 946 S4 T16 745 S5 T11 217 S5 T12 980 S5 T13 705 S5 T14 603 S5 T15 673 S5 T16 540 T11 T21 293 T11 T22 946 T11 T23 814 T12 T21 491 T12 T22 125 T12 T23 534 T13 T21 272 T13 T22 399 T13 T23 744 T14 T21 425 T14 T22 509 T14 T23 217 T15 T21 935 T15 T22 886 T15 T23 823 T16 T21 324 T16 T22 631 T16 T23 212 T21 D1 239 T21 D2 658 T21 D3 100 T21 D4 189 T21 D5 753 T21 D6 419 T22 D1 649 T22 D2 904 T22 D3 993 T22 D4 238 T22 D5 610 T22 D6 967 T23 D1 944 T23 D2 358 T23 D3 772 T23 D4 789 T23 D5 109 T23 D6 360;

param supply_demand (tr): S1 S2 S3 S4 S5 T11 T12 T13 T14 T15 T16 T21 T22 T23 D1 D2 D3 D4 D5 D6 dummy_sup dummy_dem := 
    I1 60 60 70 70 140 0 0 0 0 0 0 0 0 0 -60 -90 -50 -20 -80 -200 100 0
    I2 70 90 70 50 120 0 0 0 0 0 0 0 0 0 -60 -90 -30 -50 -130 -140 100 0
    I3 70 80 60 70 120 0 0 0 0 0 0 0 0 0 -100 -70 -60 -50 -90 -130 100 0
    I4 50 100 50 90 110 0 0 0 0 0 0 0 0 0 -40 -80 -90 -100 -80 -110 100 0
    I5 30 80 90 110 90 0 0 0 0 0 0 0 0 0 -60 -150 -80 -100 -50 -60 100 0
    I6 90 80 90 80 60 0 0 0 0 0 0 0 0 0 -90 -40 -70 -40 -70 -190 100 0
    I7 150 90 50 70 40 0 0 0 0 0 0 0 0 0 -70 -80 -40 -80 -80 -150 100 0
;

param node_capacity := S1 520 S2 580 S3 480 S4 540 S5 680 T11 565 T12 602 T13 581 T14 572 T15 551 T16 969 T21 1173 T22 1124 T23 1517 D1 480 D2 600 D3 420 D4 440 D5 580 D6 980;

