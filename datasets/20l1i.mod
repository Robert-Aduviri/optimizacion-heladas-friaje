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

param transp_cost := S1 T11 561 S1 T12 742 S1 T13 868 S1 T14 104 S1 T15 317 S1 T16 602 S2 T11 866 S2 T12 497 S2 T13 970 S2 T14 894 S2 T15 492 S2 T16 306 S3 T11 114 S3 T12 957 S3 T13 653 S3 T14 991 S3 T15 560 S3 T16 790 S4 T11 674 S4 T12 963 S4 T13 842 S4 T14 340 S4 T15 663 S4 T16 195 S5 T11 999 S5 T12 833 S5 T13 584 S5 T14 506 S5 T15 330 S5 T16 848 T11 T21 754 T11 T22 270 T11 T23 640 T12 T21 135 T12 T22 624 T12 T23 259 T13 T21 938 T13 T22 798 T13 T23 342 T14 T21 185 T14 T22 895 T14 T23 677 T15 T21 781 T15 T22 656 T15 T23 673 T16 T21 745 T16 T22 895 T16 T23 127 T21 D1 719 T21 D2 655 T21 D3 439 T21 D4 897 T21 D5 430 T21 D6 739 T22 D1 605 T22 D2 447 T22 D3 572 T22 D4 330 T22 D5 289 T22 D6 324 T23 D1 484 T23 D2 476 T23 D3 382 T23 D4 732 T23 D5 727 T23 D6 844;

param supply_demand (tr): S1 S2 S3 S4 S5 T11 T12 T13 T14 T15 T16 T21 T22 T23 D1 D2 D3 D4 D5 D6 dummy_sup dummy_dem := 
    I1 60 60 70 70 140 0 0 0 0 0 0 0 0 0 -70 -90 -70 -50 -90 -130 100 0
;

param node_capacity := S1 60 S2 60 S3 70 S4 70 S5 140 T11 84 T12 88 T13 76 T14 75 T15 80 T16 143 T21 163 T22 160 T23 231 D1 70 D2 90 D3 70 D4 50 D5 90 D6 130;

