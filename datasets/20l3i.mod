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

param transp_cost := S1 T11 937 S1 T12 882 S1 T13 153 S1 T14 543 S1 T15 712 S1 T16 363 S2 T11 152 S2 T12 671 S2 T13 719 S2 T14 104 S2 T15 202 S2 T16 295 S3 T11 873 S3 T12 976 S3 T13 983 S3 T14 449 S3 T15 146 S3 T16 966 S4 T11 922 S4 T12 919 S4 T13 755 S4 T14 368 S4 T15 469 S4 T16 735 S5 T11 205 S5 T12 769 S5 T13 758 S5 T14 756 S5 T15 219 S5 T16 930 T11 T21 886 T11 T22 703 T11 T23 157 T12 T21 445 T12 T22 840 T12 T23 573 T13 T21 216 T13 T22 929 T13 T23 890 T14 T21 226 T14 T22 492 T14 T23 740 T15 T21 157 T15 T22 733 T15 T23 612 T16 T21 850 T16 T22 901 T16 T23 195 T21 D1 737 T21 D2 217 T21 D3 659 T21 D4 700 T21 D5 587 T21 D6 336 T22 D1 984 T22 D2 996 T22 D3 371 T22 D4 288 T22 D5 803 T22 D6 546 T23 D1 680 T23 D2 889 T23 D3 960 T23 D4 346 T23 D5 175 T23 D6 253;

param supply_demand (tr): S1 S2 S3 S4 S5 T11 T12 T13 T14 T15 T16 T21 T22 T23 D1 D2 D3 D4 D5 D6 dummy_sup dummy_dem := 
    I1 60 60 70 70 140 0 0 0 0 0 0 0 0 0 -50 -100 -50 -90 -80 -130 100 0
    I2 70 90 70 50 120 0 0 0 0 0 0 0 0 0 -80 -90 -110 -50 -90 -80 100 0
    I3 70 80 60 70 120 0 0 0 0 0 0 0 0 0 -80 -80 -120 -150 -60 -10 100 0
;

param node_capacity := S1 200 S2 230 S3 200 S4 190 S5 380 T11 212 T12 269 T13 249 T14 250 T15 233 T16 432 T21 492 T22 500 T23 663 D1 210 D2 270 D3 280 D4 290 D5 230 D6 220;

