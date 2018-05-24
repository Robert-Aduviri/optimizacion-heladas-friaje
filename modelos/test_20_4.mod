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

param transp_cost := S1 T11 98 S1 T12 59 S1 T13 32 S1 T14 40 S1 T15 51 S1 T16 16 S2 T11 25 S2 T12 99 S2 T13 69 S2 T14 11 S2 T15 10 S2 T16 57 S3 T11 21 S3 T12 78 S3 T13 46 S3 T14 41 S3 T15 18 S3 T16 28 S4 T11 57 S4 T12 89 S4 T13 12 S4 T14 29 S4 T15 33 S4 T16 63 S5 T11 42 S5 T12 33 S5 T13 84 S5 T14 81 S5 T15 45 S5 T16 47 T11 T21 93 T11 T22 98 T11 T23 34 T12 T21 27 T12 T22 91 T12 T23 75 T13 T21 63 T13 T22 44 T13 T23 89 T14 T21 70 T14 T22 50 T14 T23 42 T15 T21 77 T15 T22 42 T15 T23 23 T16 T21 30 T16 T22 57 T16 T23 29 T21 D1 17 T21 D2 16 T21 D3 76 T21 D4 26 T21 D5 42 T21 D6 57 T22 D1 85 T22 D2 68 T22 D3 95 T22 D4 31 T22 D5 39 T22 D6 47 T23 D1 60 T23 D2 63 T23 D3 17 T23 D4 36 T23 D5 36 T23 D6 30;

param supply_demand (tr): S1 S2 S3 S4 S5 T11 T12 T13 T14 T15 T16 T21 T22 T23 D1 D2 D3 D4 D5 D6 dummy_sup dummy_dem := 
    I1 120 60 110 140 70 0 0 0 0 0 0 0 0 0 -100 -20 -50 -40 -110 -80 0 -100
    I2 70 70 80 70 110 0 0 0 0 0 0 0 0 0 -100 -70 -80 -120 -70 -160 200 0
    I3 60 50 30 30 130 0 0 0 0 0 0 0 0 0 -30 -60 -80 -30 -40 -160 100 0
;

param node_capacity := S1 250 S2 180 S3 220 S4 240 S5 310 T11 233 T12 243 T13 227 T14 261 T15 251 T16 325 T21 469 T22 479 T23 585 D1 230 D2 150 D3 210 D4 190 D5 220 D6 400;

