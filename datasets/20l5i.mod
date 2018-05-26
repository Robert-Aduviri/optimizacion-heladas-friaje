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

param transp_cost := S1 T11 771 S1 T12 272 S1 T13 928 S1 T14 914 S1 T15 248 S1 T16 179 S2 T11 985 S2 T12 312 S2 T13 302 S2 T14 863 S2 T15 328 S2 T16 775 S3 T11 326 S3 T12 758 S3 T13 631 S3 T14 540 S3 T15 501 S3 T16 146 S4 T11 332 S4 T12 404 S4 T13 625 S4 T14 242 S4 T15 514 S4 T16 612 S5 T11 472 S5 T12 665 S5 T13 985 S5 T14 358 S5 T15 755 S5 T16 570 T11 T21 111 T11 T22 429 T11 T23 835 T12 T21 883 T12 T22 457 T12 T23 507 T13 T21 767 T13 T22 472 T13 T23 107 T14 T21 221 T14 T22 447 T14 T23 775 T15 T21 189 T15 T22 747 T15 T23 797 T16 T21 415 T16 T22 277 T16 T23 639 T21 D1 831 T21 D2 968 T21 D3 140 T21 D4 839 T21 D5 803 T21 D6 601 T22 D1 244 T22 D2 300 T22 D3 823 T22 D4 560 T22 D5 831 T22 D6 851 T23 D1 657 T23 D2 646 T23 D3 352 T23 D4 489 T23 D5 693 T23 D6 982;

param supply_demand (tr): S1 S2 S3 S4 S5 T11 T12 T13 T14 T15 T16 T21 T22 T23 D1 D2 D3 D4 D5 D6 dummy_sup dummy_dem := 
    I1 60 60 70 70 140 0 0 0 0 0 0 0 0 0 -90 -80 -90 -80 -120 -40 100 0
    I2 70 90 70 50 120 0 0 0 0 0 0 0 0 0 -90 -50 -70 -90 -60 -140 100 0
    I3 70 80 60 70 120 0 0 0 0 0 0 0 0 0 -50 -20 -80 -80 -60 -210 100 0
    I4 50 100 50 90 110 0 0 0 0 0 0 0 0 0 -30 -50 -130 -120 -100 -70 100 0
    I5 30 80 90 110 90 0 0 0 0 0 0 0 0 0 -60 -50 -90 -60 -40 -200 100 0
;

param node_capacity := S1 280 S2 410 S3 340 S4 390 S5 580 T11 411 T12 442 T13 392 T14 409 T15 439 T16 668 T21 869 T22 837 T23 1033 D1 320 D2 250 D3 460 D4 430 D5 380 D6 660;

