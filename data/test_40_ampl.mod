set S; # suppliers
set K; # plants
set J; # DCs
set I; # customers

param E {S}; # supplier capacity
param D {K}; # plant capacity
param W {J}; # DC capacity
param d {I}; # demand

param P_max; # max open plants
param W_max; # max open DCs
param u; # utilization rate

param g {K}; # plant operation cost
param v {J}; # DC operation cost
param t {S,K}; # supplier-plant purchasing + transportation cost
param a {K,J}; # plant-DC transportation cost
param c {J,I}; # DC-customer transportation cost

var b {S,K} >= 0 integer; # items shipped from supplier to plant
var f {K,J} >= 0 integer; # items shipped from plant to DC
var q {J,I} >= 0 integer; # items shipped from DC to customer

var p {K} binary; # open plants
var z {J} binary; # open DCs
var y {J,I} binary; # DC-customer flag

minimize Cost:
	sum {k in K} g[k] * p[k] + 
	sum {j in J} v[j] * z[j] +
	sum {s in S} sum {k in K} t[s,k] * b[s,k] +
	sum {k in K} sum {j in J} a[k,j] * f[k,j] +
	sum {j in J} sum {i in I} c[j,i] * q[j,i];
	
subject to Customer_Allocation {i in I}:
	sum {j in J} y[j,i] = 1;

subject to DCs_Capacity {j in J}:
	sum {i in I} d[i] * y[j,i] <= W[j] * z[j];

subject to Max_DCs:
	sum {j in J} z[j] <= W_max;
	
subject to DC_Customer_Connection {(j,i) in {J,I}}: 
	q[j,i] = d[i] * y[j,i];
	
subject to Supplier_Plant_Flow_Conservation {k in K}:
	u * sum {j in J} f[k,j] <= sum {s in S} b[s,k];
	
subject to Plant_DC_Flow_Conservation {j in J}:
	sum {k in K} f[k,j] = sum {i in I} q[j,i];
	
subject to Supplier_Capacity {s in S}:
	sum {k in K} b[s,k] <= E[s];
	
subject to Plant_Production_Capacity {k in K}:
	u * sum {j in J} f[k,j] <= D[k] * p[k];
	
subject to Max_Plants:
	sum {k in K} p[k] <= P_max;
	
data;

set S := S1 S2 S3 S4 S5 S6 S7 S8 S9 S10;
set K := K1 K2 K3 K4 K5 K6 K7 K8 K9 K10;
set J := J1 J2 J3 J4 J5 J6 J7 J8 J9 J10;
set I := I1 I2 I3 I4 I5 I6 I7 I8 I9 I10;

param E := S1 760 S2 890 S3 840 S4 800 S5 770 S6 760 S7 880 S8 800 S9 800 S10 730;
param D := K1 570 K2 520 K3 510 K4 610 K5 550 K6 510 K7 500 K8 610 K9 610 K10 660;
param W := J1 390 J2 450 J3 440 J4 440 J5 480 J6 410 J7 490 J8 320 J9 340 J10 480;
param d := I1 160 I2 180 I3 160 I4 270 I5 130 I6 230 I7 270 I8 180 I9 110 I10 290;

param g := K1 1270 K2 1460 K3 1060 K4 1430 K5 1070 K6 1460 K7 1340 K8 1130 K9 1160 K10 1350;
param v := J1 1490 J2 1390 J3 1030 J4 1010 J5 1050 J6 1410 J7 1030 J8 1280 J9 1170 J10 1250;

param t: K1 K2 K3 K4 K5 K6 K7 K8 K9 K10 :=
      S1 2.0 4.0 8.0 7.0 9.0 8.0 5.0 2.0 5.0 8.0
      S2 9.0 9.0 1.0 9.0 7.0 9.0 8.0 1.0 8.0 8.0
      S3 3.0 1.0 8.0 3.0 3.0 1.0 5.0 7.0 9.0 7.0
      S4 9.0 8.0 2.0 1.0 7.0 7.0 8.0 5.0 3.0 8.0
      S5 6.0 3.0 1.0 3.0 5.0 3.0 1.0 5.0 7.0 7.0
      S6 9.0 3.0 7.0 1.0 4.0 4.0 5.0 7.0 7.0 4.0
      S7 7.0 3.0 6.0 2.0 9.0 5.0 6.0 4.0 7.0 9.0
      S8 7.0 1.0 1.0 9.0 9.0 4.0 9.0 3.0 7.0 6.0
      S9 8.0 9.0 5.0 1.0 3.0 8.0 6.0 8.0 9.0 4.0
      S10 1.0 1.0 4.0 7.0 2.0 3.0 1.0 5.0 1.0 8.0
;

param a: J1 J2 J3 J4 J5 J6 J7 J8 J9 J10 :=
      K1 1.0 1.0 2.0 2.0 6.0 7.0 5.0 1.0 1.0 3.0
      K2 2.0 5.0 6.0 7.0 4.0 7.0 8.0 1.0 6.0 8.0
      K3 5.0 4.0 2.0 6.0 6.0 1.0 9.0 6.0 3.0 4.0
      K4 4.0 3.0 3.0 3.0 4.0 7.0 4.0 9.0 1.0 8.0
      K5 7.0 2.0 8.0 1.0 9.0 9.0 2.0 7.0 3.0 7.0
      K6 9.0 4.0 1.0 2.0 1.0 5.0 5.0 7.0 9.0 9.0
      K7 3.0 3.0 3.0 4.0 8.0 6.0 8.0 1.0 8.0 4.0
      K8 1.0 8.0 4.0 6.0 8.0 4.0 3.0 9.0 3.0 9.0
      K9 2.0 2.0 2.0 6.0 3.0 9.0 4.0 1.0 4.0 1.0
      K10 5.0 4.0 8.0 8.0 7.0 3.0 1.0 1.0 3.0 6.0
;

param c: I1 I2 I3 I4 I5 I6 I7 I8 I9 I10 :=
      J1 7.0 6.0 6.0 6.0 3.0 6.0 8.0 2.0 5.0 1.0
      J2 1.0 5.0 3.0 4.0 3.0 1.0 1.0 5.0 6.0 3.0
      J3 9.0 5.0 8.0 1.0 5.0 3.0 1.0 4.0 5.0 7.0
      J4 1.0 3.0 2.0 9.0 6.0 3.0 8.0 8.0 2.0 6.0
      J5 7.0 2.0 2.0 1.0 8.0 1.0 9.0 6.0 7.0 7.0
      J6 3.0 2.0 9.0 8.0 7.0 9.0 4.0 4.0 1.0 8.0
      J7 3.0 7.0 2.0 2.0 7.0 6.0 3.0 9.0 6.0 6.0
      J8 1.0 4.0 6.0 6.0 5.0 1.0 8.0 5.0 5.0 7.0
      J9 4.0 6.0 4.0 3.0 7.0 8.0 4.0 2.0 3.0 1.0
      J10 8.0 3.0 7.0 5.0 5.0 7.0 9.0 5.0 1.0 1.0
;

param P_max := 10;
param W_max := 10;
param u := 1;
