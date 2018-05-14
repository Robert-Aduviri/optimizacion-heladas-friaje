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

set S := S1 S2 S3 S4 S5 S6 S7;
set K := K1 K2 K3 K4 K5 K6 K7;
set J := J1 J2 J3 J4 J5 J6 J7 J8;
set I := I1 I2 I3 I4 I5 I6 I7 I8;

param E := S1 760 S2 890 S3 840 S4 800 S5 770 S6 760 S7 880;
param D := K1 600 K2 600 K3 530 K4 570 K5 520 K6 510 K7 610;
param W := J1 350 J2 310 J3 300 J4 410 J5 410 J6 460 J7 390 J8 450;
param d := I1 240 I2 240 I3 280 I4 210 I5 290 I6 120 I7 140 I8 280;

param g := K1 1060 K2 1200 K3 1080 K4 1380 K5 1170 K6 1030 K7 1240;
param v := J1 1130 J2 1490 J3 1080 J4 1250 J5 1010 J6 1190 J7 1270 J8 1460;

param t: K1 K2 K3 K4 K5 K6 K7 :=
      S1 7.0 8.0 3.0 1.0 4.0 2.0 8.0
      S2 4.0 2.0 6.0 6.0 4.0 6.0 2.0
      S3 2.0 4.0 8.0 7.0 9.0 8.0 5.0
      S4 2.0 5.0 8.0 9.0 9.0 1.0 9.0
      S5 7.0 9.0 8.0 1.0 8.0 8.0 3.0
      S6 1.0 8.0 3.0 3.0 1.0 5.0 7.0
      S7 9.0 7.0 9.0 8.0 2.0 1.0 7.0
;

param a: J1 J2 J3 J4 J5 J6 J7 J8 :=
      K1 7.0 8.0 5.0 3.0 8.0 6.0 3.0 1.0
      K2 3.0 5.0 3.0 1.0 5.0 7.0 7.0 9.0
      K3 3.0 7.0 1.0 4.0 4.0 5.0 7.0 7.0
      K4 4.0 7.0 3.0 6.0 2.0 9.0 5.0 6.0
      K5 4.0 7.0 9.0 7.0 1.0 1.0 9.0 9.0
      K6 4.0 9.0 3.0 7.0 6.0 8.0 9.0 5.0
      K7 1.0 3.0 8.0 6.0 8.0 9.0 4.0 1.0
;

param c: I1 I2 I3 I4 I5 I6 I7 I8 :=
      J1 1.0 4.0 7.0 2.0 3.0 1.0 5.0 1.0
      J2 8.0 1.0 1.0 2.0 2.0 6.0 7.0 5.0
      J3 1.0 1.0 3.0 2.0 5.0 6.0 7.0 4.0
      J4 7.0 8.0 1.0 6.0 8.0 5.0 4.0 2.0
      J5 6.0 6.0 1.0 9.0 6.0 3.0 4.0 4.0
      J6 3.0 3.0 3.0 4.0 7.0 4.0 9.0 1.0
      J7 8.0 7.0 2.0 8.0 1.0 9.0 9.0 2.0
      J8 7.0 3.0 7.0 9.0 4.0 1.0 2.0 1.0
;

param P_max := 7;
param W_max := 8;
param u := 1;
