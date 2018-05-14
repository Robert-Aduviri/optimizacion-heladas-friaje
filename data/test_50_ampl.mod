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

set S := S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 S12;
set K := K1 K2 K3 K4 K5 K6 K7 K8 K9 K10 K11 K12;
set J := J1 J2 J3 J4 J5 J6 J7 J8 J9 J10 J11 J12;
set I := I1 I2 I3 I4 I5 I6 I7 I8 I9 I10 I11 I12 I13 I14;

param E := S1 760 S2 890 S3 840 S4 800 S5 770 S6 760 S7 880 S8 800 S9 800 S10 730 S11 770 S12 720;
param D := K1 510 K2 610 K3 550 K4 510 K5 500 K6 610 K7 610 K8 660 K9 590 K10 650 K11 640 K12 640;
param W := J1 480 J2 410 J3 490 J4 320 J5 340 J6 480 J7 360 J8 380 J9 360 J10 470 J11 330 J12 430;
param d := I1 270 I2 180 I3 110 I4 290 I5 240 I6 160 I7 210 I8 170 I9 240 I10 120 I11 230 I12 260 I13 130 I14 270;

param g := K1 1390 K2 1030 K3 1010 K4 1050 K5 1410 K6 1030 K7 1280 K8 1170 K9 1250 K10 1430 K11 1330 K12 1090;
param v := J1 1350 J2 1130 J3 1300 J4 1470 J5 1140 J6 1070 J7 1130 J8 1220 J9 1390 J10 1200 J11 1150 J12 1440;

param t: K1 K2 K3 K4 K5 K6 K7 K8 K9 K10 K11 K12 :=
      S1 2.0 5.0 8.0 9.0 9.0 1.0 9.0 7.0 9.0 8.0 1.0 8.0
      S2 8.0 3.0 1.0 8.0 3.0 3.0 1.0 5.0 7.0 9.0 7.0 9.0
      S3 8.0 2.0 1.0 7.0 7.0 8.0 5.0 3.0 8.0 6.0 3.0 1.0
      S4 3.0 5.0 3.0 1.0 5.0 7.0 7.0 9.0 3.0 7.0 1.0 4.0
      S5 4.0 5.0 7.0 7.0 4.0 7.0 3.0 6.0 2.0 9.0 5.0 6.0
      S6 4.0 7.0 9.0 7.0 1.0 1.0 9.0 9.0 4.0 9.0 3.0 7.0
      S7 6.0 8.0 9.0 5.0 1.0 3.0 8.0 6.0 8.0 9.0 4.0 1.0
      S8 1.0 4.0 7.0 2.0 3.0 1.0 5.0 1.0 8.0 1.0 1.0 2.0
      S9 2.0 6.0 7.0 5.0 1.0 1.0 3.0 2.0 5.0 6.0 7.0 4.0
      S10 7.0 8.0 1.0 6.0 8.0 5.0 4.0 2.0 6.0 6.0 1.0 9.0
      S11 6.0 3.0 4.0 4.0 3.0 3.0 3.0 4.0 7.0 4.0 9.0 1.0
      S12 8.0 7.0 2.0 8.0 1.0 9.0 9.0 2.0 7.0 3.0 7.0 9.0
;

param a: J1 J2 J3 J4 J5 J6 J7 J8 J9 J10 J11 J12 :=
      K1 4.0 1.0 2.0 1.0 5.0 5.0 7.0 9.0 9.0 3.0 3.0 3.0
      K2 4.0 8.0 6.0 8.0 1.0 8.0 4.0 1.0 8.0 4.0 6.0 8.0
      K3 4.0 3.0 9.0 3.0 9.0 2.0 2.0 2.0 6.0 3.0 9.0 4.0
      K4 1.0 4.0 1.0 5.0 4.0 8.0 8.0 7.0 3.0 1.0 1.0 3.0
      K5 6.0 7.0 6.0 6.0 6.0 3.0 6.0 8.0 2.0 5.0 1.0 1.0
      K6 5.0 3.0 4.0 3.0 1.0 1.0 5.0 6.0 3.0 9.0 5.0 8.0
      K7 1.0 5.0 3.0 1.0 4.0 5.0 7.0 1.0 3.0 2.0 9.0 6.0
      K8 3.0 8.0 8.0 2.0 6.0 7.0 2.0 2.0 1.0 8.0 1.0 9.0
      K9 6.0 7.0 7.0 3.0 2.0 9.0 8.0 7.0 9.0 4.0 4.0 1.0
      K10 8.0 3.0 7.0 2.0 2.0 7.0 6.0 3.0 9.0 6.0 6.0 1.0
      K11 4.0 6.0 6.0 5.0 1.0 8.0 5.0 5.0 7.0 4.0 6.0 4.0
      K12 3.0 7.0 8.0 4.0 2.0 3.0 1.0 8.0 3.0 7.0 5.0 5.0
;

param c: I1 I2 I3 I4 I5 I6 I7 I8 I9 I10 I11 I12 I13 I14 :=
      J1 7.0 9.0 5.0 1.0 1.0 2.0 6.0 9.0 8.0 5.0 1.0 7.0 5.0 6.0
      J2 7.0 3.0 3.0 5.0 6.0 9.0 5.0 1.0 4.0 5.0 5.0 7.0 4.0 1.0
      J3 5.0 7.0 6.0 5.0 4.0 2.0 4.0 3.0 1.0 8.0 5.0 4.0 8.0 7.0
      J4 2.0 1.0 4.0 8.0 2.0 3.0 1.0 1.0 3.0 5.0 3.0 1.0 1.0 8.0
      J5 2.0 3.0 2.0 3.0 7.0 1.0 8.0 2.0 3.0 9.0 7.0 4.0 5.0 2.0
      J6 8.0 4.0 9.0 5.0 9.0 4.0 5.0 9.0 8.0 3.0 1.0 3.0 4.0 2.0
      J7 1.0 7.0 8.0 7.0 5.0 1.0 7.0 7.0 9.0 3.0 9.0 1.0 1.0 4.0
      J8 9.0 6.0 3.0 1.0 4.0 9.0 3.0 9.0 7.0 4.0 3.0 5.0 5.0 3.0
      J9 9.0 4.0 5.0 4.0 5.0 7.0 9.0 7.0 5.0 7.0 5.0 3.0 7.0 2.0
      J10 9.0 1.0 6.0 7.0 8.0 9.0 2.0 2.0 5.0 5.0 6.0 3.0 8.0 1.0
      J11 6.0 4.0 1.0 7.0 9.0 4.0 4.0 6.0 3.0 6.0 7.0 3.0 7.0 3.0
      J12 2.0 4.0 8.0 9.0 7.0 1.0 3.0 9.0 1.0 9.0 8.0 1.0 6.0 5.0
;

param P_max := 12;
param W_max := 12;
param u := 1;
