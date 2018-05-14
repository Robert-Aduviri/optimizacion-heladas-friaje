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

set S := S1 S2 S3 S4 S5;
set K := K1 K2 K3 K4 K5;
set J := J1 J2 J3 J4 J5;
set I := I1 I2 I3 I4 I5;

param E := S1 760 S2 890 S3 840 S4 800 S5 770;
param D := K1 560 K2 680 K3 600 K4 600 K5 530;
param W := J1 370 J2 320 J3 310 J4 410 J5 350;
param d := I1 110 I2 100 I3 210 I4 210 I5 260;

param g := K1 1260 K2 1410 K3 1270 K4 1150 K5 1140;
param v := J1 1460 J2 1430 J3 1020 J4 1360 J5 1060;

param t: K1 K2 K3 K4 K5 :=
      S1 5.0 9.0 7.0 2.0 4.0
      S2 9.0 2.0 9.0 5.0 2.0
      S3 4.0 7.0 8.0 3.0 1.0
      S4 4.0 2.0 8.0 4.0 2.0
      S5 6.0 6.0 4.0 6.0 2.0
;

param a: J1 J2 J3 J4 J5 :=
      K1 2.0 4.0 8.0 7.0 9.0
      K2 8.0 5.0 2.0 5.0 8.0
      K3 9.0 9.0 1.0 9.0 7.0
      K4 9.0 8.0 1.0 8.0 8.0
      K5 3.0 1.0 8.0 3.0 3.0
;

param c: I1 I2 I3 I4 I5 :=
      J1 1.0 5.0 7.0 9.0 7.0
      J2 9.0 8.0 2.0 1.0 7.0
      J3 7.0 8.0 5.0 3.0 8.0
      J4 6.0 3.0 1.0 3.0 5.0
      J5 3.0 1.0 5.0 7.0 7.0
;

param P_max := 5;
param W_max := 5;
param u := 1;
