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

set S := S1 S2 S3;
set K := K1 K2 K3;
set J := J1 J2 J3 J4;
set I := I1 I2 I3 I4 I5;

param E := S1 760 S2 890 S3 840;
param D := K1 600 K2 570 K3 560;
param W := J1 480 J2 400 J3 400 J4 330;
param d := I1 170 I2 120 I3 110 I4 210 I5 150;

param g := K1 1010 K2 1200 K3 1320;
param v := J1 1110 J2 1210 J3 1430 J4 1240;

param t: K1 K2 K3 :=
      S1 1.0 3.0 7.0
      S2 4.0 9.0 3.0
      S3 5.0 3.0 7.0
;

param a: J1 J2 J3 J4 :=
      K1 5.0 9.0 7.0 2.0
      K2 4.0 9.0 2.0 9.0
      K3 5.0 2.0 4.0 7.0
;

param c: I1 I2 I3 I4 I5 :=
      J1 8.0 3.0 1.0 4.0 2.0
      J2 8.0 4.0 2.0 6.0 6.0
      J3 4.0 6.0 2.0 2.0 4.0
      J4 8.0 7.0 9.0 8.0 5.0
;

param P_max := 3;
param W_max := 4;
param u := 1;
