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

set S := S1 S2;
set K := K1 K2;
set J := J1 J2 J3;
set I := I1 I2 I3;

param E := S1 760 S2 890;
param D := K1 640 K2 600;
param W := J1 370 J2 360 J3 480;
param d := I1 200 I2 200 I3 130;

param g := K1 1390 K2 1230;
param v := J1 1020 J2 1210 J3 1010;

param t: K1 K2 :=
      S1 8.0 6.0
      S2 2.0 5.0
;

param a: J1 J2 J3 :=
      K1 1.0 6.0 9.0
      K2 1.0 3.0 7.0
;

param c: I1 I2 I3 :=
      J1 4.0 9.0 3.0
      J2 5.0 3.0 7.0
      J3 5.0 9.0 7.0
;

param P_max := 2;
param W_max := 3;
param u := 1;
