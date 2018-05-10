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
set K := K1 K2 K3 K4;
set J := J1 J2 J3 J4;
set I := I1 I2 I3 I4 I5;

param E := S1 400 S2 600 S3 500;
param D := K1 300 K2 400 K3 500 K4 350;
param W := J1 150 J2 150 J3 350 J4 500;
param d := I1 200 I2 150 I3 200 I4 250 I5 100;

param g := K1 1150 K2 1200 K3 900 K4 1250;
param v := J1 900 J2 1050 J3 1300 J4 1100;

param t: K1 K2 K3 K4 :=
	  S1 6 5 6 7
	  S2 4 2 7 6
	  S3 4 3 7 4;
	  
param a: J1 J2 J3 J4 := 
	  K1 4 4 2 1
	  K2 3 2 2 4
	  K3 4 6 5 3 
	  K4 8 3 7 2;
	  
param c: I1 I2 I3 I4 I5 :=
	  J1 7 4 7 7 6
	  J2 9 2 3 8 9
	  J3 6 7 3 8 3
	  J4 3 7 9 4 9;
	
param P_max := 4;
param W_max := 4;
param u := 1;