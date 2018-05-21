set I; # items

set ST; # supply and transshipment nodes
set D; # demand nodes
set K := ST union D; # all nodes 
set E within {K, K}; # all edges

set DU; # dummy nodes
set KDU := K union DU; # all nodes + dummies
set EDU within {KDU,KDU}; # all edges + dummies

param transp_cost {E};
param supply_demand {KDU};
param node_capacity {K};

var X {EDU} >= 0 integer; # items to be transported

minimize Cost:
	(sum {(k,j) in E} transp_cost[k,j] * X[k,j]) + 
	1000*(sum {k in D} ((sum {(j,k) in E} X[j,k]) / -supply_demand[k]) ^ 2); # fairness
	
subject to Transportation_Balance {k in KDU}:
	sum {(k,j) in EDU} X[k,j] - sum {(j,k) in EDU} X[j,k] = supply_demand[k];

subject to Inbound_Capacity {k in K}:
	sum {(j,k) in E} X[j,k] <= node_capacity[k];
	
subject to Outbound_Capacity {k in K}:
	sum {(k,j) in E} X[k,j] <= node_capacity[k];
	
data;

set ST := A B C D E F G;
set D  := H I J;
set DU := dummy;

set E := (A,D) (A,E) (B,D) (B,E) (C,D) (C,E)
		 (D,F) (D,G) (E,F) (E,G)
		 (F,H) (F,I) (F,J) (G,H) (G,I) (G,J);

set EDU := (A,D) (A,E) (B,D) (B,E) (C,D) (C,E)
		 (D,F) (D,G) (E,F) (E,G)
		 (F,H) (F,I) (F,J) (G,H) (G,I) (G,J)
		 (dummy,H) (dummy,I) (dummy,J);
		 
param transp_cost :=
	A D 11 
	A E 19 
	B D 17
	B E 18
	C D 13
	C E 14
	
	D F 16 
	D G 14 
	E F 18 
	E G 15
	
	F H 15 
	F I 16 
	F J 19 
	G H 13 
	G I 17
	G J 16
	;
	
param supply_demand := 
	A 20 B 20 C 20 
	D 0 E 0 F 0 G 0 
	H -50 I -80 J -20 
	dummy 90;
	
param node_capacity := 
	A 100 B 100 C 100 
	D 40 E 40 F 40 G 40 
	H 100 I 100 J 100;
