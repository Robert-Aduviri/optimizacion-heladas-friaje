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
set ST := A B C D E F G;
set D  := H I J;
set DU := dummy_supply dummy_demand;

set E := (A,D) (A,E) (B,D) (B,E) (C,D) (C,E)
		 (D,F) (D,G) (E,F) (E,G)
		 (F,H) (F,I) (F,J) (G,H) (G,I) (G,J);

set EDU := (A,D) (A,E) (B,D) (B,E) (C,D) (C,E)
		 (D,F) (D,G) (E,F) (E,G)
		 (F,H) (F,I) (F,J) (G,H) (G,I) (G,J)
		 (dummy_supply,H) (dummy_supply,I) (dummy_supply,J)
		 (A,dummy_demand) (B,dummy_demand) (C,dummy_demand);
		 
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
	
# param item_priority := 
# 	I1 10 I2 10 I3 10; 
	
param supply_demand (tr): A  B  C  D  E  F  G   H   I   J dummy_demand dummy_supply :=
				     I1  20 20 20  0  0  0  0 -50 -80 -20       0           90
				     I2  40 40 40  0  0  0  0 -30 -20 -30     -40            0     
				     I3  30 30 30  0  0  0  0 -20 -40 -40       0           10
				     ; 	 
	
param node_capacity := 
	A 200 B 200 C 200 
	D 150 E 150 F 150 G 150 
	H 200 I 200 J 200;
