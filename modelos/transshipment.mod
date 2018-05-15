set K;
set E within {K, K};

param c {E};
param b {K};
param u {K};

var x {E} >= 0 integer;

minimize Cost:
	sum {(k,j) in E} c[k,j] * x[k,j];

subject to Balance  {k in K}:
	sum {(k,j) in E} x[k,j] - sum {(j,k) in E} x[j,k] = b[k];

subject to In_Capacity {k in K}:
	sum {(j,k) in E} x[j,k] <= u[k];

subject to Out_Capacity {k in K}:
	sum {(k,j) in E} x[k,j] <= u[k];
	
data;

set K := A B C D E F G H I J dummy;
set E := (A,D) (A,E) (B,D) (B,E) (C,D) (C,E)
		 (D,F) (D,G) (E,F) (E,G)
		 (F,H) (F,I) (F,J) (G,H) (G,I) (G,J)
		 (dummy,H) (dummy,I) (dummy,J)
		 (dummy,D) (dummy,E) (dummy,F) (dummy,G);
			 			 
param c := 
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
	
	dummy H 0
	dummy I 0
	dummy J 0
	dummy D 0
	dummy E 0
	dummy F 0
	dummy G 0
	;
	
param b := 
	A 10 B 10 C 10 
	D 0 E 0 F 0 G 0 
	H -80 I -80 J -80 
	dummy 210;
	
param u := 
	A 100 B 100 C 100 
	D 20 E 20 F 20 G 20 
	H 100 I 100 J 100 
	dummy 10000;





