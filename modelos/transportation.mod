set K;
set J;
set LINKS within {K, J};

param CostoTransp {LINKS};
param Oferta {K};
param Demanda {J};

var Transp {LINKS} >= 0 integer;

minimize Cost:
	sum {(k,j) in LINKS} CostoTransp[k,j] * Transp[k,j];

subject to Satisface_Oferta  {k in K}:
	sum {(k,j) in LINKS} Transp[k,j] = Oferta[k];

subject to Satisface_Demanda {j in J}:
	sum {(k,j) in LINKS} Transp[k,j] = Demanda[j];	
	
data;

set K := A B C;
set J := D E F G;
set LINKS := (A,D) (A,E) (A,F) (A,G) 
			 (B,D) (B,E) (B,F) (B,G) 
			 (C,D) (C,E) (C,F) (C,G);
			 
param CostoTransp := 
	A D 11 
	A E 19 
	A F 17 
	A G 18
	
	B D 16 
	B E 14 
	B F 18 
	B G 15
	
	C D 15 
	C E 16 
	C F 19 
	C G 13 ;
	
param Oferta  := A 550 B 300 C 450;
param Demanda := D 300 E 350 F 300 G 350;





