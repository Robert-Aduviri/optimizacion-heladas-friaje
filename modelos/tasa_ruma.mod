set CALIDAD;
set RUMA; 

param Proteina {RUMA};
param TVN {RUMA};
param Precio {CALIDAD};
param Max_Proteina {CALIDAD};
param Min_Proteina {CALIDAD};
param Min_TVN {CALIDAD};

var Proporcion {RUMA, CALIDAD} >= 0;

maximize Ventas:
	sum {c in CALIDAD} Precio[c] * (sum {r in RUMA} Proporcion[r,c] * 50);
	
subject to Total_Proporcion {r in RUMA}:
	sum {c in CALIDAD} Proporcion[r,c] = 1;
	
subject to Condiciones_Proteina_Max {c in CALIDAD}:
	sum {r in RUMA} Proteina[r] * Proporcion[r,c] <= Max_Proteina[c] * sum {r in RUMA} Proporcion[r,c];
	
subject to Condiciones_Proteina_Min {c in CALIDAD}:
	sum {r in RUMA} Proteina[r] * Proporcion[r,c] >= Min_Proteina[c] * sum {r in RUMA} Proporcion[r,c];
	
subject to Condiciones_TVN_Min {c in CALIDAD}:
	sum {r in RUMA} TVN[r] * Proporcion[r,c] >= Min_TVN[c] * sum {r in RUMA} Proporcion[r,c];
	
data;

set CALIDAD := SuperPrime Prime Taiwan Thailand Standard;
set RUMA := R1 R2 R3 R4;

param Proteina := 
	R1 67.900731	 
	R2 68.197840
	R3 66.447480
	R4 68.280007
	;
	
param TVN := 
	R1 95.045980
	R2 98.036407
	R3 113.930677
	R4 140.014649
	;
	
param Precio := 
	SuperPrime 1430
	Prime      1380
	Taiwan     1320
	Thailand    950
	Standard    900
	;
	
param Max_Proteina := 
	SuperPrime 100000000
	Prime      100000000
	Taiwan            66
	Thailand   100000000
	Standard   100000000
	;
	
param Min_Proteina :=
	SuperPrime 68
	Prime      67
	Taiwan     66
	Thailand   67
	Standard   0
	; 
	
	
param Min_TVN :=
	SuperPrime 68
	Prime      67
	Taiwan     66
	Thailand   67
	Standard   0
	; 