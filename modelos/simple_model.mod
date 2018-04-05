set K;
set LINKS within {K, K};

param T >= 0;
param CapacidadVeh >= 0;
param CostoTransp {LINKS};

param Demanda0 {K};
param Inv0 {K};
param VehDisp0 {K};

var Transp    {LINKS, 0..T} >= 0 integer;
var Veh       {LINKS, 0..T} >= 0 integer;
var DemandaNS {K, 0..T} >= 0 integer;
var Consumo   {K, 0..T} >= 0 integer;
var Inv       {K, 0..T} >= 0 integer;
var VehDisp   {K, 0..T} >= 0 integer;

minimize Cost:
	sum {k in K, t in 0..T}         100 * DemandaNS[k,t] + 
	sum {(k,j) in LINKS, t in 1..T} CostoTransp[k,j] * Veh[k,j,t];
	
subject to Update_Invent {k in K, t in 1..T}:
	Inv[k,t] = Inv[k,t-1] 
				+ sum {(j,k) in LINKS} Transp[j,k,t-1] # Llegada de viajes que empezaron ayer 
				- Consumo[k,t] 
				- sum {(k,j) in LINKS} Transp[k,j,t-1]; # Salida de transporte que se fue ayer
	
subject to Transporation_Requirement {k in K, t in 0..T}:
	sum {(k,j) in LINKS} Transp[k,j,t] <= Inv[k,t];	
	
subject to Update_Demand {k in K, t in 1..T}:
	DemandaNS[k,t] = DemandaNS[k,t-1] - Consumo[k,t];
	
subject to Update_Vehicles {k in K, t in 1..T}:
	VehDisp[k,t] = VehDisp[k,t-1] 
	    			+ sum {(j,k) in LINKS} Veh[j,k,t-1]
	    			- sum {(k,j) in LINKS} Veh[k,j,t-1];

subject to Vehicle_Capacity {(k,j) in LINKS, t in 0..T}:
	Transp[k,j,t] = Veh[k,j,t] * CapacidadVeh;

subject to Initial_Demand {k in K}:
	DemandaNS[k,0] = Demanda0[k];
	
subject to Initial_Invent {k in K}:
	Inv[k,0] = Inv0[k];
	
subject to Initial_Vehicles {k in K}:
	VehDisp[k,0] = VehDisp0[k];
	
	
data;

# param T := 2;

# set K := a b c;
# set LINKS := (a, b) (b, c);

# param CapacidadVeh := 20;
# param CostoTransp := a b 10 b c 10;
# param Demanda0 := a 0 b 20 c 10;
# param Inv0 := a 50 b 0 c 0;
# param VehDisp0 := a 4 b 0 c 0;

param T := 6;
set K := a b c d e f g h;
set LINKS := (a,c) (b,c) (c,d) (c,e) (d,f) (d,g) (e,g) (e,h);
param CapacidadVeh := 20;
param CostoTransp default 10;
param Demanda0 := a   0 b  0 c  0 d 30 e 20 f 40 g 40 h 40;
param Inv0     := a 100 b 80 c 10 d  0 e  0 f  0 g  0 h  0;
param VehDisp0 default 0 := a 10 b 10;




