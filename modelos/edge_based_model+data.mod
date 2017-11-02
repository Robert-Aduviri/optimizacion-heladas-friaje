set S; # SUPPLIERS
set W; # WAREHOUSES
set R; # RECIPIENTS
# set A; # AIRPORTS;
# set F; # SEAPORTS;
# set B; # RAIL TERMINALS;

set I; # ITEMS;
set V; # VEHICLE TYPES;
set K = S union W union R; # NODES
# set G = S union W union A union F union B; # INTERMEDIATE NODES;
set LINKS within {K, K}; # LINKS

param T >= 0; # DAYS

# i : items transported
# v : vehicles
# j, k : network nodes (S, W, R, A, F, B)
# t : day within the planning horizon

param Cap       {0..T}; # available capacity time in each period t
param Invent0   {K, I}; # initial inventory at node k of item i
param Demand    {K, I, 0..T}; # demand at node k for item i on day t
param Udemand0  {K, I}; # unsatisfied demand at node k for item i

param VehCap    {V}; # maximum capacity that each vehicle v can carry
param VehAvail  {K, V, 0..T}; # number of vehicles available at node k of a vehicle type v on day t
param LeadTime  {LINKS, V}; # lead time from node k to node j using a vehicle of type v
param Transp0   {LINKS, V, I}; # * amount of item i already sent from node k to node j, using a vehicle of type v, i at past times t <= 0
param FixedCost {LINKS, V}; # fixed transport cost of using each vehicle type v between node k and node j
param Weight    {I}; # weight of each item i
param CapNode   {K}; # capacity limits of each node k

var Transp {LINKS, V, I, 0..T} >= 0; # amount of item i sent from node k to node j, using vehicle type v, on day t
var Vehicles {LINKS, V, 0..T} >= 0; # number of vehicles of type v sent from node k to node j on day t

var Invent  {K, I, 0..T} >= 0; # inventory at node k of item i at the end of day t
var Udemand {K, I, 0..T} >= 0; # unsatisfied demand at node k for item i at the end of day t
 
minimize Cost:
	        sum {k in K, i in I, t in 0..T}         Udemand[k,i,t] +
	0.001 * sum {(k,j) in LINKS, v in V, t in 0..T} FixedCost[k,j,v] * Vehicles[k,j,v,t] +
	0.01  * sum {k in K, i in I, t in 0..T}         Invent[k,i,t] +
	        sum {(k,j) in LINKS, v in V, t in 0..T} Vehicles[k,j,v,t];
	        
subject to Initial_Udemand {k in K, i in I}:
	Udemand[k,i,0] = Udemand0[k,i];

subject to Initial_Invent {k in K, i in I}:
	Invent[k,i,0] = Invent0[k,i];
	
subject to Initial_Transp {(k,j) in LINKS, v in V, i in I}:
	Transp[k,j,v,i,0] = Transp0[k,j,v,i]; # * t = 0, could be < 0 

subject to Invent_Balance {k in K, i in I, t in 1..T}:
	sum {(j,k) in LINKS, v in V} Transp[j,k,v,i,t - LeadTime[j,k,v]] + Invent[k,i,t-1] - Udemand[k,i,t-1] =
	sum {(k,j) in LINKS, v in V} Transp[k,j,v,i,t] + Invent[k,i,t] - Udemand[k,i,t] + Demand[k,i,t];
	
subject to Enough_Vehicles {(k,j) in LINKS, v in V, t in 0..T}:
 	Vehicles[k,j,v,t] >= 0.001 * (sum {i in I} Transp[k,j,v,i,t] * Weight[i]) / VehCap[v];
	
subject to Capacity_Limit {k in K, t in 0..T}:
	0.001 * sum {i in I} Invent[k,i,t] * Weight[i] <= CapNode[k];	   

data;

param T := 3;

set S := s1 s2;
set W := t1 t2;
set R := r1 r2;

set LINKS := (s1, t1) (s1, t2) (s2, t1) (s2, t2) 
			 (t1, r1) (t1, r2) (t2, r1) (t2, r2);
			 
set I := A B;
set V := V1 V2;

param Cap := 0 10 1 10 2 10 3 10 4 10;
param Invent0 := 
	[*, A] s1 60 s2 60 t1 10 t2 10 r1 0 r2 0
	[*, B] s1 50 s2 50 t1 8 t2 8 r1 0 r2 0;

param Demand := 
	[*, A, 0] s1 0 s2 0 t1 0 t2 0 r1 10 r2 10
	[*, A, 1] s1 0 s2 0 t1 0 t2 0 r1 10 r2 10
	[*, A, 2] s1 0 s2 0 t1 0 t2 0 r1 10 r2 10
	[*, A, 3] s1 0 s2 0 t1 0 t2 0 r1 10 r2 10
	
	[*, B, 0] s1 0 s2 0 t1 0 t2 0 r1 10 r2 10
	[*, B, 1] s1 0 s2 0 t1 0 t2 0 r1 10 r2 10
	[*, B, 2] s1 0 s2 0 t1 0 t2 0 r1 10 r2 10
	[*, B, 3] s1 0 s2 0 t1 0 t2 0 r1 10 r2 10;
	
param Udemand0 :=
	[*, A] s1 0 s2 0 t1 0 t2 0 r1 5 r2 5
	[*, B] s1 0 s2 0 t1 0 t2 0 r1 5 r2 5;

param VehCap := V1 5 V2 5;

param VehAvail :=
	[*, V1, 0] s1 2 s2 2 t1 2 t2 2 r1 0 r2 0
	[*, V1, 1] s1 2 s2 2 t1 2 t2 2 r1 0 r2 0
	[*, V1, 2] s1 2 s2 2 t1 2 t2 2 r1 0 r2 0
	[*, V1, 3] s1 2 s2 2 t1 2 t2 2 r1 0 r2 0
	
	[*, V2, 0] s1 2 s2 2 t1 2 t2 2 r1 0 r2 0
	[*, V2, 1] s1 2 s2 2 t1 2 t2 2 r1 0 r2 0
	[*, V2, 2] s1 2 s2 2 t1 2 t2 2 r1 0 r2 0
	[*, V2, 3] s1 2 s2 2 t1 2 t2 2 r1 0 r2 0;
	
param LeadTime := 
	[*, *, V1] s1 t1 1  s1 t2 1  s2 t1 1  s2 t2 1 
			   t1 r1 1  t1 r2 1  t2 r1 1  t2 r2 1
	[*, *, V2] s1 t1 1  s1 t2 1  s2 t1 1  s2 t2 1 
			   t1 r1 1  t1 r2 1  t2 r1 1  t2 r2 1;
			   
param Transp0 := 
	[*, *, V1, A] s1 t1 0  s1 t2 0  s2 t1 0  s2 t2 0 
			      t1 r1 0  t1 r2 0  t2 r1 0  t2 r2 0
	[*, *, V2, A] s1 t1 0  s1 t2 0  s2 t1 0  s2 t2 0 
			      t1 r1 0  t1 r2 0  t2 r1 0  t2 r2 0
	[*, *, V1, B] s1 t1 0  s1 t2 0  s2 t1 0  s2 t2 0 
			      t1 r1 0  t1 r2 0  t2 r1 0  t2 r2 0
	[*, *, V2, B] s1 t1 0  s1 t2 0  s2 t1 0  s2 t2 0 
			      t1 r1 0  t1 r2 0  t2 r1 0  t2 r2 0;

param FixedCost := 
	[*, *, V1] s1 t1 5  s1 t2 5  s2 t1 5  s2 t2 5 
			   t1 r1 5  t1 r2 5  t2 r1 5  t2 r2 5
	[*, *, V2] s1 t1 5  s1 t2 5  s2 t1 5  s2 t2 5 
			   t1 r1 5  t1 r2 5  t2 r1 5  t2 r2 5;
			   
param Weight := A 2 B 2;

param CapNode := s1 20 s2 20 t1 20 t2 20 r1 20 r2 20;


			   
	
	

# 10-day planning horizon
# 5 items
# 4 vehicle types
# 54 nodes (83)
#    25 suppliers (49)
#    6 warehouses (6)
#    8 recipients (13)
#    8 airports or heliports (8)
#    1 seaport (1)
#    6 rail terminals (6)
# UNJLC website for the 2005 Pakistan earthquake


	
	
	
	
	
	
