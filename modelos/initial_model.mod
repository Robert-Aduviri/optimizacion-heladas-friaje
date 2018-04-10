set S; # SUPPLIERS
set W; # WAREHOUSES
set R; # RECIPIENTS
set A; # AIRPORTS;
set F; # SEAPORTS;
set B; # RAIL TERMINALS;

set I; # ITEMS;
set V; # VEHICLE TYPES;
set K = S union W union R union A union F union B; # NODES
set G = S union W union A union F union B; # INTERMEDIATE NODES;
# set L; # LINKS

set T; # DAYS

# i : items transported
# v : vehicles
# j, k : network nodes (S, W, R, A, F, B)
# t : day within the planning horizon

param Cap       {T}; # available capacity time in each period t
param Invent0    {K, I}; # initial inventory at node k of item i
param Demand    {K, I, T}; # demand at node k for item i on day t
param Udemand0  {K, I}; # unsatisfied demand at node k for item i

param VehCap    {V}; # maximum capacity that each vehicle v can carry
param VehAvail  {K, V, T}; # number of vehicles available at node k of a vehicle type v on day t
param LeadTime  {K, K, V}; # lead time from node k to node j using a vehicle of type v
param Transp0   {K, K, V, I}; # * amount of item i already sent from node k to node j, using a vehicle of type v, i at past times t <= 0
param FixedCost {K, K, V}; # fixed transport cost of using each vehicle type v between node k and node j
param Weight    {I}; # weight of each item i
param CapNode   {K}; # capacity limits of each node k

var Transp {K, K, V, I, T}; # amount of item i sent from node k to node j, using vehicle type v, on day t
var Vehicles {K, K, V, T}; # number of vehicles of type v sent from node k to node j on day t

var Invent   {K, I, T}; # inventory at node k of item i at the end of day t
var Udemand {K, I, T}; # unsatisfied demand at node k for item i at the end of day t
 
minimize Cost:
	        sum {k in K, i in I, t in T}         Udemand[k,i,t] +
	0.001 * sum {k in K, j in K, v in V, t in T} FixedCost[k,j,v] * Vehicles[k,j,v,t] +
	0.01  * sum {k in K, i in I, t in T}         Invent[k,i,t] +
	        sum {k in K, j in K, v in V, t in T} Vehicles[k,j,v,t];
	        
subject to Initial_Udemand {k in K, i in I}:
	Udemand[k,i,0] = Udemand0[k,i];

subject to Initial_Invent {k in K, i in I}:
	Invent[k,i,0] = Invent0[k,i];
	
subject to Initial_Transp {k in K, j in K, v in V, i in I}:
	Transp[k,j,v,i,0] = Transp0[k,j,v,i]; # * t = 0, could be < 0 

subject to Invent_Balance {k in K, i in I, t in T}:
	sum {j in K, v in V} Transp[j,k,v,i,t - LeadTime[j,k,v]] + Invent[k,i,t-1] - Udemand[k,i,t-1] =
	sum {j in K, v in V} Transp[k,j,v,i,t] + Invent[k,i,t] - Udemand[k,i,t] + Demand[k,i,t];
	
subject to Enough_Vehicles {k in K, j in K, v in V, t in T}:
	Vehicles[k,j,v,t] >= 0.001 * (sum {i in I} Transp[k,j,v,i,t] * Weight[i]) / VehCap[v];
	
subject to Capacity_Limit {k in K, t in T}:
	0.001 * sum {i in I} Invent[k,i,t] * Weight[i] <= CapNode[k];	   



# data;

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


	
	
	
	
	
	
