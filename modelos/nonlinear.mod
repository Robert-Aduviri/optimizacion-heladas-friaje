set K;
var x {K} >= 0 integer;

minimize Cost:
	sum {k in K} (x[k]) ^ 2;
	
subject to Max:
	sum {k in K} x[k] = 30;
	
data;

set K := A B C;