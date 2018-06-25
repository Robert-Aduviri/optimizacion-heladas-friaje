model datasets/evaluation/model.mod;
solution model.sol;

printf "\n%s\n", "# OBJECTIVE VALUES";
print (sum {(k,j) in E, i in I} transp_cost[k,j] * X[k,j,i]);
print (sum {k in D, i in I} ((sum {(j,k) in E} X[j,k,i]) / -supply_demand[k,i]) ^ 2);

printf "\n%s\n", "# SETS";
print {i in I} i;
print {k in K} k;

printf "\n%s\n", "# SOLUTION";
printf {(k,j) in E, i in I}: "%s %s %s %s\n", k, j, i, X[k,j,i];