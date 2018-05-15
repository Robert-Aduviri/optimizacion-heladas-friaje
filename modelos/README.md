# AMPL
- `reset;`
- `model model.mod;`
- `option solver cplex;`
- `solve;`
- `option display_1col 0;`
- `display Variable;`
- `write gmyampl;`

- `glpsol --math model.mod`
