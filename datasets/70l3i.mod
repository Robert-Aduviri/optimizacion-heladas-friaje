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
set ST := S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 S12 S13 S14 S15 S16 T11 T12 T13 T14 T15 T16 T17 T18 T19 T110 T111 T112 T113 T21 T22 T23 T24 T25 T26 T27 T28 T29 T210 T211 T212 T213 T214 T31 T32 T33 T34 T35 T36 T37 T38 T39 T310 T311 T312 T313 T314 T315 T316;
set D := D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 D11;
set DU := dummy_sup dummy_dem;

set E := 
   (S1,T11) (S1,T12) (S1,T13) (S1,T14) (S1,T15) (S1,T16) (S1,T17) (S1,T18) (S1,T19) (S1,T110) (S1,T111) (S1,T112) (S1,T113) (S2,T11) (S2,T12) (S2,T13) (S2,T14) (S2,T15) (S2,T16) (S2,T17) (S2,T18) (S2,T19) (S2,T110) (S2,T111) (S2,T112) (S2,T113) (S3,T11) (S3,T12) (S3,T13) (S3,T14) (S3,T15) (S3,T16) (S3,T17) (S3,T18) (S3,T19) (S3,T110) (S3,T111) (S3,T112) (S3,T113) (S4,T11) (S4,T12) (S4,T13) (S4,T14) (S4,T15) (S4,T16) (S4,T17) (S4,T18) (S4,T19) (S4,T110) (S4,T111) (S4,T112) (S4,T113) (S5,T11) (S5,T12) (S5,T13) (S5,T14) (S5,T15) (S5,T16) (S5,T17) (S5,T18) (S5,T19) (S5,T110) (S5,T111) (S5,T112) (S5,T113) (S6,T11) (S6,T12) (S6,T13) (S6,T14) (S6,T15) (S6,T16) (S6,T17) (S6,T18) (S6,T19) (S6,T110) (S6,T111) (S6,T112) (S6,T113) (S7,T11) (S7,T12) (S7,T13) (S7,T14) (S7,T15) (S7,T16) (S7,T17) (S7,T18) (S7,T19) (S7,T110) (S7,T111) (S7,T112) (S7,T113) (S8,T11) (S8,T12) (S8,T13) (S8,T14) (S8,T15) (S8,T16) (S8,T17) (S8,T18) (S8,T19) (S8,T110) (S8,T111) (S8,T112) (S8,T113) (S9,T11) (S9,T12) (S9,T13) (S9,T14) (S9,T15) (S9,T16) (S9,T17) (S9,T18) (S9,T19) (S9,T110) (S9,T111) (S9,T112) (S9,T113) (S10,T11) (S10,T12) (S10,T13) (S10,T14) (S10,T15) (S10,T16) (S10,T17) (S10,T18) (S10,T19) (S10,T110) (S10,T111) (S10,T112) (S10,T113) (S11,T11) (S11,T12) (S11,T13) (S11,T14) (S11,T15) (S11,T16) (S11,T17) (S11,T18) (S11,T19) (S11,T110) (S11,T111) (S11,T112) (S11,T113) (S12,T11) (S12,T12) (S12,T13) (S12,T14) (S12,T15) (S12,T16) (S12,T17) (S12,T18) (S12,T19) (S12,T110) (S12,T111) (S12,T112) (S12,T113) (S13,T11) (S13,T12) (S13,T13) (S13,T14) (S13,T15) (S13,T16) (S13,T17) (S13,T18) (S13,T19) (S13,T110) (S13,T111) (S13,T112) (S13,T113) (S14,T11) (S14,T12) (S14,T13) (S14,T14) (S14,T15) (S14,T16) (S14,T17) (S14,T18) (S14,T19) (S14,T110) (S14,T111) (S14,T112) (S14,T113) (S15,T11) (S15,T12) (S15,T13) (S15,T14) (S15,T15) (S15,T16) (S15,T17) (S15,T18) (S15,T19) (S15,T110) (S15,T111) (S15,T112) (S15,T113) (S16,T11) (S16,T12) (S16,T13) (S16,T14) (S16,T15) (S16,T16) (S16,T17) (S16,T18) (S16,T19) (S16,T110) (S16,T111) (S16,T112) (S16,T113)
   (T11,T21) (T11,T22) (T11,T23) (T11,T24) (T11,T25) (T11,T26) (T11,T27) (T11,T28) (T11,T29) (T11,T210) (T11,T211) (T11,T212) (T11,T213) (T11,T214) (T12,T21) (T12,T22) (T12,T23) (T12,T24) (T12,T25) (T12,T26) (T12,T27) (T12,T28) (T12,T29) (T12,T210) (T12,T211) (T12,T212) (T12,T213) (T12,T214) (T13,T21) (T13,T22) (T13,T23) (T13,T24) (T13,T25) (T13,T26) (T13,T27) (T13,T28) (T13,T29) (T13,T210) (T13,T211) (T13,T212) (T13,T213) (T13,T214) (T14,T21) (T14,T22) (T14,T23) (T14,T24) (T14,T25) (T14,T26) (T14,T27) (T14,T28) (T14,T29) (T14,T210) (T14,T211) (T14,T212) (T14,T213) (T14,T214) (T15,T21) (T15,T22) (T15,T23) (T15,T24) (T15,T25) (T15,T26) (T15,T27) (T15,T28) (T15,T29) (T15,T210) (T15,T211) (T15,T212) (T15,T213) (T15,T214) (T16,T21) (T16,T22) (T16,T23) (T16,T24) (T16,T25) (T16,T26) (T16,T27) (T16,T28) (T16,T29) (T16,T210) (T16,T211) (T16,T212) (T16,T213) (T16,T214) (T17,T21) (T17,T22) (T17,T23) (T17,T24) (T17,T25) (T17,T26) (T17,T27) (T17,T28) (T17,T29) (T17,T210) (T17,T211) (T17,T212) (T17,T213) (T17,T214) (T18,T21) (T18,T22) (T18,T23) (T18,T24) (T18,T25) (T18,T26) (T18,T27) (T18,T28) (T18,T29) (T18,T210) (T18,T211) (T18,T212) (T18,T213) (T18,T214) (T19,T21) (T19,T22) (T19,T23) (T19,T24) (T19,T25) (T19,T26) (T19,T27) (T19,T28) (T19,T29) (T19,T210) (T19,T211) (T19,T212) (T19,T213) (T19,T214) (T110,T21) (T110,T22) (T110,T23) (T110,T24) (T110,T25) (T110,T26) (T110,T27) (T110,T28) (T110,T29) (T110,T210) (T110,T211) (T110,T212) (T110,T213) (T110,T214) (T111,T21) (T111,T22) (T111,T23) (T111,T24) (T111,T25) (T111,T26) (T111,T27) (T111,T28) (T111,T29) (T111,T210) (T111,T211) (T111,T212) (T111,T213) (T111,T214) (T112,T21) (T112,T22) (T112,T23) (T112,T24) (T112,T25) (T112,T26) (T112,T27) (T112,T28) (T112,T29) (T112,T210) (T112,T211) (T112,T212) (T112,T213) (T112,T214) (T113,T21) (T113,T22) (T113,T23) (T113,T24) (T113,T25) (T113,T26) (T113,T27) (T113,T28) (T113,T29) (T113,T210) (T113,T211) (T113,T212) (T113,T213) (T113,T214) (T21,T31) (T21,T32) (T21,T33) (T21,T34) (T21,T35) (T21,T36) (T21,T37) (T21,T38) (T21,T39) (T21,T310) (T21,T311) (T21,T312) (T21,T313) (T21,T314) (T21,T315) (T21,T316) (T22,T31) (T22,T32) (T22,T33) (T22,T34) (T22,T35) (T22,T36) (T22,T37) (T22,T38) (T22,T39) (T22,T310) (T22,T311) (T22,T312) (T22,T313) (T22,T314) (T22,T315) (T22,T316) (T23,T31) (T23,T32) (T23,T33) (T23,T34) (T23,T35) (T23,T36) (T23,T37) (T23,T38) (T23,T39) (T23,T310) (T23,T311) (T23,T312) (T23,T313) (T23,T314) (T23,T315) (T23,T316) (T24,T31) (T24,T32) (T24,T33) (T24,T34) (T24,T35) (T24,T36) (T24,T37) (T24,T38) (T24,T39) (T24,T310) (T24,T311) (T24,T312) (T24,T313) (T24,T314) (T24,T315) (T24,T316) (T25,T31) (T25,T32) (T25,T33) (T25,T34) (T25,T35) (T25,T36) (T25,T37) (T25,T38) (T25,T39) (T25,T310) (T25,T311) (T25,T312) (T25,T313) (T25,T314) (T25,T315) (T25,T316) (T26,T31) (T26,T32) (T26,T33) (T26,T34) (T26,T35) (T26,T36) (T26,T37) (T26,T38) (T26,T39) (T26,T310) (T26,T311) (T26,T312) (T26,T313) (T26,T314) (T26,T315) (T26,T316) (T27,T31) (T27,T32) (T27,T33) (T27,T34) (T27,T35) (T27,T36) (T27,T37) (T27,T38) (T27,T39) (T27,T310) (T27,T311) (T27,T312) (T27,T313) (T27,T314) (T27,T315) (T27,T316) (T28,T31) (T28,T32) (T28,T33) (T28,T34) (T28,T35) (T28,T36) (T28,T37) (T28,T38) (T28,T39) (T28,T310) (T28,T311) (T28,T312) (T28,T313) (T28,T314) (T28,T315) (T28,T316) (T29,T31) (T29,T32) (T29,T33) (T29,T34) (T29,T35) (T29,T36) (T29,T37) (T29,T38) (T29,T39) (T29,T310) (T29,T311) (T29,T312) (T29,T313) (T29,T314) (T29,T315) (T29,T316) (T210,T31) (T210,T32) (T210,T33) (T210,T34) (T210,T35) (T210,T36) (T210,T37) (T210,T38) (T210,T39) (T210,T310) (T210,T311) (T210,T312) (T210,T313) (T210,T314) (T210,T315) (T210,T316) (T211,T31) (T211,T32) (T211,T33) (T211,T34) (T211,T35) (T211,T36) (T211,T37) (T211,T38) (T211,T39) (T211,T310) (T211,T311) (T211,T312) (T211,T313) (T211,T314) (T211,T315) (T211,T316) (T212,T31) (T212,T32) (T212,T33) (T212,T34) (T212,T35) (T212,T36) (T212,T37) (T212,T38) (T212,T39) (T212,T310) (T212,T311) (T212,T312) (T212,T313) (T212,T314) (T212,T315) (T212,T316) (T213,T31) (T213,T32) (T213,T33) (T213,T34) (T213,T35) (T213,T36) (T213,T37) (T213,T38) (T213,T39) (T213,T310) (T213,T311) (T213,T312) (T213,T313) (T213,T314) (T213,T315) (T213,T316) (T214,T31) (T214,T32) (T214,T33) (T214,T34) (T214,T35) (T214,T36) (T214,T37) (T214,T38) (T214,T39) (T214,T310) (T214,T311) (T214,T312) (T214,T313) (T214,T314) (T214,T315) (T214,T316)
   (T31,D1) (T31,D2) (T31,D3) (T31,D4) (T31,D5) (T31,D6) (T31,D7) (T31,D8) (T31,D9) (T31,D10) (T31,D11) (T32,D1) (T32,D2) (T32,D3) (T32,D4) (T32,D5) (T32,D6) (T32,D7) (T32,D8) (T32,D9) (T32,D10) (T32,D11) (T33,D1) (T33,D2) (T33,D3) (T33,D4) (T33,D5) (T33,D6) (T33,D7) (T33,D8) (T33,D9) (T33,D10) (T33,D11) (T34,D1) (T34,D2) (T34,D3) (T34,D4) (T34,D5) (T34,D6) (T34,D7) (T34,D8) (T34,D9) (T34,D10) (T34,D11) (T35,D1) (T35,D2) (T35,D3) (T35,D4) (T35,D5) (T35,D6) (T35,D7) (T35,D8) (T35,D9) (T35,D10) (T35,D11) (T36,D1) (T36,D2) (T36,D3) (T36,D4) (T36,D5) (T36,D6) (T36,D7) (T36,D8) (T36,D9) (T36,D10) (T36,D11) (T37,D1) (T37,D2) (T37,D3) (T37,D4) (T37,D5) (T37,D6) (T37,D7) (T37,D8) (T37,D9) (T37,D10) (T37,D11) (T38,D1) (T38,D2) (T38,D3) (T38,D4) (T38,D5) (T38,D6) (T38,D7) (T38,D8) (T38,D9) (T38,D10) (T38,D11) (T39,D1) (T39,D2) (T39,D3) (T39,D4) (T39,D5) (T39,D6) (T39,D7) (T39,D8) (T39,D9) (T39,D10) (T39,D11) (T310,D1) (T310,D2) (T310,D3) (T310,D4) (T310,D5) (T310,D6) (T310,D7) (T310,D8) (T310,D9) (T310,D10) (T310,D11) (T311,D1) (T311,D2) (T311,D3) (T311,D4) (T311,D5) (T311,D6) (T311,D7) (T311,D8) (T311,D9) (T311,D10) (T311,D11) (T312,D1) (T312,D2) (T312,D3) (T312,D4) (T312,D5) (T312,D6) (T312,D7) (T312,D8) (T312,D9) (T312,D10) (T312,D11) (T313,D1) (T313,D2) (T313,D3) (T313,D4) (T313,D5) (T313,D6) (T313,D7) (T313,D8) (T313,D9) (T313,D10) (T313,D11) (T314,D1) (T314,D2) (T314,D3) (T314,D4) (T314,D5) (T314,D6) (T314,D7) (T314,D8) (T314,D9) (T314,D10) (T314,D11) (T315,D1) (T315,D2) (T315,D3) (T315,D4) (T315,D5) (T315,D6) (T315,D7) (T315,D8) (T315,D9) (T315,D10) (T315,D11) (T316,D1) (T316,D2) (T316,D3) (T316,D4) (T316,D5) (T316,D6) (T316,D7) (T316,D8) (T316,D9) (T316,D10) (T316,D11)
;

set EDU := 
   (S1,T11) (S1,T12) (S1,T13) (S1,T14) (S1,T15) (S1,T16) (S1,T17) (S1,T18) (S1,T19) (S1,T110) (S1,T111) (S1,T112) (S1,T113) (S2,T11) (S2,T12) (S2,T13) (S2,T14) (S2,T15) (S2,T16) (S2,T17) (S2,T18) (S2,T19) (S2,T110) (S2,T111) (S2,T112) (S2,T113) (S3,T11) (S3,T12) (S3,T13) (S3,T14) (S3,T15) (S3,T16) (S3,T17) (S3,T18) (S3,T19) (S3,T110) (S3,T111) (S3,T112) (S3,T113) (S4,T11) (S4,T12) (S4,T13) (S4,T14) (S4,T15) (S4,T16) (S4,T17) (S4,T18) (S4,T19) (S4,T110) (S4,T111) (S4,T112) (S4,T113) (S5,T11) (S5,T12) (S5,T13) (S5,T14) (S5,T15) (S5,T16) (S5,T17) (S5,T18) (S5,T19) (S5,T110) (S5,T111) (S5,T112) (S5,T113) (S6,T11) (S6,T12) (S6,T13) (S6,T14) (S6,T15) (S6,T16) (S6,T17) (S6,T18) (S6,T19) (S6,T110) (S6,T111) (S6,T112) (S6,T113) (S7,T11) (S7,T12) (S7,T13) (S7,T14) (S7,T15) (S7,T16) (S7,T17) (S7,T18) (S7,T19) (S7,T110) (S7,T111) (S7,T112) (S7,T113) (S8,T11) (S8,T12) (S8,T13) (S8,T14) (S8,T15) (S8,T16) (S8,T17) (S8,T18) (S8,T19) (S8,T110) (S8,T111) (S8,T112) (S8,T113) (S9,T11) (S9,T12) (S9,T13) (S9,T14) (S9,T15) (S9,T16) (S9,T17) (S9,T18) (S9,T19) (S9,T110) (S9,T111) (S9,T112) (S9,T113) (S10,T11) (S10,T12) (S10,T13) (S10,T14) (S10,T15) (S10,T16) (S10,T17) (S10,T18) (S10,T19) (S10,T110) (S10,T111) (S10,T112) (S10,T113) (S11,T11) (S11,T12) (S11,T13) (S11,T14) (S11,T15) (S11,T16) (S11,T17) (S11,T18) (S11,T19) (S11,T110) (S11,T111) (S11,T112) (S11,T113) (S12,T11) (S12,T12) (S12,T13) (S12,T14) (S12,T15) (S12,T16) (S12,T17) (S12,T18) (S12,T19) (S12,T110) (S12,T111) (S12,T112) (S12,T113) (S13,T11) (S13,T12) (S13,T13) (S13,T14) (S13,T15) (S13,T16) (S13,T17) (S13,T18) (S13,T19) (S13,T110) (S13,T111) (S13,T112) (S13,T113) (S14,T11) (S14,T12) (S14,T13) (S14,T14) (S14,T15) (S14,T16) (S14,T17) (S14,T18) (S14,T19) (S14,T110) (S14,T111) (S14,T112) (S14,T113) (S15,T11) (S15,T12) (S15,T13) (S15,T14) (S15,T15) (S15,T16) (S15,T17) (S15,T18) (S15,T19) (S15,T110) (S15,T111) (S15,T112) (S15,T113) (S16,T11) (S16,T12) (S16,T13) (S16,T14) (S16,T15) (S16,T16) (S16,T17) (S16,T18) (S16,T19) (S16,T110) (S16,T111) (S16,T112) (S16,T113)
   (T11,T21) (T11,T22) (T11,T23) (T11,T24) (T11,T25) (T11,T26) (T11,T27) (T11,T28) (T11,T29) (T11,T210) (T11,T211) (T11,T212) (T11,T213) (T11,T214) (T12,T21) (T12,T22) (T12,T23) (T12,T24) (T12,T25) (T12,T26) (T12,T27) (T12,T28) (T12,T29) (T12,T210) (T12,T211) (T12,T212) (T12,T213) (T12,T214) (T13,T21) (T13,T22) (T13,T23) (T13,T24) (T13,T25) (T13,T26) (T13,T27) (T13,T28) (T13,T29) (T13,T210) (T13,T211) (T13,T212) (T13,T213) (T13,T214) (T14,T21) (T14,T22) (T14,T23) (T14,T24) (T14,T25) (T14,T26) (T14,T27) (T14,T28) (T14,T29) (T14,T210) (T14,T211) (T14,T212) (T14,T213) (T14,T214) (T15,T21) (T15,T22) (T15,T23) (T15,T24) (T15,T25) (T15,T26) (T15,T27) (T15,T28) (T15,T29) (T15,T210) (T15,T211) (T15,T212) (T15,T213) (T15,T214) (T16,T21) (T16,T22) (T16,T23) (T16,T24) (T16,T25) (T16,T26) (T16,T27) (T16,T28) (T16,T29) (T16,T210) (T16,T211) (T16,T212) (T16,T213) (T16,T214) (T17,T21) (T17,T22) (T17,T23) (T17,T24) (T17,T25) (T17,T26) (T17,T27) (T17,T28) (T17,T29) (T17,T210) (T17,T211) (T17,T212) (T17,T213) (T17,T214) (T18,T21) (T18,T22) (T18,T23) (T18,T24) (T18,T25) (T18,T26) (T18,T27) (T18,T28) (T18,T29) (T18,T210) (T18,T211) (T18,T212) (T18,T213) (T18,T214) (T19,T21) (T19,T22) (T19,T23) (T19,T24) (T19,T25) (T19,T26) (T19,T27) (T19,T28) (T19,T29) (T19,T210) (T19,T211) (T19,T212) (T19,T213) (T19,T214) (T110,T21) (T110,T22) (T110,T23) (T110,T24) (T110,T25) (T110,T26) (T110,T27) (T110,T28) (T110,T29) (T110,T210) (T110,T211) (T110,T212) (T110,T213) (T110,T214) (T111,T21) (T111,T22) (T111,T23) (T111,T24) (T111,T25) (T111,T26) (T111,T27) (T111,T28) (T111,T29) (T111,T210) (T111,T211) (T111,T212) (T111,T213) (T111,T214) (T112,T21) (T112,T22) (T112,T23) (T112,T24) (T112,T25) (T112,T26) (T112,T27) (T112,T28) (T112,T29) (T112,T210) (T112,T211) (T112,T212) (T112,T213) (T112,T214) (T113,T21) (T113,T22) (T113,T23) (T113,T24) (T113,T25) (T113,T26) (T113,T27) (T113,T28) (T113,T29) (T113,T210) (T113,T211) (T113,T212) (T113,T213) (T113,T214) (T21,T31) (T21,T32) (T21,T33) (T21,T34) (T21,T35) (T21,T36) (T21,T37) (T21,T38) (T21,T39) (T21,T310) (T21,T311) (T21,T312) (T21,T313) (T21,T314) (T21,T315) (T21,T316) (T22,T31) (T22,T32) (T22,T33) (T22,T34) (T22,T35) (T22,T36) (T22,T37) (T22,T38) (T22,T39) (T22,T310) (T22,T311) (T22,T312) (T22,T313) (T22,T314) (T22,T315) (T22,T316) (T23,T31) (T23,T32) (T23,T33) (T23,T34) (T23,T35) (T23,T36) (T23,T37) (T23,T38) (T23,T39) (T23,T310) (T23,T311) (T23,T312) (T23,T313) (T23,T314) (T23,T315) (T23,T316) (T24,T31) (T24,T32) (T24,T33) (T24,T34) (T24,T35) (T24,T36) (T24,T37) (T24,T38) (T24,T39) (T24,T310) (T24,T311) (T24,T312) (T24,T313) (T24,T314) (T24,T315) (T24,T316) (T25,T31) (T25,T32) (T25,T33) (T25,T34) (T25,T35) (T25,T36) (T25,T37) (T25,T38) (T25,T39) (T25,T310) (T25,T311) (T25,T312) (T25,T313) (T25,T314) (T25,T315) (T25,T316) (T26,T31) (T26,T32) (T26,T33) (T26,T34) (T26,T35) (T26,T36) (T26,T37) (T26,T38) (T26,T39) (T26,T310) (T26,T311) (T26,T312) (T26,T313) (T26,T314) (T26,T315) (T26,T316) (T27,T31) (T27,T32) (T27,T33) (T27,T34) (T27,T35) (T27,T36) (T27,T37) (T27,T38) (T27,T39) (T27,T310) (T27,T311) (T27,T312) (T27,T313) (T27,T314) (T27,T315) (T27,T316) (T28,T31) (T28,T32) (T28,T33) (T28,T34) (T28,T35) (T28,T36) (T28,T37) (T28,T38) (T28,T39) (T28,T310) (T28,T311) (T28,T312) (T28,T313) (T28,T314) (T28,T315) (T28,T316) (T29,T31) (T29,T32) (T29,T33) (T29,T34) (T29,T35) (T29,T36) (T29,T37) (T29,T38) (T29,T39) (T29,T310) (T29,T311) (T29,T312) (T29,T313) (T29,T314) (T29,T315) (T29,T316) (T210,T31) (T210,T32) (T210,T33) (T210,T34) (T210,T35) (T210,T36) (T210,T37) (T210,T38) (T210,T39) (T210,T310) (T210,T311) (T210,T312) (T210,T313) (T210,T314) (T210,T315) (T210,T316) (T211,T31) (T211,T32) (T211,T33) (T211,T34) (T211,T35) (T211,T36) (T211,T37) (T211,T38) (T211,T39) (T211,T310) (T211,T311) (T211,T312) (T211,T313) (T211,T314) (T211,T315) (T211,T316) (T212,T31) (T212,T32) (T212,T33) (T212,T34) (T212,T35) (T212,T36) (T212,T37) (T212,T38) (T212,T39) (T212,T310) (T212,T311) (T212,T312) (T212,T313) (T212,T314) (T212,T315) (T212,T316) (T213,T31) (T213,T32) (T213,T33) (T213,T34) (T213,T35) (T213,T36) (T213,T37) (T213,T38) (T213,T39) (T213,T310) (T213,T311) (T213,T312) (T213,T313) (T213,T314) (T213,T315) (T213,T316) (T214,T31) (T214,T32) (T214,T33) (T214,T34) (T214,T35) (T214,T36) (T214,T37) (T214,T38) (T214,T39) (T214,T310) (T214,T311) (T214,T312) (T214,T313) (T214,T314) (T214,T315) (T214,T316)
   (T31,D1) (T31,D2) (T31,D3) (T31,D4) (T31,D5) (T31,D6) (T31,D7) (T31,D8) (T31,D9) (T31,D10) (T31,D11) (T32,D1) (T32,D2) (T32,D3) (T32,D4) (T32,D5) (T32,D6) (T32,D7) (T32,D8) (T32,D9) (T32,D10) (T32,D11) (T33,D1) (T33,D2) (T33,D3) (T33,D4) (T33,D5) (T33,D6) (T33,D7) (T33,D8) (T33,D9) (T33,D10) (T33,D11) (T34,D1) (T34,D2) (T34,D3) (T34,D4) (T34,D5) (T34,D6) (T34,D7) (T34,D8) (T34,D9) (T34,D10) (T34,D11) (T35,D1) (T35,D2) (T35,D3) (T35,D4) (T35,D5) (T35,D6) (T35,D7) (T35,D8) (T35,D9) (T35,D10) (T35,D11) (T36,D1) (T36,D2) (T36,D3) (T36,D4) (T36,D5) (T36,D6) (T36,D7) (T36,D8) (T36,D9) (T36,D10) (T36,D11) (T37,D1) (T37,D2) (T37,D3) (T37,D4) (T37,D5) (T37,D6) (T37,D7) (T37,D8) (T37,D9) (T37,D10) (T37,D11) (T38,D1) (T38,D2) (T38,D3) (T38,D4) (T38,D5) (T38,D6) (T38,D7) (T38,D8) (T38,D9) (T38,D10) (T38,D11) (T39,D1) (T39,D2) (T39,D3) (T39,D4) (T39,D5) (T39,D6) (T39,D7) (T39,D8) (T39,D9) (T39,D10) (T39,D11) (T310,D1) (T310,D2) (T310,D3) (T310,D4) (T310,D5) (T310,D6) (T310,D7) (T310,D8) (T310,D9) (T310,D10) (T310,D11) (T311,D1) (T311,D2) (T311,D3) (T311,D4) (T311,D5) (T311,D6) (T311,D7) (T311,D8) (T311,D9) (T311,D10) (T311,D11) (T312,D1) (T312,D2) (T312,D3) (T312,D4) (T312,D5) (T312,D6) (T312,D7) (T312,D8) (T312,D9) (T312,D10) (T312,D11) (T313,D1) (T313,D2) (T313,D3) (T313,D4) (T313,D5) (T313,D6) (T313,D7) (T313,D8) (T313,D9) (T313,D10) (T313,D11) (T314,D1) (T314,D2) (T314,D3) (T314,D4) (T314,D5) (T314,D6) (T314,D7) (T314,D8) (T314,D9) (T314,D10) (T314,D11) (T315,D1) (T315,D2) (T315,D3) (T315,D4) (T315,D5) (T315,D6) (T315,D7) (T315,D8) (T315,D9) (T315,D10) (T315,D11) (T316,D1) (T316,D2) (T316,D3) (T316,D4) (T316,D5) (T316,D6) (T316,D7) (T316,D8) (T316,D9) (T316,D10) (T316,D11)
   (S1,dummy_dem) (S2,dummy_dem) (S3,dummy_dem) (S4,dummy_dem) (S5,dummy_dem) (S6,dummy_dem) (S7,dummy_dem) (S8,dummy_dem) (S9,dummy_dem) (S10,dummy_dem) (S11,dummy_dem) (S12,dummy_dem) (S13,dummy_dem) (S14,dummy_dem) (S15,dummy_dem) (S16,dummy_dem)
   (dummy_sup,D1) (dummy_sup,D2) (dummy_sup,D3) (dummy_sup,D4) (dummy_sup,D5) (dummy_sup,D6) (dummy_sup,D7) (dummy_sup,D8) (dummy_sup,D9) (dummy_sup,D10) (dummy_sup,D11)
;

param transp_cost := S1 T11 929 S1 T12 890 S1 T13 226 S1 T14 492 S1 T15 740 S1 T16 157 S1 T17 733 S1 T18 612 S1 T19 850 S1 T110 901 S1 T111 195 S1 T112 737 S1 T113 217 S2 T11 659 S2 T12 700 S2 T13 587 S2 T14 336 S2 T15 984 S2 T16 996 S2 T17 371 S2 T18 288 S2 T19 803 S2 T110 546 S2 T111 680 S2 T112 889 S2 T113 960 S3 T11 346 S3 T12 175 S3 T13 253 S3 T14 755 S3 T15 534 S3 T16 185 S3 T17 796 S3 T18 384 S3 T19 319 S3 T110 168 S3 T111 146 S3 T112 193 S3 T113 849 S4 T11 552 S4 T12 303 S4 T13 317 S4 T14 573 S4 T15 531 S4 T16 440 S4 T17 650 S4 T18 711 S4 T19 388 S4 T110 353 S4 T111 833 S4 T112 456 S4 T113 122 S5 T11 861 S5 T12 621 S5 T13 857 S5 T14 936 S5 T15 199 S5 T16 901 S5 T17 279 S5 T18 322 S5 T19 861 S5 T110 758 S5 T111 541 S5 T112 707 S5 T113 868 S6 T11 424 S6 T12 615 S6 T13 115 S6 T14 891 S6 T15 435 S6 T16 858 S6 T17 357 S6 T18 596 S6 T19 995 S6 T110 259 S6 T111 574 S6 T112 951 S6 T113 763 S7 T11 789 S7 T12 774 S7 T13 479 S7 T14 644 S7 T15 790 S7 T16 526 S7 T17 712 S7 T18 367 S7 T19 934 S7 T110 676 S7 T111 516 S7 T112 267 S7 T113 941 S8 T11 142 S8 T12 655 S8 T13 384 S8 T14 496 S8 T15 111 S8 T16 706 S8 T17 401 S8 T18 997 S8 T19 352 S8 T110 598 S8 T111 853 S8 T112 134 S8 T113 826 S9 T11 948 S9 T12 189 S9 T13 875 S9 T14 704 S9 T15 701 S9 T16 517 S9 T17 214 S9 T18 716 S9 T19 295 S9 T110 925 S9 T111 600 S9 T112 725 S9 T113 592 S10 T11 174 S10 T12 512 S10 T13 475 S10 T14 519 S10 T15 828 S10 T16 376 S10 T17 860 S10 T18 775 S10 T19 493 S10 T110 968 S10 T111 556 S10 T112 291 S10 T113 838 S11 T11 788 S11 T12 198 S11 T13 647 S11 T14 195 S11 T15 763 S11 T16 762 S11 T17 289 S11 T18 835 S11 T19 136 S11 T110 879 S11 T111 468 S11 T112 794 S11 T113 624 S12 T11 378 S12 T12 316 S12 T13 966 S12 T14 972 S12 T15 897 S12 T16 372 S12 T17 980 S12 T18 161 S12 T19 695 S12 T110 979 S12 T111 828 S12 T112 441 S12 T113 496 S13 T11 798 S13 T12 118 S13 T13 276 S13 T14 711 S13 T15 495 S13 T16 544 S13 T17 332 S13 T18 175 S13 T19 364 S13 T110 554 S13 T111 895 S13 T112 817 S13 T113 834 S14 T11 483 S14 T12 663 S14 T13 950 S14 T14 605 S14 T15 466 S14 T16 243 S14 T17 984 S14 T18 168 S14 T19 198 S14 T110 495 S14 T111 124 S14 T112 990 S14 T113 568 S15 T11 583 S15 T12 664 S15 T13 250 S15 T14 243 S15 T15 668 S15 T16 138 S15 T17 208 S15 T18 792 S15 T19 141 S15 T110 285 S15 T111 497 S15 T112 322 S15 T113 733 S16 T11 232 S16 T12 262 S16 T13 314 S16 T14 832 S16 T15 334 S16 T16 942 S16 T17 757 S16 T18 850 S16 T19 687 S16 T110 108 S16 T111 173 S16 T112 591 S16 T113 352 T11 T21 329 T11 T22 618 T11 T23 273 T11 T24 752 T11 T25 267 T11 T26 269 T11 T27 492 T11 T28 894 T11 T29 733 T11 T210 293 T11 T211 616 T11 T212 128 T11 T213 264 T11 T214 521 T12 T21 438 T12 T22 747 T12 T23 595 T12 T24 464 T12 T25 932 T12 T26 441 T12 T27 599 T12 T28 756 T12 T29 610 T12 T210 426 T12 T211 316 T12 T212 400 T12 T213 231 T12 T214 903 T13 T21 169 T13 T22 351 T13 T23 514 T13 T24 886 T13 T25 544 T13 T26 975 T13 T27 281 T13 T28 266 T13 T29 190 T13 T210 813 T13 T211 957 T13 T212 630 T13 T213 138 T13 T214 225 T14 T21 550 T14 T22 272 T14 T23 752 T14 T24 853 T14 T25 319 T14 T26 737 T14 T27 157 T14 T28 759 T14 T29 575 T14 T210 555 T14 T211 928 T14 T212 994 T14 T213 460 T14 T214 100 T15 T21 486 T15 T22 447 T15 T23 289 T15 T24 604 T15 T25 290 T15 T26 607 T15 T27 468 T15 T28 508 T15 T29 923 T15 T210 216 T15 T211 233 T15 T212 157 T15 T213 655 T15 T214 784 T16 T21 771 T16 T22 272 T16 T23 928 T16 T24 914 T16 T25 248 T16 T26 179 T16 T27 985 T16 T28 312 T16 T29 302 T16 T210 863 T16 T211 328 T16 T212 775 T16 T213 326 T16 T214 758 T17 T21 631 T17 T22 540 T17 T23 501 T17 T24 146 T17 T25 332 T17 T26 404 T17 T27 625 T17 T28 242 T17 T29 514 T17 T210 612 T17 T211 472 T17 T212 665 T17 T213 985 T17 T214 358 T18 T21 755 T18 T22 570 T18 T23 111 T18 T24 429 T18 T25 835 T18 T26 883 T18 T27 457 T18 T28 507 T18 T29 767 T18 T210 472 T18 T211 107 T18 T212 221 T18 T213 447 T18 T214 775 T19 T21 189 T19 T22 747 T19 T23 797 T19 T24 415 T19 T25 277 T19 T26 639 T19 T27 831 T19 T28 968 T19 T29 140 T19 T210 839 T19 T211 803 T19 T212 601 T19 T213 244 T19 T214 300 T110 T21 823 T110 T22 560 T110 T23 831 T110 T24 851 T110 T25 657 T110 T26 646 T110 T27 352 T110 T28 489 T110 T29 693 T110 T210 982 T110 T211 355 T110 T212 808 T110 T213 914 T110 T214 549 T111 T21 109 T111 T22 923 T111 T23 897 T111 T24 341 T111 T25 350 T111 T26 976 T111 T27 104 T111 T28 218 T111 T29 900 T111 T210 473 T111 T211 164 T111 T212 245 T111 T213 323 T111 T214 338 T112 T21 276 T112 T22 878 T112 T23 952 T112 T24 381 T112 T25 162 T112 T26 316 T112 T27 953 T112 T28 926 T112 T29 894 T112 T210 788 T112 T211 560 T112 T212 709 T112 T213 204 T112 T214 198 T113 T21 610 T113 T22 484 T113 T23 504 T113 T24 922 T113 T25 617 T113 T26 575 T113 T27 808 T113 T28 962 T113 T29 744 T113 T210 969 T113 T211 870 T113 T212 536 T113 T213 122 T113 T214 664 T21 T31 776 T21 T32 813 T21 T33 557 T21 T34 182 T21 T35 244 T21 T36 184 T21 T37 177 T21 T38 556 T21 T39 977 T21 T310 100 T21 T311 150 T21 T312 784 T21 T313 816 T21 T314 871 T21 T315 545 T21 T316 548 T22 T31 980 T22 T32 587 T22 T33 899 T22 T34 133 T22 T35 447 T22 T36 194 T22 T37 171 T22 T38 650 T22 T39 253 T22 T310 349 T22 T311 773 T22 T312 537 T22 T313 217 T22 T314 870 T22 T315 990 T22 T316 405 T23 T31 367 T23 T32 153 T23 T33 335 T23 T34 488 T23 T35 833 T23 T36 862 T23 T37 924 T23 T38 500 T23 T39 723 T23 T310 866 T23 T311 402 T23 T312 306 T23 T313 328 T23 T314 952 T23 T315 677 T23 T316 174 T24 T31 534 T24 T32 202 T24 T33 521 T24 T34 803 T24 T35 863 T24 T36 837 T24 T37 905 T24 T38 405 T24 T39 197 T24 T310 949 T24 T311 897 T24 T312 434 T24 T313 958 T24 T314 534 T24 T315 546 T24 T316 407 T25 T31 348 T25 T32 265 T25 T33 580 T25 T34 434 T25 T35 641 T25 T36 858 T25 T37 205 T25 T38 662 T25 T39 180 T25 T310 232 T25 T311 896 T25 T312 999 T25 T313 237 T25 T314 283 T25 T315 500 T25 T316 429 T26 T31 884 T26 T32 607 T26 T33 951 T26 T34 168 T26 T35 645 T26 T36 873 T26 T37 152 T26 T38 549 T26 T39 304 T26 T310 225 T26 T311 846 T26 T312 142 T26 T313 851 T26 T314 982 T26 T315 210 T26 T316 860 T27 T31 762 T27 T32 282 T27 T33 947 T27 T34 194 T27 T35 473 T27 T36 619 T27 T37 999 T27 T38 459 T27 T39 999 T27 T310 795 T27 T311 380 T27 T312 294 T27 T313 963 T27 T314 294 T27 T315 766 T27 T316 192 T28 T31 771 T28 T32 160 T28 T33 733 T28 T34 150 T28 T35 502 T28 T36 888 T28 T37 104 T28 T38 565 T28 T39 703 T28 T310 781 T28 T311 928 T28 T312 633 T28 T313 504 T28 T314 937 T28 T315 612 T28 T316 744 T29 T31 332 T29 T32 623 T29 T33 445 T29 T34 145 T29 T35 645 T29 T36 177 T29 T37 829 T29 T38 912 T29 T39 172 T29 T310 893 T29 T311 348 T29 T312 146 T29 T313 220 T29 T314 313 T29 T315 338 T29 T316 155 T210 T31 449 T210 T32 206 T210 T33 930 T210 T34 147 T210 T35 672 T210 T36 436 T210 T37 125 T210 T38 903 T210 T39 356 T210 T310 107 T210 T311 724 T210 T312 966 T210 T313 352 T210 T314 791 T210 T315 818 T210 T316 402 T211 T31 738 T211 T32 155 T211 T33 825 T211 T34 625 T211 T35 573 T211 T36 127 T211 T37 177 T211 T38 315 T211 T39 485 T211 T310 464 T211 T311 765 T211 T312 625 T211 T313 798 T211 T314 411 T211 T315 982 T211 T316 106 T212 T31 102 T212 T32 722 T212 T33 849 T212 T34 462 T212 T35 850 T212 T36 607 T212 T37 373 T212 T38 605 T212 T39 905 T212 T310 838 T212 T311 982 T212 T312 114 T212 T313 291 T212 T314 730 T212 T315 584 T212 T316 127 T213 T31 813 T213 T32 138 T213 T33 372 T213 T34 330 T213 T35 185 T213 T36 573 T213 T37 225 T213 T38 911 T213 T39 636 T213 T310 244 T213 T311 624 T213 T312 311 T213 T313 124 T213 T314 679 T213 T315 493 T213 T316 166 T214 T31 332 T214 T32 208 T214 T33 245 T214 T34 210 T214 T35 327 T214 T36 210 T214 T37 569 T214 T38 133 T214 T39 722 T214 T310 875 T214 T311 468 T214 T312 847 T214 T313 694 T214 T314 653 T214 T315 524 T214 T316 712 T31 D1 617 T31 D2 791 T31 D3 381 T31 D4 419 T31 D5 837 T31 D6 542 T31 D7 414 T31 D8 976 T31 D9 840 T31 D10 476 T31 D11 388 T32 D1 536 T32 D2 761 T32 D3 376 T32 D4 809 T32 D5 681 T32 D6 602 T32 D7 723 T32 D8 615 T32 D9 449 T32 D10 942 T32 D11 545 T33 D1 929 T33 D2 449 T33 D3 706 T33 D4 763 T33 D5 922 T33 D6 236 T33 D7 858 T33 D8 998 T33 D9 342 T33 D10 642 T33 D11 139 T34 D1 328 T34 D2 647 T34 D3 322 T34 D4 105 T34 D5 421 T34 D6 567 T34 D7 831 T34 D8 174 T34 D9 103 T34 D10 946 T34 D11 745 T35 D1 217 T35 D2 980 T35 D3 705 T35 D4 603 T35 D5 673 T35 D6 540 T35 D7 293 T35 D8 946 T35 D9 814 T35 D10 491 T35 D11 125 T36 D1 534 T36 D2 272 T36 D3 399 T36 D4 744 T36 D5 425 T36 D6 509 T36 D7 217 T36 D8 935 T36 D9 886 T36 D10 823 T36 D11 324 T37 D1 631 T37 D2 212 T37 D3 239 T37 D4 658 T37 D5 100 T37 D6 189 T37 D7 753 T37 D8 419 T37 D9 649 T37 D10 904 T37 D11 993 T38 D1 238 T38 D2 610 T38 D3 967 T38 D4 944 T38 D5 358 T38 D6 772 T38 D7 789 T38 D8 109 T38 D9 360 T38 D10 749 T38 D11 271 T39 D1 997 T39 D2 112 T39 D3 267 T39 D4 741 T39 D5 695 T39 D6 420 T39 D7 418 T39 D8 712 T39 D9 428 T39 D10 884 T39 D11 876 T310 D1 649 T310 D2 518 T310 D3 705 T310 D4 194 T310 D5 916 T310 D6 680 T310 D7 417 T310 D8 335 T310 D9 433 T310 D10 984 T310 D11 475 T311 D1 558 T311 D2 236 T311 D3 687 T311 D4 646 T311 D5 612 T311 D6 395 T311 D7 419 T311 D8 671 T311 D9 931 T311 D10 448 T311 D11 939 T312 D1 878 T312 D2 972 T312 D3 328 T312 D4 369 T312 D5 927 T312 D6 897 T312 D7 390 T312 D8 824 T312 D9 264 T312 D10 104 T312 D11 950 T313 D1 478 T313 D2 177 T313 D3 509 T313 D4 673 T313 D5 231 T313 D6 700 T313 D7 848 T313 D8 141 T313 D9 188 T313 D10 501 T313 D11 395 T314 D1 939 T314 D2 394 T314 D3 715 T314 D4 625 T314 D5 899 T314 D6 790 T314 D7 736 T314 D8 521 T314 D9 964 T314 D10 122 T314 D11 674 T315 D1 242 T315 D2 764 T315 D3 244 T315 D4 324 T315 D5 602 T315 D6 920 T315 D7 715 T315 D8 203 T315 D9 138 T315 D10 297 T315 D11 745 T316 D1 678 T316 D2 490 T316 D3 278 T316 D4 653 T316 D5 547 T316 D6 599 T316 D7 114 T316 D8 722 T316 D9 768 T316 D10 388 T316 D11 577;

param supply_demand (tr): S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 S12 S13 S14 S15 S16 T11 T12 T13 T14 T15 T16 T17 T18 T19 T110 T111 T112 T113 T21 T22 T23 T24 T25 T26 T27 T28 T29 T210 T211 T212 T213 T214 T31 T32 T33 T34 T35 T36 T37 T38 T39 T310 T311 T312 T313 T314 T315 T316 D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 D11 dummy_sup dummy_dem := 
    I1 30 20 10 20 30 10 20 20 30 30 30 10 30 10 20 80 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -50 -10 -60 -50 -60 -40 -60 -30 -70 -60 -10 100 0
    I2 40 20 60 10 10 20 10 20 50 10 30 20 40 10 10 40 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -30 -40 -40 -70 -50 -70 -30 -30 -40 -30 -70 100 0
    I3 40 20 50 10 10 20 10 20 10 10 10 20 10 10 10 140 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -40 -10 -20 -20 -10 -30 -40 -40 -60 -30 -200 100 0
;

param node_capacity := S1 110 S2 60 S3 120 S4 40 S5 50 S6 50 S7 40 S8 60 S9 90 S10 50 S11 70 S12 50 S13 80 S14 30 S15 40 S16 260 T11 102 T12 107 T13 110 T14 100 T15 101 T16 125 T17 106 T18 122 T19 121 T110 107 T111 104 T112 118 T113 345 T21 107 T22 126 T23 113 T24 109 T25 105 T26 105 T27 120 T28 110 T29 85 T210 117 T211 102 T212 108 T213 102 T214 227 T31 81 T32 93 T33 109 T34 95 T35 81 T36 93 T37 75 T38 84 T39 84 T310 95 T311 101 T312 94 T313 90 T314 82 T315 89 T316 293 D1 120 D2 60 D3 120 D4 140 D5 120 D6 140 D7 130 D8 100 D9 170 D10 120 D11 280;
