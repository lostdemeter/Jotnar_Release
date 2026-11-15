# Row-wise F norms and rotary, with eps/base coming from CONFIG.
CONFIG eps_ln 1e-6
CONFIG eps_rms 1e-6
CONFIG rope_base 10000
IN X
IN W
IN B
IN P
LN = LAYERNORM(X, W, B)
R = RMSNORM(X, W)
Y = ROTARY(X, P)
OUT = LAYERNORM(R, W, B)
