# Workload #2: a batch GPU chain.
#
# Each stage reads one F32 stream and doubles it. Doubling is exact for the
# small integer payload values the gate generates, so a CUDA stage and a HIP
# stage must produce byte-identical F32 output.
CONFIG warm_frame 1
IN X
OUT = ADD(X, X)
