# 8-way tree reduction.  A balanced fan-in-2 tree: the "reduce" specimen.
IN A0
IN A1
IN A2
IN A3
IN A4
IN A5
IN A6
IN A7

s0 = IADD(A0, A1)
s1 = IADD(A2, A3)
s2 = IADD(A4, A5)
s3 = IADD(A6, A7)
t0 = IADD(s0, s1)
t1 = IADD(s2, s3)
OUT = IADD(t0, t1)
