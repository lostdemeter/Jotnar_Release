# 2x2 contraction, then a 2-D slice of the result.
IN A
IN B
C = IMATMUL(A, B)
D = SLICE(C, 1, 0, 1)
OUT = IADD(D, D)
