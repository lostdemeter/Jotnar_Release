# SAXPY: A*X + Y.  A linear elementwise specimen.
IN A
IN X
IN Y

ax = IMUL(A, X)
OUT = IADD(ax, Y)
