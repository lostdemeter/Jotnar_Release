# 2x2 matrix multiply.  Four dot products with a shared operand graph.
IN a00
IN a01
IN a10
IN a11
IN b00
IN b01
IN b10
IN b11

t00 = IMUL(a00, b00)
t01 = IMUL(a01, b10)
c00 = IADD(t00, t01)
t10 = IMUL(a10, b00)
t11 = IMUL(a11, b10)
c10 = IADD(t10, t11)
t02 = IMUL(a00, b01)
t03 = IMUL(a01, b11)
c01 = IADD(t02, t03)
t12 = IMUL(a10, b01)
t13 = IMUL(a11, b11)
c11 = IADD(t12, t13)
