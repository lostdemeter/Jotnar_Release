# Broad integer-family listing for the arena soak.
# Every ported kernel family appears at least once.
IN X
IN I
IN S

a = IADD(X, X)
b = IMUL(a, 3)
c = IXOR(b, 7)
d = IMAX(c, X)
e = ISUB(d, I)
f = IAND(e, 255)
g = IOR(f, 1)
h = ISHL(g, 2)
i = ISHR(h, 1)
j = ISAR(i, 1)
k = IMAD(j, j, 5)
l = IADD(k, X)
m = IMOD(l, 7)
n = IDIV(m, 3)
p = ISELECT(X, l, n)

# index path
q = IGATHER1(p, I)
r = ISCATTER1(p, I, q)
s = IWINDOW(r, 0, 8)

# reductions, scalars, packing, checksums
t = ISUM(s, 0, 8)
u = IREDUCE_SUM(s)
v = IGET(s, 2)
w = IASSEMBLE(s, 0, 4)
cr = ICRC32(s)
cc = ICRC32C(s, S)

# multi-output encoder
V, N = IVARINT(X, 0)

OUT = IADD(r, p)
