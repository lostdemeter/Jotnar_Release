# RESHAPE2/RESHAPE3 on both arena kinds: a pure bit-copy with new dims.
IN XI
IN XF
A = RESHAPE2(XI, 4, 2)
B = RESHAPE3(XF, 2, 1, 4)
OUT = RESHAPE2(B, 1, 8)
