# F CONV and INTERP: edge-replicate convolution + dyadic bilinear scales.
IN X
IN K
IN Y
C = CONV(X, K)
I1 = INTERP(Y, 0.5, 0.5)
I2 = INTERP(Y, 2.0, 1.0)
OUT = CONV(C, K)
