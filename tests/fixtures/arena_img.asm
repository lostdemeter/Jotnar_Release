# Bilinear HWC resize, both align conventions.
IN X
Y = RESIZE2D(X, 3, 2, 0)
Z = RESIZE2D(X, 2, 3, 1)
OUT = RELU(Y)
