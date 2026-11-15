# Convolution and transposed convolution.
IN X
IN W
IN B
IN WD
IN BD
Y = CONV2D(X, W, B, 1, 0)
D = DECONV2D(X, WD, BD, 2, 1)
OUT = RELU(Y)
