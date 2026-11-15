# Float path: activation, F -> I, I -> F.
IN X
R = RELU(X)
I = F2I(R)
Y = I2F(I)
OUT = RELU(Y)
