# F matmul: 2-D, 3-D batched, and a singleton batch broadcast.
IN A2
IN B2
IN A3
IN B3
IN B1
C = MATMUL(A2, B2)
D = BATCH_MATMUL(A3, B3)
E = BATCH_MATMUL(A3, B1)
OUT = MATMUL(C, B2)
