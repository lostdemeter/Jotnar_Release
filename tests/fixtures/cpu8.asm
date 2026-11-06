# cpu8: a single-cycle 8-bit CPU on the 16-bit Jotnar datapath.
# Word = [15:12] opcode, [7:0] operand.  Memory 256 x 16, two read ports
# (PC -> INSTR, ARG -> DATA) and one clocked write.  ACC is 16-bit but kept
# 8-bit by masking.  Z updates only on ALU ops, so STA preserves it.
#
#  0 HLT   1 LDA #   2 LDA addr   3 STA addr   4 ADD #   5 ADD addr
#  6 JMP   7 JZ      8 JNZ        9 INC       10 DEC    11 SUB #
#
# State: PC, ACC, Z, HALT (all 16-bit).  Reset clears them; memory keeps
# its init image and any writes.
IN PC
IN ACC
IN Z
IN HALT
IN INSTR
IN DATA

# fetch fields
OP = ISHR(INSTR, 12)
ARG = IAND(INSTR, 255)

# decode
H = IEQ(OP, 0)
L1 = IEQ(OP, 1)
L2 = IEQ(OP, 2)
S3 = IEQ(OP, 3)
A4 = IEQ(OP, 4)
A5 = IEQ(OP, 5)
J6 = IEQ(OP, 6)
Z7 = IEQ(OP, 7)
N8 = IEQ(OP, 8)
I9 = IEQ(OP, 9)
D10 = IEQ(OP, 10)
B11 = IEQ(OP, 11)

# operands and ALU candidates (8-bit semantics by masking)
D8 = IAND(DATA, 255)
T1 = IADD(ACC, ARG)
C1 = IAND(T1, 255)
T2 = IADD(ACC, D8)
C2 = IAND(T2, 255)
T3 = ISUB(ACC, ARG)
C3 = IAND(T3, 255)
T4 = IADD(ACC, 1)
C4 = IAND(T4, 255)
T5 = ISUB(ACC, 1)
C5 = IAND(T5, 255)

# accumulator next (default: hold)
K0 = ISELECT(L1, ARG, ACC)
K1 = ISELECT(L2, D8, K0)
K2 = ISELECT(A4, C1, K1)
K3 = ISELECT(A5, C2, K2)
K4 = ISELECT(B11, C3, K3)
K5 = ISELECT(I9, C4, K4)
ACC_NEXT = ISELECT(D10, C5, K5)

# Z updates on ALU ops and loads; STA/JMP preserve it
U1 = IOR(I9, D10)
U2 = IOR(B11, U1)
U3 = IOR(A5, U2)
U4 = IOR(A4, U3)
U5 = IOR(L2, U4)
ZSET = IOR(L1, U5)
Z_NEW = IEQ(ACC_NEXT, 0)
Z_NEXT = ISELECT(ZSET, Z_NEW, Z)

# control
HALTED = IOR(HALT, H)
HALT_NEXT = IXOR(HALTED, 0)
NOTZ = IXOR(Z, 1)        # 1-Z for Z in {0,1}
JZ_T = IAND(Z7, Z)
JNZ_T = IAND(N8, NOTZ)
J1 = IOR(JZ_T, JNZ_T)
JT = IOR(J6, J1)
PC1 = IADD(PC, 1)
PC_TAKE = ISELECT(JT, ARG, PC1)
PC_NEXT = ISELECT(HALTED, PC, PC_TAKE)

# write port
NH = IXOR(HALTED, 1)     # 1-HALTED
WE = IAND(S3, NH)
WDATA = IAND(ACC, 255)
