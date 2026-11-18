# Two-node DOOM frame path.
#
# Each node receives one 160x200 RGB half of the RV32IM framebuffer and runs
# this Jotnar program. The coordinator stitches the two 320x400 results into a
# 640x400 frame. Warm frames keep the GPU context and artifact resident so the
# per-frame path is the Jotnar op plus buffer copies, not process startup.
CONFIG warm_frame 1
IN X
OUT = INTERP(X, 2.0, 2.0)
