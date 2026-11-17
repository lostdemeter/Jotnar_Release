# Minimal persistent payload: echo STDIN to the local console each tick.
# The supervisor supplies the seed STDIN on every tick until an event-write
# replaces it for one execution.
IN STDIN
STDOUT = IADD(STDIN, 0)
