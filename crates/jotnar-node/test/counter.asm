# Stateful heartbeat: COUNTER survives events and restarts via state.ndjson.
STATE COUNT
IN COUNT
IN TICK

COUNT = IADD(COUNT, 1)
STDOUT = IADD(COUNT, 48)
