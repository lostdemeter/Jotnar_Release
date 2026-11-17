# denied-policy: maps EVENT 10 -> ACTION 1 (console.start) but has no grants.
# The host must return a deterministic denied capability result.
IN EVENT
ACTION = ISUB(EVENT, 9)
