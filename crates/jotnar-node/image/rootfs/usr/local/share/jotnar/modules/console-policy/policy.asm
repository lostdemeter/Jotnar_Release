# console-policy: map controller events to console capability actions.
# EVENT 10 -> ACTION 1 (start), 11 -> 2 (stop), 12 -> 3 (status).
# ACTIONS is STATE, so the snapshot proves the module host carries state.
STATE ACTIONS
IN ACTIONS
IN EVENT

ACTIONS = IADD(ACTIONS, 1)
ACTION = ISUB(EVENT, 9)
