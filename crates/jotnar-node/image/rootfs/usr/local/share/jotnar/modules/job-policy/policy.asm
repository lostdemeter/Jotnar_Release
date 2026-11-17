# job-policy: EVENT 90/91/92/93 -> ACTION 18/19/20/21
# (job.submit/status/cancel/list). The capability shim forwards to the local
# job endpoint; the node remains model-blind.
IN EVENT
ACTION = ISUB(EVENT, 72)
