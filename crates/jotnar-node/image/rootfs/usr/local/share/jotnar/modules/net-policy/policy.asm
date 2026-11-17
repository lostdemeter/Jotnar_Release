# net-policy: EVENT 80..84 -> ACTION 13..17 (net status/connect/send/recv/close).
IN EVENT
ACTION = ISUB(EVENT, 67)
