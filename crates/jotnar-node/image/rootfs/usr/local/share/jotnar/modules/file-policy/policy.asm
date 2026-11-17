# file-policy: EVENT 40/41/42 -> ACTION 5/6/7 (file.write/read/list).
IN EVENT
ACTION = ISUB(EVENT, 35)
