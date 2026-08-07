#!/usr/bin/env python3
"""
mfs_crashtest.py -- the other half of reviewer S3: does a pod killed mid-write
leave a corrupt SQLite DB on the mfs volume? SIGKILL a committing writer (the
OOM/preempt case), then reopen (triggers journal/WAL recovery over the mount)
and PRAGMA integrity_check. Repeat; compare mfs vs local disk.
"""
import argparse
import os
import signal
import sqlite3
import subprocess
import sys
import time

WRITER = r"""
import sqlite3, sys
db, jm = sys.argv[1], sys.argv[2]
con = sqlite3.connect(db, timeout=60)
con.execute(f"PRAGMA journal_mode={jm};")
con.execute("PRAGMA synchronous=NORMAL;")
con.execute("CREATE TABLE IF NOT EXISTS t(id INTEGER PRIMARY KEY, blob TEXT)")
con.commit()
while True:
    con.execute("INSERT INTO t(blob) VALUES(?)", ("x" * 400,))
    con.commit()
"""


def one_trial(db, jm):
    for suf in ("", "-wal", "-shm", "-journal"):
        try: os.remove(db + suf)
        except FileNotFoundError: pass
    p = subprocess.Popen([sys.executable, "-c", WRITER, db, jm])
    time.sleep(0.8)                         # let it commit for a while
    os.kill(p.pid, signal.SIGKILL)          # OOM/preempt mid-commit
    p.wait()
    sidecars = [s for s in ("-wal", "-shm", "-journal") if os.path.exists(db + s)]
    try:
        con = sqlite3.connect(db, timeout=60)
        con.execute(f"PRAGMA journal_mode={jm};")
        integ = con.execute("PRAGMA integrity_check").fetchone()[0]
        cnt = con.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        con.close()
        return integ, cnt, sidecars
    except sqlite3.Error as e:
        return f"OPEN_ERROR:{type(e).__name__}:{e}", -1, sidecars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locations", nargs="+", required=True)
    ap.add_argument("--trials", type=int, default=8)
    args = ap.parse_args()
    for loc in args.locations:
        label, _, d = loc.partition(":")
        os.makedirs(d, exist_ok=True)
        for jm in ("DELETE", "WAL"):
            bad = 0; first = None
            for _ in range(args.trials):
                integ, cnt, sc = one_trial(os.path.join(d, "crash.db"), jm)
                if integ != "ok":
                    bad += 1
                if first is None:
                    first = (integ, cnt, sc)
            print(f"[{label}/{jm}] {args.trials} kill-mid-commit trials: "
                  f"CORRUPT={bad}/{args.trials}  first_trial(integrity,rows,sidecars)={first}")


if __name__ == "__main__":
    main()
