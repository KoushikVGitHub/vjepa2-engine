#!/usr/bin/env python3
"""
mfs_locktest.py -- does SQLite locking (and rename) actually work on the eur-is-1
mfs volume? Decides the eval-framework storage design (reviewer S3).

Runs three probes, each on the mfs volume AND on local disk as a control:
  A. concurrent INSERT integrity  -- W procs hammer one DB; expect rows == W*M,
     integrity_check == 'ok', no lost writes. Journal modes DELETE and WAL.
  B. atomic-claim double-claim    -- the queue primitive. W procs race
     UPDATE q SET owner=? WHERE id=? AND owner IS NULL. If locking is a no-op,
     two procs both see owner NULL and both get rowcount==1 -> DOUBLE CLAIM.
  C. rename() claim primitive     -- files-as-truth fallback. W procs race to
     os.rename a job out of queued/; POSIX gives exactly one winner (others
     ENOENT). A broken FUSE rename double-claims.

Verdict: SQLite-on-mfs is SAFE only if it matches local disk on A and B with no
lost writes, no integrity failure, and no double-claims.
"""
import argparse
import os
import shutil
import sqlite3
import sys
import time
import multiprocessing as mp


# ------------------------------------------------------------------ A: inserts
def _insert_worker(db, wid, m, journal, q):
    con = sqlite3.connect(db, timeout=60)
    con.execute(f"PRAGMA journal_mode={journal};")
    con.execute("PRAGMA busy_timeout=60000;")
    ok = err = 0
    for j in range(m):
        done = False
        for _ in range(200):
            try:
                con.execute("INSERT INTO t(wid, j) VALUES(?,?)", (wid, j))
                con.commit(); ok += 1; done = True; break
            except sqlite3.Error:
                time.sleep(0.005)
        if not done:
            err += 1
    con.close()
    q.put((ok, err))


def test_inserts(db, W, M, journal):
    for suf in ("", "-wal", "-shm", "-journal"):
        try: os.remove(db + suf)
        except FileNotFoundError: pass
    con = sqlite3.connect(db)
    jm = con.execute(f"PRAGMA journal_mode={journal};").fetchone()[0]
    con.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, wid INT, j INT)")
    con.commit(); con.close()

    q = mp.Queue()
    procs = [mp.Process(target=_insert_worker, args=(db, w, M, journal, q)) for w in range(W)]
    [p.start() for p in procs]
    res = [q.get() for _ in procs]
    [p.join() for p in procs]

    ok = sum(r[0] for r in res); err = sum(r[1] for r in res)
    con = sqlite3.connect(db)
    rows = con.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    integ = con.execute("PRAGMA integrity_check").fetchone()[0]
    con.close()
    return dict(mode_requested=journal, mode_actual=jm, expected=W * M,
                committed_ok=ok, commit_err=err, rows_in_db=rows,
                lost=W * M - rows, integrity=integ)


# ------------------------------------------------------------------ B: claims
def _claim_worker(db, wid, journal, q):
    con = sqlite3.connect(db, timeout=60)
    con.execute(f"PRAGMA journal_mode={journal};")
    con.execute("PRAGMA busy_timeout=60000;")
    got = []
    while True:
        row = con.execute("SELECT id FROM q WHERE owner IS NULL ORDER BY id LIMIT 1").fetchone()
        if row is None:
            break
        jid = row[0]
        try:
            cur = con.execute("UPDATE q SET owner=? WHERE id=? AND owner IS NULL", (wid, jid))
            con.commit()
            if cur.rowcount == 1:
                got.append(jid)
        except sqlite3.Error:
            time.sleep(0.003)
    con.close()
    q.put((wid, got))


def test_claims(db, W, K, journal):
    for suf in ("", "-wal", "-shm", "-journal"):
        try: os.remove(db + suf)
        except FileNotFoundError: pass
    con = sqlite3.connect(db)
    con.execute(f"PRAGMA journal_mode={journal};")
    con.execute("CREATE TABLE q(id INTEGER PRIMARY KEY, owner INT)")
    con.executemany("INSERT INTO q(id, owner) VALUES(?, NULL)", [(i,) for i in range(K)])
    con.commit(); con.close()

    q = mp.Queue()
    procs = [mp.Process(target=_claim_worker, args=(db, w, journal, q)) for w in range(W)]
    [p.start() for p in procs]
    claims = [q.get() for _ in procs]
    [p.join() for p in procs]

    # a double-claim = a job id that TWO workers each got rowcount==1 for
    seen = {}
    for wid, got in claims:
        for jid in got:
            seen.setdefault(jid, []).append(wid)
    doubles = {jid: ws for jid, ws in seen.items() if len(ws) > 1}
    con = sqlite3.connect(db)
    unclaimed = con.execute("SELECT COUNT(*) FROM q WHERE owner IS NULL").fetchone()[0]
    con.close()
    total_claimed = sum(len(g) for _, g in claims)
    return dict(jobs=K, total_claim_events=total_claimed, distinct_claimed=len(seen),
                double_claims=len(doubles), unclaimed=unclaimed,
                example_double=next(iter(doubles.items()), None))


# ------------------------------------------------------------------ C: rename
def _rename_worker(qdir, cdir, wid, q):
    got = []
    while True:
        jobs = sorted(os.listdir(qdir))
        if not jobs:
            break
        src = os.path.join(qdir, jobs[0])
        dst = os.path.join(cdir, f"{jobs[0]}.{wid}")
        try:
            os.rename(src, dst)
            if os.path.exists(dst):
                got.append(jobs[0])
        except OSError:
            time.sleep(0.002)
    q.put((wid, got))


def test_rename(base, W, K):
    root = os.path.join(base, "renametest")
    shutil.rmtree(root, ignore_errors=True)
    qdir, cdir = os.path.join(root, "queued"), os.path.join(root, "claimed")
    os.makedirs(qdir); os.makedirs(cdir)
    for i in range(K):
        open(os.path.join(qdir, f"job{i:04d}"), "w").close()

    q = mp.Queue()
    procs = [mp.Process(target=_rename_worker, args=(qdir, cdir, w, q)) for w in range(W)]
    [p.start() for p in procs]
    claims = [q.get() for _ in procs]
    [p.join() for p in procs]

    seen = {}
    for wid, got in claims:
        for j in got:
            seen.setdefault(j, []).append(wid)
    doubles = {j: ws for j, ws in seen.items() if len(ws) > 1}
    leftover = len(os.listdir(qdir))
    shutil.rmtree(root, ignore_errors=True)
    return dict(jobs=K, distinct_claimed=len(seen), double_claims=len(doubles),
                unclaimed_leftover=leftover, example_double=next(iter(doubles.items()), None))


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locations", nargs="+", required=True,
                    help="label:dir pairs, e.g. mfs:/workspace/eval_locktest local:/tmp/eval_locktest")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--inserts", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=400)
    args = ap.parse_args()

    for loc in args.locations:
        label, _, d = loc.partition(":")
        os.makedirs(d, exist_ok=True)
        db = os.path.join(d, "t.db")
        print(f"\n########## {label}  ({d}) ##########")
        for journal in ("DELETE", "WAL"):
            try:
                r = test_inserts(db, args.workers, args.inserts, journal)
                print(f"[A inserts/{journal}] expected {r['expected']} rows={r['rows_in_db']} "
                      f"lost={r['lost']} commit_err={r['commit_err']} integrity={r['integrity']} "
                      f"(mode_actual={r['mode_actual']})")
            except Exception as e:
                print(f"[A inserts/{journal}] EXCEPTION: {type(e).__name__}: {e}")
            try:
                c = test_claims(db, args.workers, args.jobs, journal)
                print(f"[B claims/{journal}]  jobs={c['jobs']} distinct={c['distinct_claimed']} "
                      f"DOUBLE_CLAIMS={c['double_claims']} unclaimed={c['unclaimed']} "
                      f"ex={c['example_double']}")
            except Exception as e:
                print(f"[B claims/{journal}]  EXCEPTION: {type(e).__name__}: {e}")
        try:
            rn = test_rename(d, args.workers, args.jobs)
            print(f"[C rename]        jobs={rn['jobs']} distinct={rn['distinct_claimed']} "
                  f"DOUBLE_CLAIMS={rn['double_claims']} leftover={rn['unclaimed_leftover']} "
                  f"ex={rn['example_double']}")
        except Exception as e:
            print(f"[C rename]        EXCEPTION: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
