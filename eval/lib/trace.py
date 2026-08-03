"""Files-as-truth run store (reviewer S3). Each run is an immutable directory of
JSON on the volume; the write path is write-temp-then-rename (the one filesystem
primitive that is broadly reliable, incl. on the mfs FUSE mount). metrics.json is
written LAST -- its presence marks the run complete, so a crash mid-write leaves a
started-but-not-finalized directory (visible, countable), never a corrupt store.

SQLite, if ever added, is a derived index rebuilt by scanning these directories;
it is never the source of truth. This module has no external deps.
"""
import hashlib
import json
import os
import socket
import subprocess
import tempfile
import time


def _git_sha(repo):
    try:
        return subprocess.check_output(
            ["git", "-C", repo or ".", "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _atomic_write_json(path, obj):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
        os.replace(tmp, path)                 # atomic within the directory
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def file_sha(path, cache=True):
    """sha256 of a (possibly large) file, cached alongside as <path>.sha256 so a
    multi-GB map file is hashed once, not per run (reviewer S11 dataset_sha)."""
    cp = path + ".sha256"
    if cache and os.path.exists(cp) and os.path.getmtime(cp) >= os.path.getmtime(path):
        return open(cp).read().strip()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    d = h.hexdigest()
    if cache:
        try:
            with open(cp, "w") as f:
                f.write(d)
        except OSError:
            pass
    return d


def log_run(runs_dir, run_id, config, metrics, provenance, repo=None):
    """Finalize an immutable run directory. Raises if it already exists finalized."""
    rd = os.path.join(runs_dir, run_id)
    final = os.path.join(rd, "metrics.json")
    if os.path.exists(final):
        raise FileExistsError(f"run {run_id!r} already finalized (immutable): {rd}")
    prov = dict(provenance)
    prov.setdefault("host", socket.gethostname())
    prov.setdefault("created_at", time.time())
    prov.setdefault("git_sha", _git_sha(repo))
    _atomic_write_json(os.path.join(rd, "config.resolved.json"), config)
    _atomic_write_json(os.path.join(rd, "provenance.json"), prov)
    _atomic_write_json(final, metrics)        # LAST -> presence == complete
    return rd


def read_runs(runs_dir):
    """All finalized runs, keyed by run_id. Started-but-not-finalized dirs (no
    metrics.json) are skipped here and surfaced separately by started_without_terminal."""
    out = {}
    if not os.path.isdir(runs_dir):
        return out
    for rid in sorted(os.listdir(runs_dir)):
        rd = os.path.join(runs_dir, rid)
        final = os.path.join(rd, "metrics.json")
        if not os.path.isfile(final):
            continue
        out[rid] = dict(
            run_id=rid,
            config=json.load(open(os.path.join(rd, "config.resolved.json"))),
            metrics=json.load(open(final)),
            provenance=json.load(open(os.path.join(rd, "provenance.json"))),
        )
    return out


def started_without_terminal(runs_dir):
    """Run dirs with a config but no metrics.json -> a run that vanished (OOM/preempt).
    A non-zero count downgrades a DECISIVE verdict to INDECISIVE (reviewer S8)."""
    out = []
    if not os.path.isdir(runs_dir):
        return out
    for rid in sorted(os.listdir(runs_dir)):
        rd = os.path.join(runs_dir, rid)
        if (os.path.isfile(os.path.join(rd, "config.resolved.json"))
                and not os.path.isfile(os.path.join(rd, "metrics.json"))):
            out.append(rid)
    return out
