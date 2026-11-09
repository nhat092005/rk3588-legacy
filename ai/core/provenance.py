"""Provenance fields written into every result file (Guide section 8, traceability rule)."""
from __future__ import annotations

import csv
import datetime
import functools
import hashlib
import subprocess
from pathlib import Path

from ai.paths import EXPERIMENTS_DIR, REPO_ROOT


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


@functools.lru_cache(maxsize=None)
def git_dirty() -> bool:
    """True if code under ai/ or scripts/ (tracked or new) differs from the commit.

    Generated output (experiments/, results/) lives outside ai/ and scripts/, so it never makes this true.

    Cached per process: a long-running process (e.g. training for hours) keeps
    executing the code it loaded at start regardless of later edits on disk, so every
    file that process writes should report the git state from its own start, not from
    whenever this happens to be called.
    """
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "ai/", "scripts/"], cwd=REPO_ROOT, text=True)
        return bool(out.strip())
    except Exception:
        return True


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(**extra) -> dict:
    return {
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        **extra,
    }


INDEX_FIELDS = ["exp_id", "date", "domain", "topic", "run_id", "script", "rule_file", "result_file", "git_commit", "status", "note"]


def log_experiment(exp_id: str, domain: str, topic: str, run_id: str, script: str, result_file: str,
                   rule_file: str = "", status: str = "recorded", note: str = "", index_path: str | Path | None = None) -> None:
    """Append one row to experiments/index.csv (header written when the file is new). date and git_commit are taken now.

    Several values in one field are joined with '|'. The file is an append-only log: re-running an experiment adds a row.
    """
    path = Path(index_path) if index_path else EXPERIMENTS_DIR / "index.csv"
    row = dict(exp_id=exp_id, date=datetime.date.today().isoformat(), domain=domain, topic=topic, run_id=run_id, script=script,
               rule_file=rule_file, result_file=result_file, git_commit=git_commit(), status=status, note=note)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_FIELDS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(row)
