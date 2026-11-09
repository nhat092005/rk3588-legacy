"""Acceptance of an NPU candidate against a profile in ai/automation/profiles/ (see npu_g4.yaml for the numbers).

Only conditions A, B and C of the gate are computed here. G1 to G3 (sources, workload table, per-layer reading) need human reading
of the documentation, so `passes` is necessary for a "Giữ" label, not sufficient.

    from ai.automation.acceptance import evaluate, load_profile
    evaluate(load_profile("npu_g4"), coverage_per_session=[0.976], drop_pct=33.4, drop_ms=4.29, gap_ms=0.02, diff_pp={"map50_95": -0.3, ...})
"""
from __future__ import annotations

from pathlib import Path

import yaml

PROFILES_DIR = Path(__file__).resolve().parent / "profiles"


def load_profile(name: str) -> dict:
    return yaml.safe_load((PROFILES_DIR / f"{name}.yaml").read_text())


def check_a(coverage_per_session: list[float], profile: dict) -> bool:
    return min(coverage_per_session) >= profile["a_coverage_min"]


def check_b(drop_pct: float, drop_ms: float, gap_ms: float, profile: dict) -> bool:
    return drop_pct >= profile["b_drop_pct_min"] and drop_ms > gap_ms


def check_c(diff_pp: dict[str, float], profile: dict) -> bool:
    return all(diff_pp[c] >= -profile["c_max_drop_pp"] for c in profile["c_columns"])


def evaluate(profile: dict, coverage_per_session: list[float], drop_pct: float, drop_ms: float, gap_ms: float, diff_pp: dict[str, float]) -> dict:
    a = check_a(coverage_per_session, profile)
    b = check_b(drop_pct, drop_ms, gap_ms, profile)
    c = check_c(diff_pp, profile)
    return dict(A=a, B=b, C=c, passes=a and b and c)
