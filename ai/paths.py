"""Single source of truth for repository paths. Scripts import locations from here, never rebuild them."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

RUNS_DIR = EXPERIMENTS_DIR / "runs"
LEADERBOARDS_DIR = EXPERIMENTS_DIR / "leaderboards"
GOLDEN_DIR = EXPERIMENTS_DIR / "golden"
RESULTS_DIR = REPO_ROOT / "results"
OPERATOR_SUPPORT_RESULTS_DIR = RESULTS_DIR / "npu" / "operator_support"
LATENCY_RESULTS_DIR = RESULTS_DIR / "npu" / "latency"
THROUGHPUT_RESULTS_DIR = RESULTS_DIR / "npu" / "throughput"
NPU_DIR = EXPERIMENTS_DIR / "npu"
CAMERA_DIR = EXPERIMENTS_DIR / "camera"

BASELINE_RUN = "ppe5_yolov8n_baseline_100ep_s0"

# Board and analysis experiments, formerly runs/<run_id>/bench/<name>/. name -> (topic, per_run).
# per_run is True when the experiment was run on several trained models: the directory then holds one <run_id>/ subdirectory each.
_EXPERIMENTS = {
    "eval_g4": ("operator_support", False),
    "eval_x7": ("operator_support", False),
    "eval_x7b": ("operator_support", False),
    "onboard_g4b": ("operator_support", True),
    "onboard_g4v": ("operator_support", True),
    "onboard_g4v2": ("operator_support", True),
    "onboard_q2c": ("operator_support", False),
    "onboard_q2e": ("operator_support", False),
    "onboard_q2e_qat": ("operator_support", False),
    "onboard_q2h": ("operator_support", True),
    "onboard_q2i": ("operator_support", True),
    "onboard_q2j": ("operator_support", False),
    "analysis_x10": ("latency", False),
    "onboard": ("latency", False),
    "onboard_e2": ("latency", False),
    "onboard_e2b": ("latency", False),
    "onboard_e2c": ("latency", False),
    "onboard_e2d": ("latency", False),
    "onboard_e2e": ("latency", False),
    "onboard_x1": ("latency", False),
    "onboard_x2": ("latency", False),
    "onboard_x3": ("latency", False),
    "onboard_x4": ("latency", False),
    "onboard_x5": ("latency", False),
    "onboard_x6": ("latency", False),
    "onboard_x8": ("latency", False),
    "onboard_x9": ("latency", False),
    "onboard_q2f_split": ("throughput", False),
    "onboard_q2f_thr": ("throughput", False),
    "onboard_thr": ("throughput", False),
}


def bench_dir(run) -> Path:
    """Pipeline outputs of a run (tier1/tier2/gpu/camera files). run is a run_id or the run directory."""
    return RUNS_DIR / Path(run).name / "bench"


def experiment_dir(run, name: str) -> Path:
    """Directory of experiment `name` for a run. Single-run experiments belong to BASELINE_RUN only."""
    topic, per_run = _EXPERIMENTS[name]
    run_id = Path(run).name
    base = NPU_DIR / topic / name
    if per_run:
        return base / run_id
    if run_id != BASELINE_RUN:
        raise ValueError(f"experiment {name} exists for {BASELINE_RUN} only, not {run_id}")
    return base


def bench_path(run, rel: str = "") -> Path:
    """Resolve a path relative to the run's bench directory, where its first component may be an experiment (onboard_x4/p2)."""
    first, _, rest = rel.partition("/")
    if first in _EXPERIMENTS:
        return experiment_dir(run, first) / rest if rest else experiment_dir(run, first)
    return bench_dir(run) / rel if rel else bench_dir(run)
