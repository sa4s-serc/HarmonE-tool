"""Session identity for the Versioned Model Repository, shared by both managed systems.

HarmonE's VMR is meant to be scoped to a single run of the managed system: a
session retrains, stores model+data pairs, and later reuses one of *its own*
versions when drift realigns with a distribution it already saw (the paper
induces drift twice for exactly this). Letting the VMR accumulate across
sessions breaks that - a fresh run could "reuse" a model trained in some
earlier, unrelated run, and version numbers climb forever instead of
describing the run you're looking at.

So versions live under versionedMR/<session>/... and numbering restarts at 1
each session. The session id is the MLflow session run that
run_managed_system.py creates per session and writes to
knowledge/mlflow_session_run_id.txt - already the cross-process handle
execute.py uses, so retrain/analyse just read the same file rather than
inventing a second notion of "which run is this".
"""
import os

SESSION_RUN_ID_FILENAME = "mlflow_session_run_id.txt"

# Used when there is no managed-system session at all - a bare `python
# retrain.py` or `mlflow run ... -e retrain`. Those still need somewhere to
# write, and keeping them out of the real sessions' namespaces means a manual
# debugging retrain can't pollute a session's reuse candidates.
STANDALONE_SESSION = "standalone"


def current_session_id(knowledge_dir):
    """The VMR namespace for the session this process belongs to.

    knowledge_dir: absolute path to the managed system's knowledge/ directory.
    """
    try:
        with open(os.path.join(knowledge_dir, SESSION_RUN_ID_FILENAME)) as f:
            run_id = f.read().strip()
    except (FileNotFoundError, OSError):
        return STANDALONE_SESSION
    # Short prefix keeps paths readable; MLflow run ids are hex and unique
    # enough at 12 chars for the handful of sessions a VMR ever holds.
    return f"session-{run_id[:12]}" if run_id else STANDALONE_SESSION


def session_versioned_dir(versioned_root, knowledge_dir):
    """versionedMR root for this session, created if absent."""
    path = os.path.join(versioned_root, current_session_id(knowledge_dir))
    os.makedirs(path, exist_ok=True)
    return path
