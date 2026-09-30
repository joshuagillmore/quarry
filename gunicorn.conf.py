"""Gunicorn server config for Quarry.

IMPORTANT: exactly 1 worker. The live job/mission trace (jobs.py's in-memory
_store) is process-global state, so more than one worker would split it and
the UI would show a different job list depending on which worker answered.
gthread keeps the arbiter's heartbeat off request threads, so a long-lived
SSE stream or a slow crawl request is not killed by the worker timeout.
"""

bind = "0.0.0.0:5000"
workers = 1
worker_class = "gthread"
threads = 16
timeout = 120
graceful_timeout = 30
accesslog = "-"
errorlog = "-"


def post_worker_init(worker):
    """Runs once in the single worker process, right after it forks — the
    natural home for startup that used to live behind app.py's ensure_db
    before_request hook (init_db, reconcile_interrupted_missions,
    start_scheduler). Doing it here means the first real request never races
    the double-checked lock in ensure_db, and the scheduler is guaranteed
    running before gunicorn reports the worker healthy."""
    from app import initialize
    initialize()
