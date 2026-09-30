"""The job store must stay bounded: cap in-flight jobs, evict old traces."""
import time

import pytest

import jobs


@pytest.fixture(autouse=True)
def clean_store():
    with jobs._lock:
        jobs._store.clear()
        jobs._recent_job_ids.clear()
    yield
    with jobs._lock:
        jobs._store.clear()
        jobs._recent_job_ids.clear()


def test_active_job_cap():
    for _ in range(jobs.MAX_ACTIVE_JOBS):
        jobs.create_job("q", 5, False, "")
    with pytest.raises(jobs.JobLimitReached):
        jobs.create_job("one too many", 5, False, "")
    with pytest.raises(jobs.JobLimitReached):
        jobs.create_mission_job("mission too", 10)


def test_finished_jobs_do_not_count_toward_cap():
    ids = [jobs.create_job("q", 5, False, "") for _ in range(jobs.MAX_ACTIVE_JOBS)]
    jobs.finish_job(ids[0])
    jobs.create_job("fits now", 5, False, "")  # must not raise


def test_old_finished_traces_are_evicted():
    jid = jobs.create_job("ancient", 5, False, "")
    jobs.finish_job(jid)
    with jobs._lock:  # age it past the TTL (keyed on when it finished)
        jobs._store[jid].started_at = time.time() - (jobs._DONE_TTL_S + 120)
        jobs._store[jid].finished_at = time.time() - (jobs._DONE_TTL_S + 60)
    jobs.create_job("fresh", 5, False, "")  # create() prunes
    assert jobs.get_job(jid) is None, "aged-out finished trace must be evicted"


def test_done_keep_window():
    keep = jobs._DONE_KEEP
    ids = []
    for i in range(keep + 5):
        jid = jobs.create_job(f"q{i}", 5, False, "")
        jobs.finish_job(jid)
        ids.append(jid)
    jobs.create_job("trigger prune", 5, False, "")
    done_left = sum(1 for j in jobs._store.values() if j.done)
    assert done_left <= keep


def test_running_jobs_never_evicted():
    jid = jobs.create_job("running", 5, False, "")
    with jobs._lock:  # even if ancient
        jobs._store[jid].started_at = time.time() - (jobs._DONE_TTL_S * 10)
    # fill and churn
    for i in range(3):
        j2 = jobs.create_job(f"x{i}", 5, False, "")
        jobs.finish_job(j2)
    jobs.create_job("churn", 5, False, "")
    assert jobs.get_job(jid) is not None
