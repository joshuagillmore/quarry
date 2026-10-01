"""scheduler.py: cron triggers run in UTC, a scheduled launch never overlaps
an agent's in-flight mission, and a failed launch never leaks a job slot."""
import asyncio
import json

import pytest

import agent_runner
import jobs
import scheduler
import storage
from models import Agent, Mission


def _init(active=1, question="What changed?"):
    async def go():
        await storage.init_db()
        await storage.insert_agent(Agent(
            id="a1", name="N", expertise="x", persona_prompt="p",
            schedule_cron="0 7 * * *", schedule_question=question,
            active=active, created_at="t"))
    asyncio.run(go())


def _add_mission(mid, status, finished_at=None, question="What changed?"):
    asyncio.run(storage.insert_mission(Mission(
        id=mid, agent_id="a1", question=question, status=status,
        created_at="t", finished_at=finished_at)))


class _Started(list):
    """The mission ids start_planning was called with; `job_ids` holds the
    job id passed alongside each."""

    def __init__(self):
        super().__init__()
        self.job_ids = []

    def __call__(self, mission_id, job_id=None):
        self.append(mission_id)
        self.job_ids.append(job_id)


@pytest.fixture
def started(monkeypatch):
    calls = _Started()
    monkeypatch.setattr(agent_runner, "start_planning", calls)
    return calls


def _active_jobs():
    with jobs._lock:
        return sum(1 for j in jobs._store.values() if not j.done)


def test_launch_returns_the_mission_id(started):
    _init()
    _add_mission("P", "done", finished_at="2026-01-01")
    mid = scheduler.launch_scheduled_mission("a1")
    assert mid and started == [mid]
    m = asyncio.run(storage.get_mission(mid))
    assert m.parent_mission_id == "P"
    assert json.loads(m.budget_json)["auto_approve"] is True
    assert jobs.get_job(m.job_id) is not None


def test_launch_skips_when_a_mission_is_active(started):
    _init()
    _add_mission("busy", "collecting")
    assert scheduler.launch_scheduled_mission("a1") is None
    assert started == []
    assert jobs.get_in_memory_job_ids() == set(), "no job slot taken"


@pytest.mark.parametrize("kwargs", [{"active": 0}, {"question": "  "}])
def test_launch_skips_inactive_or_questionless_agents(started, kwargs):
    _init(**kwargs)
    assert scheduler.launch_scheduled_mission("a1") is None
    assert started == []


def test_launch_unknown_agent(started):
    _init()
    assert scheduler.launch_scheduled_mission("nope") is None


def test_launch_propagates_job_limit(started):
    _init()
    for _ in range(jobs.MAX_ACTIVE_JOBS):
        jobs.create_job("q", 5, False, "")
    with pytest.raises(jobs.JobLimitReached):
        scheduler.launch_scheduled_mission("a1")
    assert asyncio.run(storage.list_missions()) == []
    assert started == []


def test_insert_failure_finishes_the_job(monkeypatch, started):
    _init()

    async def broken_insert(mission):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(storage, "insert_mission", broken_insert)
    with pytest.raises(RuntimeError):
        scheduler.launch_scheduled_mission("a1")
    assert _active_jobs() == 0
    assert all(j.stage == "error" for j in jobs._store.values())
    assert started == []


def test_scheduler_thread_survives_launch_failures(monkeypatch):
    def full(agent_id):
        raise jobs.JobLimitReached("full")

    monkeypatch.setattr(scheduler, "launch_scheduled_mission", full)
    scheduler._run_scheduled_agent("a1")  # must not raise


def test_cron_triggers_are_utc(monkeypatch):
    _init()
    seen = []
    real = scheduler.CronTrigger.from_crontab

    def spy(expr, timezone=None):
        seen.append(timezone)
        return real(expr, timezone=timezone)

    monkeypatch.setattr(scheduler.CronTrigger, "from_crontab", staticmethod(spy))

    class FakeScheduler:
        def __init__(self):
            self.triggers = []

        def get_jobs(self):
            return []

        def remove_job(self, job_id):
            pass

        def add_job(self, func, trigger, **kwargs):
            self.triggers.append(trigger)

    fake = FakeScheduler()
    monkeypatch.setattr(scheduler, "_scheduler", fake)

    assert scheduler.validate_cron("0 7 * * *")[0]
    assert scheduler.describe_next_run("0 7 * * *").endswith("07:00 UTC")
    assert scheduler.sync_agent_jobs() == 1

    assert seen and all(tz == "UTC" for tz in seen), seen
    assert str(fake.triggers[0].timezone) == "UTC"


def test_mission_construction_failure_finishes_the_job(monkeypatch, started):
    _init()

    def broken_mission(**kwargs):
        raise ValueError("bad budget")

    monkeypatch.setattr(scheduler, "Mission", broken_mission)
    with pytest.raises(ValueError):
        scheduler.launch_scheduled_mission("a1")
    assert _active_jobs() == 0
    assert [j.stage for j in jobs._store.values()] == ["error"]
    assert started == []


def test_start_planning_failure_finishes_job_and_mission(monkeypatch):
    _init()

    def cannot_start(mission_id, job_id=None):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(agent_runner, "start_planning", cannot_start)
    with pytest.raises(RuntimeError):
        scheduler.launch_scheduled_mission("a1")
    assert _active_jobs() == 0
    (mission,) = asyncio.run(storage.list_missions())
    # No worker will ever move it on; left in `planning` it would also block
    # every later scheduled run of this agent (agent_has_active_mission).
    assert mission.status == "error" and mission.finished_at
    assert jobs.get_job(mission.job_id).stage == "error"
    assert not asyncio.run(storage.agent_has_active_mission("a1"))


def test_launch_hands_its_job_to_the_worker(started):
    """The worker finishes the job it was given even if the mission row is
    deleted before it reads it, so the launch must pass it along."""
    _init()
    mid = scheduler.launch_scheduled_mission("a1")
    m = asyncio.run(storage.get_mission(mid))
    assert started.job_ids == [m.job_id] and m.job_id
