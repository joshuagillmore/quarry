"""Cron scheduling for agents — the "morning brief".

An agent with a `schedule_cron` runs unattended: it plans, approves its own plan
(nobody is at the keyboard), collects, and writes a brief that leads with what
is new since its previous run on the same question.

Single-process by design. The Docker image runs gunicorn with exactly one
worker because the live job trace is in-process memory; that same constraint
means exactly one scheduler, so a cron window fires once.

Cron expressions are evaluated in UTC (every trigger is built with
timezone="UTC"), matching the "HH:MM UTC" the UI shows for the next run.
"""
import asyncio
import json
import sys
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from models import Mission

_scheduler: BackgroundScheduler | None = None
_lock = threading.Lock()
# Serialises _launch's "is a mission already running?" check with the insert
# that makes one run, so a cron fire racing a "Run now" click cannot both pass
# the check and start two overlapping missions for the same agent.
_launch_lock = threading.Lock()
JOB_PREFIX = "agent:"


def _log(msg: str) -> None:
    print(f"[SCHED] {msg}", file=sys.stderr, flush=True)


def validate_cron(expr: str) -> tuple[bool, str]:
    """Check a 5-field crontab expression without registering anything."""
    expr = (expr or "").strip()
    if not expr:
        return True, ""  # empty simply means "not scheduled"
    try:
        CronTrigger.from_crontab(expr, timezone="UTC")
        return True, ""
    except Exception as e:
        return False, str(e)


def describe_next_run(expr: str) -> str:
    ok, _ = validate_cron(expr)
    if not ok or not expr.strip():
        return ""
    try:
        trigger = CronTrigger.from_crontab(expr, timezone="UTC")
        nxt = trigger.get_next_fire_time(None, datetime.now(timezone.utc))
        return nxt.strftime("%Y-%m-%d %H:%M UTC") if nxt else ""
    except Exception:
        return ""


def _run_scheduled_agent(agent_id: str) -> None:
    """Fired by APScheduler on its own thread (and by 'Run now'). Creates a
    mission that will auto-approve, then hands off to the normal collection
    machinery. A scheduler thread must never die, so every failure is logged
    and swallowed here."""
    from jobs import JobLimitReached

    try:
        launch_scheduled_mission(agent_id)  # logs its own reason for a None
    except JobLimitReached as e:
        _log(f"agent {agent_id}: skipped this run, job store is full ({e})")
    except Exception as e:  # noqa: BLE001
        _log(f"agent {agent_id} failed to launch: {type(e).__name__}: {e}")


def launch_scheduled_mission(agent_id: str) -> Optional[str]:
    """Start a scheduled (auto-approving) mission for an agent now, and
    return its id. Synchronous: call it from a thread with no running event
    loop (a request handler, the scheduler's thread). Returns None when the
    agent is missing or inactive, has no standing question, or already has a
    mission in flight. Raises jobs.JobLimitReached when the job store is
    full."""
    return asyncio.run(_launch(agent_id))


async def _launch(agent_id: str) -> Optional[str]:
    from storage import (get_agent, insert_mission, update_mission,
                         get_latest_finished_mission,
                         agent_has_active_mission)
    from jobs import create_mission_job, finish_job
    from agent_runner import start_planning

    agent = await get_agent(agent_id)
    if not agent or not agent.active:
        _log(f"agent {agent_id} is missing or inactive; skipping")
        return None
    question = (agent.schedule_question or "").strip()
    if not question:
        _log(f"agent {agent.name} has a schedule but no question; skipping")
        return None

    job_id = None        # set once a job slot is taken
    inserted_id = None   # set once the mission row exists
    try:
        with _launch_lock:
            # Never overlap: APScheduler's max_instances only covers this
            # launch call (which returns in milliseconds), not the mission it
            # starts.
            if await agent_has_active_mission(agent_id):
                _log(f"agent {agent.name} already has a mission in flight; skipping")
                return None

            # Link to the previous run on the same question so the brief can diff.
            prior = await get_latest_finished_mission(agent_id, question, "")
            job_id = create_mission_job(question, agent.default_max_sources)  # may raise JobLimitReached
            mission = Mission(
                id=str(uuid.uuid4()), agent_id=agent_id, question=question,
                status="planning", job_id=job_id,
                parent_mission_id=prior.id if prior else None,
                budget_json=json.dumps({
                    "max_passes": agent.default_max_passes,
                    "max_sources": agent.default_max_sources,
                    "per_req_attempts": agent.default_per_req_attempts,
                    "auto_approve": True,
                    "scheduled": True,
                }),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            await insert_mission(mission)
            inserted_id = mission.id

        _log(f"launching '{question[:60]}' for agent {agent.name}")
        start_planning(mission.id)
        return mission.id
    except BaseException as e:
        # Anything that fails after the job slot is taken — building the
        # mission, recording it, or starting its worker — leaves nothing that
        # would ever finish the job, so release it here. A recorded mission
        # with no worker is closed too: left in `planning` it would block
        # every later run of this agent until a restart reconciled it.
        reason = f"scheduled launch failed: {type(e).__name__}"
        if job_id:
            finish_job(job_id, stage="error", error=reason)
        if inserted_id:
            try:
                await update_mission(inserted_id, status="error", error=reason,
                                     finished_at=datetime.now(timezone.utc).isoformat())
            except Exception as cleanup_error:  # noqa: BLE001 - keep the original error
                _log(f"could not mark mission {inserted_id} failed: {cleanup_error!r}")
        raise


def run_scheduled_agent_now(agent_id: str) -> None:
    """Fire a scheduled run immediately, on a worker thread so the request
    returns straight away. Failures are logged, not reported to the caller;
    launch_scheduled_mission is the variant that reports them."""
    threading.Thread(target=_run_scheduled_agent, args=(agent_id,), daemon=True).start()


def sync_agent_jobs() -> int:
    """Make the registered jobs match the agents table. Called at boot and
    whenever an agent is created, edited or deleted."""
    from storage import list_scheduled_agents

    sched = _scheduler
    if sched is None:
        return 0
    try:
        agents = asyncio.run(list_scheduled_agents())
    except Exception as e:  # noqa: BLE001
        print(f"[SCHED] could not load agents: {e}", file=sys.stderr, flush=True)
        return 0

    wanted = {}
    for a in agents:
        ok, _ = validate_cron(a.schedule_cron or "")
        if ok and (a.schedule_cron or "").strip() and (a.schedule_question or "").strip():
            wanted[JOB_PREFIX + a.id] = a

    for job in list(sched.get_jobs()):
        if job.id.startswith(JOB_PREFIX) and job.id not in wanted:
            sched.remove_job(job.id)

    for job_id, a in wanted.items():
        sched.add_job(
            _run_scheduled_agent, CronTrigger.from_crontab(a.schedule_cron, timezone="UTC"),
            args=[a.id], id=job_id, replace_existing=True,
            # Only stops two *launch calls* overlapping; the launch returns as
            # soon as the mission thread starts. Overlapping missions are
            # prevented by _launch's agent_has_active_mission check.
            max_instances=1,
            coalesce=True,         # a missed window fires once, not N times
            misfire_grace_time=3600,
        )
    return len(wanted)


def start_scheduler() -> None:
    global _scheduler
    with _lock:
        if _scheduler is not None:
            return
        sched = BackgroundScheduler(timezone="UTC")
        sched.start()
        _scheduler = sched
    n = sync_agent_jobs()
    print(f"[SCHED] scheduler started with {n} agent schedule(s)",
          file=sys.stderr, flush=True)


def scheduled_jobs() -> list[dict]:
    if _scheduler is None:
        return []
    out = []
    for job in _scheduler.get_jobs():
        if not job.id.startswith(JOB_PREFIX):
            continue
        out.append({
            "agent_id": job.id[len(JOB_PREFIX):],
            "next_run": job.next_run_time.strftime("%Y-%m-%d %H:%M UTC")
            if job.next_run_time else "",
        })
    return out
