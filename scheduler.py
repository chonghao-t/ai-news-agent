import logging
import signal
import sys
from datetime import datetime, timezone
from apscheduler.triggers.cron import CronTrigger

import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger


def load_schedule_config() -> tuple[str, str]:
    """
    Read schedule_cron and timezone from config.yaml.
    Returns (cron_expression, timezone_string).
    Falls back to safe defaults if config is unreadable.
    """
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        cron     = config.get("schedule_cron", "0 7 * * *")
        timezone = config.get("timezone",      "Asia/Singapore")
        return cron, timezone
    except Exception as e:
        logging.warning(f"scheduler: could not read config.yaml ({e}), using defaults")
        return "0 7 * * *", "Asia/Singapore"


def run_pipeline_job():
    """
    The job function APScheduler calls on each trigger.
    Runs the full pipeline and logs the outcome.
    """
    logger = logging.getLogger(__name__)
    start  = datetime.now(timezone.utc)
    logger.info(f"Scheduler triggered pipeline at {start.isoformat()}")

    try:
        from pipeline import run_pipeline
        state  = run_pipeline()
        errors = state.get("errors", [])
        items  = len(state.get("summarized", []))

        duration = (datetime.now(timezone.utc) - start).total_seconds()

        if errors:
            logger.warning(
                f"Pipeline completed with {len(errors)} error(s) in "
                f"{duration:.1f}s — {items} articles sent"
            )
            for e in errors:
                logger.warning(f"  {e}")
        else:
            logger.info(
                f"Pipeline completed successfully in {duration:.1f}s "
                f"— {items} articles sent"
            )

    except Exception as e:
        logger.error(f"Pipeline raised unhandled exception: {e}", exc_info=True)


def main():
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s %(levelname)-8s %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    cron_expr, tz = load_schedule_config()

    scheduler = BlockingScheduler(timezone=tz)

    trigger = CronTrigger.from_crontab(cron_expr, timezone=tz)

    scheduler.add_job(
        run_pipeline_job,
        trigger=trigger,
        id="morning_digest",
        name="Morning News Digest",
        misfire_grace_time=600,
        coalesce=True,
        replace_existing=True,
    )

    logger.info("Scheduler configured")
    logger.info(f"Cron: {cron_expr} ({tz})")
    logger.info("Registered jobs:")
    scheduler.print_jobs()
    logger.info("Press Ctrl+C to stop")

    logger.info(f"Scheduler started")
    logger.info(f"Cron: {cron_expr} ({tz})")
    logger.info("Press Ctrl+C to stop")

    # Graceful shutdown on SIGTERM (sent by systemd on stop)
    def handle_sigterm(signum, frame):
        logger.info("SIGTERM received — shutting down scheduler")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down scheduler")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()