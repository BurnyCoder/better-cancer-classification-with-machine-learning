"""Timestamped logging to `logs/<timestamp>_<phase>.log` and the terminal at the same time.

Global context: every CLI phase calls `setup_logging(phase)` once; modules use
`logging.getLogger(__name__)` so each line carries time, module and level.
"""

import logging
import sys
from datetime import datetime

from pcam_hc.config import PATHS

FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(phase: str) -> logging.Logger:
    """Attach a file handler (one file per run) and a stdout handler to the root logger."""
    PATHS.logs.mkdir(parents=True, exist_ok=True)
    log_file = PATHS.logs / f"{datetime.now():%Y%m%d-%H%M%S}_{phase}.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()  # avoid duplicate lines when phases run in one process
    for handler in (logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(handler)
    root.info("logging to %s", log_file)
    return root
