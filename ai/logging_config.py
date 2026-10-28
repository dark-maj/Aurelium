from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formats log records as structured single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include standard exception info if present
        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        # Include custom structured extra attributes
        for key, val in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
            } and not key.startswith("_"):
                log_payload[key] = val

        return json.dumps(log_payload, default=str)


def setup_structured_logging(level: str = "INFO", json_format: bool = True) -> logging.Logger:
    """Configures root logger with structured formatting."""
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Avoid duplicate handlers
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
        )
    root.addHandler(handler)

    return logging.getLogger("aurelium.ai")
