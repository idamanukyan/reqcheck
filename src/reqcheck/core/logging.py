"""Structured logging configuration for reqcheck.

This module provides:
- Structured JSON logging for production environments
- Human-readable colored output for development
- Context managers for adding request context
- Performance timing utilities
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Generator

# Context variables for request tracking
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
requirement_id_var: ContextVar[str | None] = ContextVar("requirement_id", default=None)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging output."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add context from context variables
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id

        requirement_id = requirement_id_var.get()
        if requirement_id:
            log_data["requirement_id"] = requirement_id

        # Add source location for debugging
        log_data["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "taskName", "message",
            ):
                log_data[key] = value

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class DevFormatter(logging.Formatter):
    """Human-readable formatter for development with colors."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    DIM = "\033[2m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human readability."""
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Build context string
        context_parts = []
        request_id = request_id_var.get()
        if request_id:
            context_parts.append(f"req={request_id[:8]}")
        requirement_id = requirement_id_var.get()
        if requirement_id:
            context_parts.append(f"req_id={requirement_id}")

        context_str = f" [{', '.join(context_parts)}]" if context_parts else ""

        # Format message
        message = record.getMessage()

        # Build the final output
        level_str = f"{color}{record.levelname:7}{self.RESET}"
        location = f"{self.DIM}{record.filename}:{record.lineno}{self.RESET}"

        output = f"{self.DIM}{timestamp}{self.RESET} {level_str} {location}{context_str} {message}"

        # Add exception if present
        if record.exc_info:
            output += f"\n{self.formatException(record.exc_info)}"

        return output


def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
    stream: Any = None,
) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, use JSON formatting; otherwise use dev formatting
        stream: Output stream (defaults to sys.stderr)
    """
    if stream is None:
        stream = sys.stderr

    # Get the root logger for reqcheck
    root_logger = logging.getLogger("reqcheck")
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create handler with appropriate formatter
    handler = logging.StreamHandler(stream)
    handler.setLevel(getattr(logging, level.upper()))

    if json_output:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(DevFormatter())

    root_logger.addHandler(handler)

    # Prevent propagation to root logger
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the reqcheck namespace.

    Args:
        name: Logger name (will be prefixed with 'reqcheck.')

    Returns:
        Configured logger instance
    """
    if not name.startswith("reqcheck."):
        name = f"reqcheck.{name}"
    return logging.getLogger(name)


@contextmanager
def log_context(
    request_id: str | None = None,
    requirement_id: str | None = None,
) -> Generator[None, None, None]:
    """Context manager to add logging context.

    Args:
        request_id: Optional request ID for tracing
        requirement_id: Optional requirement ID being processed
    """
    tokens = []

    if request_id is not None:
        tokens.append(request_id_var.set(request_id))
    if requirement_id is not None:
        tokens.append(requirement_id_var.set(requirement_id))

    try:
        yield
    finally:
        for token in tokens:
            if request_id is not None:
                request_id_var.reset(tokens[0])
            if requirement_id is not None:
                requirement_id_var.reset(tokens[-1])


@contextmanager
def log_timing(
    logger: logging.Logger,
    operation: str,
    level: int = logging.DEBUG,
) -> Generator[dict[str, Any], None, None]:
    """Context manager to log operation timing.

    Args:
        logger: Logger to use
        operation: Description of the operation
        level: Logging level for timing messages

    Yields:
        Dictionary that will be populated with timing info
    """
    timing_info: dict[str, Any] = {"operation": operation}
    start_time = time.perf_counter()

    logger.log(level, f"Starting: {operation}")

    try:
        yield timing_info
        timing_info["success"] = True
    except Exception as e:
        timing_info["success"] = False
        timing_info["error"] = str(e)
        raise
    finally:
        elapsed = time.perf_counter() - start_time
        timing_info["duration_ms"] = round(elapsed * 1000, 2)

        status = "completed" if timing_info.get("success") else "failed"
        logger.log(
            level,
            f"Finished: {operation} ({status} in {timing_info['duration_ms']:.2f}ms)",
            extra={"timing": timing_info},
        )


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds structured fields to all log records."""

    def process(
        self, msg: str, kwargs: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Process the logging call to inject extra context."""
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def get_structured_logger(
    name: str,
    **default_fields: Any,
) -> LoggerAdapter:
    """Get a logger with default structured fields.

    Args:
        name: Logger name
        **default_fields: Fields to include in every log record

    Returns:
        LoggerAdapter with default fields
    """
    logger = get_logger(name)
    return LoggerAdapter(logger, default_fields)
