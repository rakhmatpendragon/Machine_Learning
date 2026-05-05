"""
logger.py
---------
Shared logger - import get_logger() anywhere in the project.
Uses Python's built-in logging so there are no extra dependencies.
"""

import logging
import sys


def get_logger(name: str = "retail_eda") -> logging.Logger:
    """
    Return a logger that writes INFO+ to stdout with a consistent format.
    Calling this multiple times with the same name returns the same logger
    (handler are not duplicated).
    """
    logger = logging.getLogger(name)

    if not logger.handlers:                     # avoid duplicate handlers
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)

        fmt = logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger
