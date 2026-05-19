"""
utils.py
--------
Shared utility helpers used across every pipeline module.

Why this file exists
--------------------
Previously each module (eda, preprocessor, clustering, cluster_interpreter,
classifier) contained its own private copy of _banner() and _save_fig().
That is a DRY violation: if you want to change the banner width or the
figure DPI you had to edit five different files.

All modules now import from here instead:

    from src.utils import banner, save_fig

Public API
----------
banner(title, logger, width)   — print a section header to the logger
save_fig(fig, path, dpi, tight) — save a matplotlib Figure to disk and close it
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from src.config import PLOT_DPI


def banner(title: str, logger: logging.Logger, width: int = 70) -> None:
    """
    Log a prominent section-header banner.

    Parameters
    ----------
    title  : str              — section name (will be upper-cased)
    logger : logging.Logger   — the calling module's logger
    width  : int              — total banner width in characters (default 70)

    Example output
    --------------
    ======================================================================
      STEP 3 — TRAIN DECISION TREE
    ======================================================================
    """
    logger.info("")
    logger.info("=" * width)
    logger.info("  %s", title.upper())
    logger.info("=" * width)


def save_fig(
    fig: plt.Figure,
    path: Path,
    dpi: int = PLOT_DPI,
    tight: bool = True,
    logger: logging.Logger | None = None,
) -> None:
    """
    Save a matplotlib Figure to *path* and close it.

    Parameters
    ----------
    fig    : plt.Figure       — the figure to save
    path   : Path             — destination file (parent dirs created if needed)
    dpi    : int              — resolution in dots-per-inch (default from config)
    tight  : bool             — call tight_layout() before saving (default True)
    logger : Logger | None    — if provided, logs the saved path at INFO level

    Notes
    -----
    Always closes the figure after saving to avoid memory leaks in
    long-running pipelines or test suites.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    if logger:
        logger.info("Plot saved → %s", path)
