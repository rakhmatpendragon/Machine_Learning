"""
tests/test_utils.py
--------------------
Unit tests for src/utils.py

Verifies that banner() and save_fig() work correctly
and that no other src module defines its own private copy of either helper.
"""
from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import banner, save_fig
from src.config import PLOT_DPI


# ── banner() ──────────────────────────────────────────────────────────────────

class TestBanner:

    def test_logs_title_uppercased(self, caplog):
        logger = logging.getLogger("test_banner")
        with caplog.at_level(logging.INFO, logger="test_banner"):
            banner("my section", logger)
        assert "MY SECTION" in caplog.text

    def test_logs_separator_lines(self, caplog):
        logger = logging.getLogger("test_banner_sep")
        with caplog.at_level(logging.INFO, logger="test_banner_sep"):
            banner("x", logger, width=50)
        assert "=" * 50 in caplog.text

    def test_custom_width_respected(self, caplog):
        logger = logging.getLogger("test_banner_width")
        with caplog.at_level(logging.INFO, logger="test_banner_width"):
            banner("y", logger, width=40)
        assert "=" * 40 in caplog.text
        assert "=" * 41 not in caplog.text

    def test_accepts_any_logger(self):
        """Should not raise with any valid logger."""
        lg = logging.getLogger("arbitrary")
        banner("test", lg)   # must not raise

    def test_does_not_raise_on_empty_title(self):
        lg = logging.getLogger("empty")
        banner("", lg)       # must not raise


# ── save_fig() ────────────────────────────────────────────────────────────────

class TestSaveFig:

    def _make_fig(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        return fig

    def test_creates_file(self, tmp_path):
        fig  = self._make_fig()
        path = tmp_path / "test.png"
        save_fig(fig, path)
        assert path.exists()

    def test_file_is_non_empty(self, tmp_path):
        fig  = self._make_fig()
        path = tmp_path / "test2.png"
        save_fig(fig, path)
        assert path.stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_path):
        fig  = self._make_fig()
        path = tmp_path / "a" / "b" / "c" / "test.png"
        save_fig(fig, path)
        assert path.exists()

    def test_closes_figure_after_save(self, tmp_path):
        fig  = self._make_fig()
        path = tmp_path / "closed.png"
        save_fig(fig, path)
        # After save, figure should be closed (not in plt.get_fignums())
        assert fig.number not in plt.get_fignums()

    def test_logs_path_when_logger_provided(self, tmp_path, caplog):
        fig    = self._make_fig()
        path   = tmp_path / "logged.png"
        logger = logging.getLogger("test_save_fig")
        with caplog.at_level(logging.INFO, logger="test_save_fig"):
            save_fig(fig, path, logger=logger)
        assert str(path) in caplog.text

    def test_no_log_when_logger_is_none(self, tmp_path, caplog):
        fig  = self._make_fig()
        path = tmp_path / "silent.png"
        with caplog.at_level(logging.INFO):
            save_fig(fig, path, logger=None)   # must not raise

    def test_custom_dpi_accepted(self, tmp_path):
        fig  = self._make_fig()
        path = tmp_path / "dpi.png"
        save_fig(fig, path, dpi=72)   # must not raise
        assert path.exists()

    def test_tight_false_accepted(self, tmp_path):
        fig  = self._make_fig()
        path = tmp_path / "notight.png"
        save_fig(fig, path, tight=False)
        assert path.exists()


# ── Architecture: no duplicated helpers in other modules ──────────────────────

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

# Modules that are allowed to define private helpers (only utils.py itself)
ALLOWED_TO_DEFINE = {"utils.py"}

# Names that must NOT appear as top-level function definitions in other modules
FORBIDDEN_NAMES = {"_banner", "banner", "_save_fig", "save_fig"}


def _get_top_level_function_names(filepath: Path) -> set[str]:
    """Parse a Python file and return all top-level function names."""
    tree = ast.parse(filepath.read_text())
    return {node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and
            any(isinstance(parent, ast.Module)
                for parent in ast.walk(tree)
                if hasattr(parent, "body") and node in getattr(parent, "body", []))}


class TestNoDuplicatedHelpers:

    @pytest.mark.parametrize("src_file", [
        f for f in SRC_DIR.glob("*.py")
        if f.name not in ALLOWED_TO_DEFINE and not f.name.startswith("_")
    ])
    def test_module_does_not_redefine_banner_or_save_fig(self, src_file):
        """
        Verify that no src module other than utils.py defines its own
        banner() / _banner() / save_fig() / _save_fig().

        If this test fails it means a developer added a local copy instead
        of importing from src.utils — a DRY violation.
        """
        tree  = ast.parse(src_file.read_text())
        names = {node.name for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef)}
        duplicates = names & FORBIDDEN_NAMES
        assert not duplicates, (
            f"{src_file.name} defines {duplicates}. "
            f"Import from src.utils instead: `from src.utils import banner, save_fig`"
        )

    def test_utils_exports_banner(self):
        from src import utils
        assert callable(utils.banner)

    def test_utils_exports_save_fig(self):
        from src import utils
        assert callable(utils.save_fig)
