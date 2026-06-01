"""Explicit warnings and errors for recoverable fallbacks (no silent degradation)."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SARExplicitFallbackWarning(UserWarning):
    """Recoverable degradation (parse fallback, synthesis fallback, etc.)."""


def warn_explicit(
    message: str,
    *,
    component: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Log at WARNING and surface a ``UserWarning`` for notebooks/CI visibility."""
    prefix = f"[langgraph_sar:{component}]"
    detail_suffix = ""
    if details:
        detail_suffix = " — " + ", ".join(f"{key}={value!r}" for key, value in details.items())
    full = f"{prefix} {message}{detail_suffix}"
    logger.warning(full)
    warnings.warn(full, SARExplicitFallbackWarning, stacklevel=3)


def raise_explicit(
    message: str,
    *,
    component: str,
    cause: Optional[BaseException] = None,
) -> None:
    """Raise a clear runtime error (optional chained cause)."""
    prefix = f"[langgraph_sar:{component}]"
    full = f"{prefix} {message}"
    if cause is not None:
        raise RuntimeError(full) from cause
    raise RuntimeError(full)
