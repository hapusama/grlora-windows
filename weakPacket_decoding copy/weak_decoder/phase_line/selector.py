"""Compatibility exports for the legacy monolithic phase-line selector.

The implementation now lives under ``variants/_legacy_core`` so new Stage-2
families can be isolated by directory.  This module keeps the old import path
working for existing diagnostics that reach into ``phase_line.selector``.
"""

from .variants._legacy_core import selector as _legacy_selector

for _name, _value in vars(_legacy_selector).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

__all__ = list(getattr(_legacy_selector, "__all__", ()))

del _legacy_selector, _name, _value
