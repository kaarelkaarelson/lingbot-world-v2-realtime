"""`lingbot` CLI: play (local window), bench, clip. See README.md.

The paper's code in reference/ imports itself as the top-level package `wan`, so reference/ goes on
the import path here, once; everything in lingbot/ then imports the paper's modules as `wan.*`.
"""
import os as _os
import sys as _sys

_REFERENCE = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "reference")
if _REFERENCE not in _sys.path:
    _sys.path.insert(0, _REFERENCE)
