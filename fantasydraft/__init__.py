"""Fantasy draft board - research pipeline and keyboard draft dashboard.

The modules here import each other flatly (`from common import ...`) because the
package began life as a directory of scripts on sys.path. Putting the package's
own directory back on sys.path keeps those imports working once installed, so
the code reads identically whether it is run from a checkout or from a wheel.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_HERE = str(_Path(__file__).resolve().parent)
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

__version__ = "0.1.0"
