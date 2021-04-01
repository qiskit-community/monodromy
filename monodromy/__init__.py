"""
monodromy/__init__.py

Top-level imports for `monodromy`.
"""

import monodromy.coverage
import monodromy.polytopes

from monodromy.backend.lrs import LRSBackend
import monodromy.backend

monodromy.backend.backend = LRSBackend()
