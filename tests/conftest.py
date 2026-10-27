import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("code_dispatcher", os.path.join("tools", "e2e")):
    sys.path.insert(0, os.path.join(ROOT, p))
