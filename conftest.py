import sys
from pathlib import Path

# Make the repo root importable so tests can `import data.prices` etc.
sys.path.insert(0, str(Path(__file__).resolve().parent))
