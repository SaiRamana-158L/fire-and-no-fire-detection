import os
import sys
from pathlib import Path

# Add project root to sys.path so app module and dependencies are resolved
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app import app
