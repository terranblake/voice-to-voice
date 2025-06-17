"""
Test configuration for pytest.
"""

import sys
import os
from pathlib import Path

# Add src to Python path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Set up test environment variables
os.environ.setdefault("HF_TOKEN", "test_token_for_testing")
os.environ.setdefault("LOG_LEVEL", "DEBUG")
