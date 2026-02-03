import os
from pathlib import Path

def get_project_root() -> Path:
    """---
    Returns the absolute path to the project root directory.
    ---"""
    return Path(__file__).parent.parent

def get_src_root() -> Path:
    """---
    Returns the absolute path to the source code directory.
    ---"""
    return Path(__file__).parent