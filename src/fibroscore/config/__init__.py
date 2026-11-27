# ==============================================================================
# File    : __init__.py
# Purpose : Initialize configuration object and class
# Author  : Sophia Mengjia Li
# Created : 2026-11-27
# ==============================================================================

from fibroscore.config.loader import load_config, section
from fibroscore.config.schema import Config

__all__ = ["Config", "load_config", "section"]

# [END]