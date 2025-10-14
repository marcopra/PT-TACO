"""
Modular Agent Architecture

This package contains a modular and extensible architecture for RL agents.
Base classes provide common functionality, and specialized agents inherit from them.
"""

from .base_agent import BaseAgent
from .drqv2 import DrQV2Agent

__all__ = [
    'BaseAgent',
    'DrQV2Agent',
]
