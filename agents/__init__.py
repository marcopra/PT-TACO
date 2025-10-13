"""
Modular Agent Architecture

This package contains a modular and extensible architecture for RL agents.
Base classes provide common functionality, and specialized agents inherit from them.
"""

from .base_agent import BaseAgent
from .base_taco_agent import BaseTACOAgent
from .drqv2_agent import DrQV2Agent
from .taco_agent import TACOAgent
from .taco_proprio_agent import TACOProprioAgent
from .taco_td3bc_agent import TACOTD3BCAgent

__all__ = [
    'BaseAgent',
    'BaseTACOAgent',
    'DrQV2Agent',
    'TACOAgent',
    'TACOProprioAgent',
    'TACOTD3BCAgent',
]
