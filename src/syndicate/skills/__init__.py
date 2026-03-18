"""
Skills module for Syndicate AI agentic framework.

Provides domain expertise modules (SkillModules) that agents can plug in
to gain specialized knowledge without training/fine-tuning.
"""

from .skill_module import SkillModule, create_skill_module
from .registry import SkillRegistry
from .rag_skill import KnowledgeBaseSkill

__all__ = [
    'SkillModule',
    'create_skill_module',
    'SkillRegistry',
    'KnowledgeBaseSkill',
]
