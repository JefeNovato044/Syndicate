# core/skills/registry.py
"""
SkillRegistry - Central registry for discovering and managing skill modules.
"""

from typing import Dict, List, Optional
from .skill_module import SkillModule


class SkillRegistry:
    """
    Central registry for skill modules.
    
    Allows skills to be registered and discovered by name,
    enabling dynamic skill composition for agents.
    
    Thread safety:
        This is a process-global singleton. The lazy init of ``_skills``
        is safe under a single-threaded asyncio event loop but is NOT
        safe if ``register()`` / ``get()`` are called from multiple OS
        threads concurrently.  If you need multi-threaded access, wrap
        calls with a ``threading.Lock``.
    
    Testing:
        Call ``SkillRegistry.clear()`` in test fixtures (setup/teardown)
        to prevent cross-test state contamination.
    """
    
    _skills: Dict[str, SkillModule] = None  # Initialized on first access
    
    @classmethod
    def _get_skills(cls) -> Dict[str, SkillModule]:
        if cls._skills is None:
            cls._skills = {}
        return cls._skills
    
    @classmethod
    def register(cls, skill: SkillModule) -> None:
        """Register a skill module."""
        cls._get_skills()[skill.name] = skill
    
    @classmethod
    def get(cls, name: str) -> Optional[SkillModule]:
        """Get a skill by name."""
        return cls._get_skills().get(name)
    
    @classmethod
    def get_or_raise(cls, name: str) -> SkillModule:
        """Get a skill by name, raising if not found."""
        skill = cls.get(name)
        if skill is None:
            available = list(cls._get_skills().keys())
            raise KeyError(f"Skill '{name}' not found. Available: {available}")
        return skill
    
    @classmethod
    def list_skills(cls) -> List[str]:
        """List all registered skill names."""
        return list(cls._get_skills().keys())
    
    @classmethod
    def get_all(cls) -> List[SkillModule]:
        """Get all registered skills."""
        return list(cls._get_skills().values())
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """Remove a skill from registry."""
        if name in cls._get_skills():
            del cls._get_skills()[name]
            return True
        return False
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered skills."""
        cls._get_skills().clear()
