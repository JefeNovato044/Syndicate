# core/skills/skill_module.py
"""
SkillModule - Domain expertise modules for AI agents.

Inspired by Cyberpunk 2077's neural skill chips: plug in a module,
instantly gain domain expertise without training/fine-tuning.

A SkillModule encapsulates:
- Domain knowledge (best practices, patterns, conventions)
- Glossary of key terms
- Bundled tools relevant to the domain
- MCP server references for external access
"""

from typing import List, Dict, Optional, Any, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..tools.base_tool import BaseTool


class SkillModule(BaseModel):
    """
    Encapsulates domain expertise for an agent.
    
    Like a neural implant that grants instant understanding of a domain.
    
    Example:
        kubernetes_skill = SkillModule(
            name="Kubernetes Expert",
            description="Deep expertise in Kubernetes cluster management",
            expertise='''
            You are an expert in Kubernetes. You know:
            - Deployment strategies (rolling, blue-green, canary)
            - Debugging with kubectl logs, describe, events
            - Resource management with requests/limits
            ...
            ''',
            capabilities=["Diagnose pod issues", "Review manifests"],
            glossary={"HPA": "Horizontal Pod Autoscaler"}
        )
    """
    
    name: str = Field(..., description="Skill module name")
    description: str = Field(..., description="Brief description of the skill")
    
    # Core knowledge - injected into system prompt
    expertise: str = Field(
        ..., 
        description="Detailed domain knowledge, patterns, best practices. "
                    "This gets injected into the agent's system prompt."
    )
    
    # What this skill enables
    capabilities: List[str] = Field(
        default_factory=list,
        description="List of capabilities this skill provides"
    )
    
    # Domain vocabulary
    glossary: Dict[str, str] = Field(
        default_factory=dict,
        description="Key terms and their definitions for this domain"
    )
    
    # Bundled tools that come with this skill
    # Using Any to avoid complex serialization of tool classes
    tools: List[Any] = Field(
        default_factory=list,
        description="Tools that are bundled with this skill"
    )
    
    # MCP servers this skill knows how to leverage
    mcp_servers: List[str] = Field(
        default_factory=list,
        description="MCP server names this skill can work with"
    )
    
    # Optional: priority for prompt ordering when multiple skills
    priority: int = Field(
        default=0,
        description="Higher priority skills appear first in system prompt"
    )
    
    class Config:
        arbitrary_types_allowed = True  # Allow tool classes
    
    def to_prompt_section(self) -> str:
        """
        Convert skill module to a system prompt section.
        
        Returns:
            Formatted string to inject into system prompt
        """
        sections = []
        
        # Header
        sections.append(f"## {self.name}")
        sections.append(f"*{self.description}*\n")
        
        # Main expertise
        sections.append(self.expertise.strip())
        
        # Capabilities
        if self.capabilities:
            sections.append("\n**Capabilities:**")
            for cap in self.capabilities:
                sections.append(f"- {cap}")
        
        # Glossary
        if self.glossary:
            sections.append("\n**Key Terms:**")
            for term, definition in self.glossary.items():
                sections.append(f"- **{term}**: {definition}")
        
        return "\n".join(sections)
    
    def get_tools(self) -> List["BaseTool"]:
        """
        Get instantiated tools from this skill.
        
        Returns:
            List of tool instances
        """
        instantiated = []
        for tool in self.tools:
            # If it's a class, instantiate it
            if isinstance(tool, type):
                instantiated.append(tool())
            else:
                instantiated.append(tool)
        return instantiated


def create_skill_module(
    name: str,
    description: str,
    expertise: str,
    capabilities: Optional[List[str]] = None,
    glossary: Optional[Dict[str, str]] = None,
    tools: Optional[List] = None,
    mcp_servers: Optional[List[str]] = None,
    priority: int = 0
) -> SkillModule:
    """
    Factory function to create a SkillModule.
    
    Args:
        name: Skill module name
        description: Brief description
        expertise: Detailed domain knowledge (main content)
        capabilities: List of what this skill enables
        glossary: Domain-specific terms
        tools: Bundled tools
        mcp_servers: MCP servers this skill leverages
        priority: Ordering priority (higher = first)
        
    Returns:
        Configured SkillModule instance
        
    Example:
        skill = create_skill_module(
            name="Python Expert",
            description="Advanced Python development expertise",
            expertise="You are an expert Python developer...",
            capabilities=["Write idiomatic Python", "Debug complex issues"],
            glossary={"GIL": "Global Interpreter Lock"}
        )
    """
    return SkillModule(
        name=name,
        description=description,
        expertise=expertise,
        capabilities=capabilities or [],
        glossary=glossary or {},
        tools=tools or [],
        mcp_servers=mcp_servers or [],
        priority=priority
    )
