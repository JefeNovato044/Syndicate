# mind/skills/examples.py
"""
Examples of skill module usage.

This file demonstrates how to create and use skill modules in the JunoVtAI system.
"""

from .skill_module import SkillModule, create_skill_module
from .elasticsearch_skill import create_elasticsearch_skill


# ==================== Example 1: Kubernetes Skill ====================

def create_kubernetes_skill() -> SkillModule:
    """
    Example: Kubernetes Expert Skill Module.
    
    Demonstrates domain expertise for Kubernetes cluster management.
    """
    return create_skill_module(
        name="Kubernetes Expert",
        description="Deep expertise in Kubernetes cluster management",
        expertise="""
        You are an expert in Kubernetes. You know:
        - Deployment strategies (rolling, blue-green, canary)
        - Debugging with kubectl logs, describe, events
        - Resource management with requests/limits
        - Pod lifecycle and troubleshooting
        - Service discovery and networking
        - ConfigMaps, Secrets, and Config Management
        - Horizontal Pod Autoscaling (HPA) and Vertical Pod Autoscaling (VPA)
        - Cluster administration and security
        """,
        capabilities=[
            "Diagnose pod issues",
            "Review manifests",
            "Optimize resource usage",
            "Configure autoscaling",
            "Manage secrets and config"
        ],
        glossary={
            "Pod": "The smallest deployable computing unit in Kubernetes",
            "Deployment": "Manages ReplicaSets and provides declarative updates",
            "Service": "Abstraction that defines a logical set of Pods",
            "Ingress": "API object to manage external access to services",
            "HPA": "Horizontal Pod Autoscaler scales pods based on metrics",
            "VPA": "Vertical Pod Autoscaler adjusts resource requests/limits",
            "DaemonSet": "Ensures all (or some) Nodes run a copy of a Pod",
            "StatefulSet": "Manages stateful applications with stable identities"
        },
        priority=5
    )


# ==================== Example 2: Python Expert Skill ====================

def create_python_skill() -> SkillModule:
    """
    Example: Python Expert Skill Module.
    
    Demonstrates domain expertise for Python development.
    """
    return create_skill_module(
        name="Python Expert",
        description="Advanced Python development expertise",
        expertise="""
        You are an expert Python developer. You know:
        - Python best practices and idioms
        - Async/await patterns and asyncio
        - Type hints and mypy
        - Testing with pytest and unittest
        - Performance optimization and profiling
        - Concurrency and multiprocessing
        - Virtual environments and package management
        - Python data structures and algorithms
        """,
        capabilities=[
            "Write idiomatic Python code",
            "Debug complex issues",
            "Optimize performance",
            "Design clean APIs",
            "Write comprehensive tests"
        ],
        glossary={
            "GIL": "Global Interpreter Lock - Python's threading limitation",
            "PEP": "Python Enhancement Proposal - Python's design document process",
            "Type Hint": "Annotation indicating variable or function types",
            "Decorator": "Function that modifies another function's behavior",
            "Generator": "Function that yields values one at a time",
            "Context Manager": "Object managing resource allocation and cleanup"
        },
        priority=3
    )


# ==================== Example 3: Elasticsearch Skill ====================

def create_elasticsearch_example() -> SkillModule:
    """
    Example: Elasticsearch Manager Skill Module.
    
    Demonstrates domain expertise for Elasticsearch cluster management.
    """
    return create_elasticsearch_skill().get_skill_module()


# ==================== Example 4: Git Expert Skill ====================

def create_git_skill() -> SkillModule:
    """
    Example: Git Expert Skill Module.
    
    Demonstrates domain expertise for Git version control.
    """
    return create_skill_module(
        name="Git Expert",
        description="Expert knowledge of Git version control",
        expertise="""
        You are a Git expert. You know:
        - Git workflow strategies (GitFlow, GitHub Flow, Trunk-Based)
        - Advanced branching and merging techniques
        - Rebase, cherry-pick, and interactive rebasing
        - Git hooks and pre-commit configuration
        - Resolving merge conflicts
        - Git bisect and blame for debugging
        - Submodules and sub-tree management
        - Git server administration (GitLab, GitHub, Gitea)
        """,
        capabilities=[
            "Set up Git workflows",
            "Resolve merge conflicts",
            "Debug with git bisect",
            "Configure hooks",
            "Manage remote repositories"
        ],
        glossary={
            "HEAD": "Reference to the current commit",
            "Branch": "A movable pointer to a commit",
            "Commit": "A snapshot of the repository's state",
            "Merge": "Combining changes from one branch into another",
            "Rebase": "Reapply commits on top of another base tip",
            "Stash": "Temporary storage for uncommitted changes",
            "Remote": "A version of your project hosted on the internet"
        },
        priority=4
    )


# ==================== Example 5: Docker Expert Skill ====================

def create_docker_skill() -> SkillModule:
    """
    Example: Docker Expert Skill Module.
    
    Demonstrates domain expertise for Docker containerization.
    """
    return create_skill_module(
        name="Docker Expert",
        description="Expert knowledge of Docker containerization",
        expertise="""
        You are a Docker expert. You know:
        - Dockerfile best practices and multi-stage builds
        - Docker Compose for multi-container applications
        - Container networking and volumes
        - Docker security and image scanning
        - Docker registry management
        - Container orchestration basics (Kubernetes integration)
        - Image optimization and layer caching
        - Docker secrets and config management
        """,
        capabilities=[
            "Write optimized Dockerfiles",
            "Configure multi-container apps",
            "Secure container images",
            "Optimize build performance",
            "Manage container registries"
        ],
        glossary={
            "Container": "Lightweight, standalone package of software",
            "Image": "Read-only template for creating containers",
            "Dockerfile": "Text file containing instructions to build an image",
            "Volume": "Persistent storage for containers",
            "Network": "Isolated communication between containers",
            "Registry": "Storage and distribution service for Docker images",
            "Multi-stage": "Build process using multiple images to reduce final size"
        },
        priority=4
    )


# ==================== Usage Examples ====================

def example_usage():
    """
    Example of how to use skill modules.
    """
    # Create a skill module
    k8s_skill = create_kubernetes_skill()
    
    # Register it with the registry
    from .registry import SkillRegistry
    SkillRegistry.register(k8s_skill)
    
    # Get the skill from registry
    retrieved = SkillRegistry.get("Kubernetes Expert")
    
    # Get all skills
    all_skills = SkillRegistry.get_all()
    
    # Convert to prompt section
    prompt_section = k8s_skill.to_prompt_section()
    
    # Get tools
    tools = k8s_skill.get_tools()
    
    print(f"Registered skills: {SkillRegistry.list_skills()}")
    print(f"Number of tools: {len(tools)}")


if __name__ == "__main__":
    # Run example
    example_usage()
