"""Knowledge base skill for RAG-enabled agents.

This module provides the KnowledgeBaseSkill that bundles the RAG search
tool with behavioral instructions, allowing agents to effectively use
the knowledge base through natural language interactions.

Example:
    ```python
    from syndicate.vectorstores import MongoVectorStore
    from syndicate.skills import KnowledgeBaseSkill
    
    # Create vector store
    vector_store = MongoVectorStore(...)
    
    # Create skill
    kb_skill = KnowledgeBaseSkill(vector_store=vector_store)
    
    # Install on agent
    agent.install_skill(kb_skill)
    ```
"""

from typing import Any, Dict, List, Optional

from .skill_module import SkillModule
from ..tools.rag_tool import RAGSearchTool
from ..vectorstores.base import BaseVectorStore


class KnowledgeBaseSkill(SkillModule):
    """Skill that enables agents to search a knowledge base.
    
    This skill bundles the RAG search tool with comprehensive
    behavioral instructions, teaching the agent when and how to
    use the knowledge base effectively.
    
    The skill provides:
    - RAGSearchTool for executing searches
    - Expertise definition for system prompt
    - Capabilities list for agent self-awareness
    - Glossary of RAG-related terms
    - Usage guidelines and examples
    
    Example:
        ```python
        from syndicate.vectorstores import MongoVectorStore
        from syndicate.skills import KnowledgeBaseSkill
        from syndicate.agents import BaseAgent
        
        # Setup vector store
        vector_store = MongoVectorStore(
            connection_string="mongodb+srv://...",
            database="mydb",
            collection="vectors",
            embedding_model=embedding_model,
            vector_dimension=384
        )
        
        # Create and install skill
        kb_skill = KnowledgeBaseSkill(
            vector_store=vector_store,
            top_k=4,
            domain="company documentation"
        )
        
        agent.install_skill(kb_skill)
        
        # Agent can now answer questions using knowledge base
        response = await agent.invoke(
            "What is the company's vacation policy?",
            owner_id="user123",
            chat_id="chat456"
        )
        ```
    """
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        top_k: int = 4,
        use_hybrid: bool = True,
        domain: str = "knowledge base",
        additional_instructions: Optional[str] = None
    ):
        """
        Args:
            vector_store: Vector store instance for searching
            top_k: Default number of search results to retrieve
            use_hybrid: Whether to use hybrid search (vector + keyword)
            domain: Description of the knowledge base domain
                   (e.g., "company documentation", "technical manuals")
            additional_instructions: Optional additional instructions
                                   to append to the expertise
        """
        # Create the search tool
        search_tool = RAGSearchTool(
            vector_store=vector_store,
            top_k=top_k,
            use_hybrid=use_hybrid
        )
        
        # Build expertise description
        expertise = self._build_expertise(domain)
        
        # Add additional instructions if provided
        if additional_instructions:
            expertise += f"\n\n{additional_instructions}"
        
        # Initialize SkillModule
        super().__init__(
            name="knowledge_base",
            expertise=expertise,
            capabilities=self._build_capabilities(),
            glossary=self._build_glossary(),
            tools=[search_tool],
            mcp_servers=[],
            priority=10  # High priority - knowledge base is often important
        )
    
    def _build_expertise(self, domain: str) -> str:
        """Build the expertise description for the skill.
        
        Args:
            domain: Description of the knowledge base domain
        
        Returns:
            Formatted expertise string for system prompt
        """
        return f"""You have access to a {domain} that contains important information
and documentation. You can search this knowledge base to find relevant
information to answer user questions.

## Knowledge Base Access

You have a tool called `search_knowledge_base` that allows you to search
the knowledge base for relevant information.

### When to Use the Knowledge Base:

1. **Factual Questions**: When the user asks about specific facts, policies,
   procedures, or information that might be documented.

2. **Domain-Specific Queries**: When the question relates to the domain
   of the knowledge base ({domain}).

3. **Verification**: When you need to verify information or provide
   citations/sources for your answers.

4. **Detailed Information**: When the user needs detailed, accurate
   information rather than general knowledge.

### How to Use the Knowledge Base:

1. **Craft Specific Queries**: When searching, create specific, focused
   queries that capture the essence of what you're looking for. Include
   key terms and concepts from the user's question.

2. **Analyze Results**: Carefully read the search results to understand
   what information is available. Note the source and relevance score
   of each result.

3. **Synthesize Answers**: Combine information from multiple results
   if needed to provide a complete answer.

4. **Cite Sources**: When providing information from the knowledge base,
   mention the source when relevant (e.g., "According to the employee
   handbook...").

5. **Acknowledge Limitations**: If the search returns no relevant results,
   honestly tell the user that you couldn't find specific information
   on that topic in the knowledge base.

### Important Guidelines:

- **Don't Hallucinate**: Only provide information that you find in the
  search results. Don't make up facts or policies.

- **Be Transparent**: Make it clear when you're using information from
  the knowledge base vs. your general knowledge.

- **Prioritize Accuracy**: If you're unsure about something, search the
  knowledge base first before answering.

- **Respect Context**: Consider the user's context and provide information
  that's relevant to their situation.

- **Handle Ambiguity**: If a search query is too broad or ambiguous,
  try to refine it or ask the user for clarification.

### Example Interaction:

User: "What is the company's remote work policy?"

Your thought process:
1. This is a factual question about company policy
2. I should search the knowledge base for "remote work policy"
3. Analyze the search results for relevant information
4. Synthesize an answer based on what I find
5. Cite the source if available

Your action:
- Call search_knowledge_base with query="remote work policy"
- Read the results carefully
- Provide a comprehensive answer based on the findings"""
    
    def _build_capabilities(self) -> List[str]:
        """Build the list of capabilities provided by this skill.
        
        Returns:
            List of capability descriptions
        """
        return [
            "Search knowledge base for relevant information",
            "Retrieve documented facts and policies",
            "Find specific information from stored documents",
            "Provide sourced answers with citations",
            "Perform semantic and keyword searches",
            "Access domain-specific documentation"
        ]
    
    def _build_glossary(self) -> Dict[str, str]:
        """Build a glossary of RAG-related terms.
        
        Returns:
            Dictionary of term definitions
        """
        return {
            "knowledge base": "A collection of stored documents and information that can be searched",
            "semantic search": "Search that understands meaning and context, not just keywords",
            "hybrid search": "Combination of semantic and keyword search for better results",
            "relevance score": "A numerical indicator of how well a result matches the query",
            "document chunk": "A smaller piece of a larger document, optimized for search"
        }
    
    def get_search_tool(self) -> RAGSearchTool:
        """Get the RAG search tool instance.
        
        Returns:
            The RAGSearchTool instance used by this skill
        """
        return self.tools[0] if self.tools else None