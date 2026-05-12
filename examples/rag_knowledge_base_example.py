"""Example: Using the new Agentic RAG architecture.

This example demonstrates how to use the new skill-based RAG architecture
in Syndicate, which replaces the old BaseRAGMemory approach.

Key Concepts:
- Ingestion Pipeline: Split documents and generate embeddings (external)
- Vector Store: Store and retrieve documents with semantic search
- RAG Search Tool: Bridge between vector store and agent tools
- Knowledge Base Skill: Bundle tool with behavioral instructions

Prerequisites:
    1. MongoDB Atlas cluster with vector search enabled
    2. Install dependencies:
       pip install sentence-transformers pymongo

    3. Set up MongoDB Atlas indexes:
       - Vector search index on 'embedding' field
       - Atlas Search index on 'text' field

See: https://www.mongodb.com/docs/atlas/atlas-vector-search/
"""

import asyncio
import os
from typing import List, Dict, Any

from syndicate.ingestion import (
    RecursiveCharacterTextSplitter,
    SentenceTransformerEmbedding
)
from syndicate.vectorstores import MongoVectorStore
from syndicate.skills import KnowledgeBaseSkill
from syndicate.tools import RAGSearchTool
from syndicate.agents import GenericAgent
from syndicate.clients import GeminiClient


async def setup_vector_store():
    """Set up the MongoDB vector store.
    
    Returns:
        MongoVectorStore instance ready for use
    """
    # Get connection string from environment or use default
    connection_string = os.getenv(
        "MONGODB_ATLAS_URI",
        "mongodb+srv://username:password@cluster.mongodb.net/"
    )
    
    # Create embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformerEmbedding(
        model_name="all-MiniLM-L6-v2"  # 384 dimensions, fast
    )
    
    # Create vector store
    vector_store = MongoVectorStore(
        connection_string=connection_string,
        database="syndicate_demo",
        collection="knowledge_base",
        embedding_model=embedding_model,
        dims=384,
        index_name="vector_index",
        search_index_name="text_index"
    )

    # Optional bootstrap: if API/permissions allow, create collection and indexes.
    # Otherwise set up Atlas indexes manually (developer responsibility).
    await vector_store.ensure_backend_ready(create_indexes=True)
    
    print("Vector store initialized successfully!")
    return vector_store


async def ingest_documents(vector_store: MongoVectorStore):
    """Ingest sample documents into the vector store.
    
    This demonstrates the ingestion pipeline:
    1. Load documents (external to framework)
    2. Split into chunks
    3. Add to vector store (embeddings auto-generated)
    
    Args:
        vector_store: Vector store instance
    """
    # Step 1: Define sample documents
    # In practice, these would come from files, databases, APIs, etc.
    documents = [
        {
            "text": """
            Syndicate Company Remote Work Policy
            
            At Syndicate, we believe in flexibility and trust. Our remote work
            policy allows employees to work from anywhere, as long as they
            maintain productivity and communication with their teams.
            
            Eligibility:
            - All full-time employees are eligible for remote work
            - Employees must have been with the company for at least 90 days
            - Performance must meet or exceed expectations
            
            Remote Work Arrangements:
            - Fully Remote: Work from home 100% of the time
            - Hybrid: Split time between home and office
            - Office-Based: Primarily work from office with occasional remote days
            
            Requirements:
            - Reliable internet connection (minimum 25 Mbps)
            - Dedicated workspace free from distractions
            - Available during core hours (10 AM - 3 PM local time)
            - Regular video check-ins with team
            """,
            "metadata": {
                "source": "employee_handbook.pdf",
                "section": "remote_work",
                "last_updated": "2024-01-15"
            }
        },
        {
            "text": """
            Syndicate Vacation and Time Off Policy
            
            We understand that rest and recharge are essential for productivity
            and well-being. Our time off policy is designed to support work-life
            balance.
            
            Vacation Days:
            - Years 1-2: 15 days per year
            - Years 3-5: 20 days per year
            - Years 6+: 25 days per year
            
            Personal Days:
            - All employees receive 3 personal days per year
            - Can be used for appointments, family matters, or mental health days
            
            Holidays:
            - 10 company-wide holidays per year
            - Floating holiday: 1 additional day for cultural/religious observances
            
            Request Process:
            - Submit request at least 2 weeks in advance
            - Manager approval required
            - Use the HR portal to track balance
            """,
            "metadata": {
                "source": "employee_handbook.pdf",
                "section": "vacation",
                "last_updated": "2024-01-15"
            }
        },
        {
            "text": """
            Syndicate Health Insurance Benefits
            
            We provide comprehensive health coverage for all full-time employees
            and their families.
            
            Coverage Options:
            - PPO Plan: Wide network, higher premium
            - HMO Plan: Restricted network, lower premium
            - HDHP + HSA: High deductible, tax-advantaged savings account
            
            Company Contribution:
            - Employee only: 100% of premium covered
            - Employee + spouse: 80% of premium covered
            - Employee + dependents: 90% of premium covered
            - Family: 85% of premium covered
            
            Additional Benefits:
            - Dental insurance: Included at no cost
            - Vision insurance: $200 annual allowance
            - Mental health: Unlimited therapy sessions
            - Wellness program: $50/month gym reimbursement
            """,
            "metadata": {
                "source": "benefits_guide.pdf",
                "section": "health_insurance",
                "last_updated": "2024-02-01"
            }
        },
        {
            "text": """
            Syndicate Code Review Guidelines
            
            Code reviews are essential for maintaining code quality and knowledge
            sharing. All code changes must go through review before merging.
            
            Reviewer Responsibilities:
            - Check for bugs and edge cases
            - Verify code follows style guidelines
            - Ensure tests are comprehensive
            - Suggest improvements for clarity and performance
            - Be respectful and constructive in feedback
            
            Review Timeline:
            - Standard PRs: Review within 24 hours
            - Urgent fixes: Review within 4 hours
            - Large refactors: Schedule dedicated review session
            
            Approval Requirements:
            - Minimum 2 approvals for production code
            - 1 approval for internal tools
            - Owner approval for critical systems
            
            Best Practices:
            - Keep PRs small (< 400 lines)
            - Write clear descriptions
            - Self-review before submitting
            - Respond to feedback promptly
            """,
            "metadata": {
                "source": "engineering_handbook.md",
                "section": "code_review",
                "last_updated": "2024-01-20"
            }
        }
    ]
    
    # Step 2: Split documents into chunks
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        metadata = doc["metadata"]
        
        for chunk in chunks:
            all_chunks.append({
                "text": chunk.strip(),
                "metadata": metadata.copy()
            })
    
    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    
    # Step 3: Add chunks to vector store
    # Embeddings are auto-generated by the vector store
    print("Adding chunks to vector store...")
    texts = [chunk["text"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]
    
    doc_ids = await vector_store.add_texts(
        texts=texts,
        metadatas=metadatas
    )
    
    print(f"Successfully added {len(doc_ids)} documents to vector store!")
    return doc_ids


async def create_agent_with_knowledge_base(vector_store: MongoVectorStore):
    """Create an agent with knowledge base access.
    
    Args:
        vector_store: Vector store instance
    
    Returns:
        Agent with KnowledgeBaseSkill installed
    """
    # Create LLM client
    api_key = os.getenv("GEMINI_API_KEY", "your-api-key-here")
    llm_client = GeminiClient(api_key=api_key, model_name="gemini-1.5-pro")
    
    # Create agent
    agent = GenericAgent(
        name="HR Assistant",
        llm_client=llm_client,
        system_prompt="""You are a helpful HR assistant at Syndicate. 
        You help employees find information about company policies, benefits,
        and procedures. Always be friendly, professional, and accurate.
        
        When answering questions about company policies, use the knowledge
        base to find the most up-to-date information. Cite your sources
        when possible."""
    )
    
    # Create and install knowledge base skill
    kb_skill = KnowledgeBaseSkill(
        vector_store=vector_store,
        top_k=4,
        use_hybrid=True,
        domain="company HR documentation",
        additional_instructions="""
        IMPORTANT: Always search the knowledge base for policy questions.
        Common topics include:
        - Remote work policy
        - Vacation and time off
        - Health benefits
        - Code review process
        
        When you find relevant information, summarize it clearly and mention
        the source (e.g., "According to the employee handbook...").
        """
    )
    
    agent.install_skill(kb_skill)
    
    print("Agent created with knowledge base skill!")
    return agent


async def demonstrate_search(vector_store: MongoVectorStore):
    """Demonstrate direct vector store search.
    
    Args:
        vector_store: Vector store instance
    """
    print("\n" + "="*60)
    print("DEMONSTRATING DIRECT VECTOR STORE SEARCH")
    print("="*60)
    
    # Create search tool
    search_tool = RAGSearchTool(
        vector_store=vector_store,
        top_k=3,
        use_hybrid=True
    )
    
    # Test queries
    queries = [
        "What is the remote work policy?",
        "How many vacation days do I get?",
        "What health insurance options are available?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)

        formatted = await search_tool.run_async(query=query, top_k=3)
        print(formatted)
        
        print()


async def demonstrate_agent_interaction(agent: GenericAgent):
    """Demonstrate agent interaction with knowledge base.
    
    Args:
        Agent with knowledge base skill
    """
    print("\n" + "="*60)
    print("DEMONSTRATING AGENT INTERACTION")
    print("="*60)
    
    questions = [
        "Can I work from home?",
        "How much vacation time do I have after 3 years?",
        "Does the company pay for health insurance?",
        "What's the process for code review?"
    ]
    
    for question in questions:
        print(f"\nUser: {question}")
        print("-" * 40)
        
        response = await agent.invoke(
            user_input=question,
            owner_id="demo_user",
            chat_id="demo_session"
        )
        
        print(f"Agent: {response}")
        print()


async def main():
    """Main demonstration function."""
    print("="*60)
    print("SYNDICATE AGENTIC RAG DEMONSTRATION")
    print("="*60)
    
    try:
        # Step 1: Set up vector store
        print("\n[1/4] Setting up vector store...")
        vector_store = await setup_vector_store()
        
        # Step 2: Ingest documents
        print("\n[2/4] Ingesting documents...")
        doc_ids = await ingest_documents(vector_store)
        
        # Step 3: Demonstrate direct search
        print("\n[3/4] Demonstrating search capabilities...")
        await demonstrate_search(vector_store)
        
        # Step 4: Create agent and demonstrate interaction
        print("\n[4/4] Creating agent with knowledge base...")
        agent = await create_agent_with_knowledge_base(vector_store)
        await demonstrate_agent_interaction(agent)
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        print("\nCleaning up...")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())


# ============================================================================
# MIGRATION GUIDE: From BaseRAGMemory to New Architecture
# ============================================================================
"""
OLD APPROACH (Deprecated):
--------------------------
Older versions used a framework-managed RAG memory abstraction where
retrieval happened behind the scenes.


NEW APPROACH (Recommended):
---------------------------
from syndicate.vectorstores import MongoVectorStore
from syndicate.skills import KnowledgeBaseSkill

# 1. Create vector store
vector_store = MongoVectorStore(
    connection_string="...",
    embedding_model=embedding_model,
    ...
)

# 2. Create skill (bundles tool + instructions)
kb_skill = KnowledgeBaseSkill(
    vector_store=vector_store,
    domain="company documentation"
)

# 3. Install on agent
agent.install_skill(kb_skill)  # LLM-controlled


KEY DIFFERENCES:
---------------
1. Control: LLM decides when to search (not framework)
2. Modularity: Vector store is independent of agent
3. Flexibility: Same vector store can serve multiple agents
4. Transparency: Tool calls are visible in conversation
5. No "prompt slop": Context only injected when needed
"""