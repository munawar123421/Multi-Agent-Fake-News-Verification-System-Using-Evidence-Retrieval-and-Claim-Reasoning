"""LangGraph workflow for multi-agent fake news verification."""
from typing import Dict, Any
from langgraph.graph import StateGraph, END
import structlog
from backend.graph.state import WorkflowState
from backend.agents.claim_agent import ClaimAgent
from backend.agents.query_agent import QueryAgent
from backend.agents.retrieval_agent import RetrievalAgent
from backend.agents.ranking_agent import RankingAgent
from backend.agents.reasoning_agent import ReasoningAgent
from backend.agents.explanation_agent import ExplanationAgent
from backend.services.llm_service import LLMService
from backend.services.retrieval_service import RetrievalService
from backend.services.ranking_service import RankingService

logger = structlog.get_logger()


class VerificationWorkflow:
    """Multi-agent workflow for fake news verification."""
    
    def __init__(self, llm_service: LLMService):
        """Initialize workflow with all agents.
        
        Args:
            llm_service: LLM service instance
        """
        self.llm_service = llm_service
        
        # Initialize services
        self.retrieval_service = RetrievalService()
        self.ranking_service = RankingService()
        
        # Initialize agents
        self.claim_agent = ClaimAgent(llm_service)
        self.query_agent = QueryAgent(llm_service)
        self.retrieval_agent = RetrievalAgent(self.retrieval_service)
        self.ranking_agent = RankingAgent(self.ranking_service)
        self.reasoning_agent = ReasoningAgent(llm_service)
        self.explanation_agent = ExplanationAgent(llm_service)
        
        # Build graph
        self.graph = self._build_graph()
        
        logger.info("verification_workflow_initialized")
    
    def _build_graph(self):
        """Build LangGraph workflow.
        
        Returns:
            Compiled StateGraph
        """
        # Create graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("claim_agent", self.claim_agent.process)
        workflow.add_node("query_agent", self.query_agent.process)
        workflow.add_node("retrieval_agent", self.retrieval_agent.process)
        workflow.add_node("ranking_agent", self.ranking_agent.process)
        workflow.add_node("reasoning_agent", self.reasoning_agent.process)
        workflow.add_node("explanation_agent", self.explanation_agent.process)
        
        # Set entry point
        workflow.set_entry_point("claim_agent")
        
        # Add edges (linear workflow)
        workflow.add_edge("claim_agent", "query_agent")
        workflow.add_edge("query_agent", "retrieval_agent")
        workflow.add_edge("retrieval_agent", "ranking_agent")
        workflow.add_edge("ranking_agent", "reasoning_agent")
        workflow.add_edge("reasoning_agent", "explanation_agent")
        
        # Set finish point
        workflow.add_edge("explanation_agent", END)
        
        return workflow.compile()
    
    async def verify_claim(self, claim: str) -> WorkflowState:
        """Execute verification workflow.
        
        Args:
            claim: User claim to verify
            
        Returns:
            Final workflow state with verification results
        """
        logger.info("workflow_started", claim=claim)
        
        # Initialize state
        initial_state: WorkflowState = {
            "claim": claim,
            "execution_trace": []
        }
        
        # Execute workflow
        result = await self.graph.ainvoke(initial_state)
        
        logger.info("workflow_completed", verdict=result.get("verdict"))
        
        return result
    
    def get_graph_visualization(self) -> str:
        """Get graph visualization in Mermaid format.
        
        Returns:
            Mermaid diagram string
        """
        mermaid = """graph TD
    A[ClaimAgent] --> B[QueryAgent]
    B --> C[RetrievalAgent]
    C --> D[RankingAgent]
    D --> E[ReasoningAgent]
    E --> F[ExplanationAgent]
    
    style A fill:#e1f5ff
    style B fill:#e1f5ff
    style C fill:#ffe1e1
    style D fill:#ffe1e1
    style E fill:#e1ffe1
    style F fill:#e1ffe1
"""
        return mermaid
