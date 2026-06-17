"""Retrieval Agent: Retrieve candidate evidence from multiple sources."""
from datetime import datetime
from typing import List
import structlog
from backend.graph.state import WorkflowState, ExecutionStep, Evidence
from backend.services.retrieval_service import RetrievalService

logger = structlog.get_logger()


class RetrievalAgent:
    """Agent responsible for retrieving evidence."""
    
    def __init__(self, retrieval_service: RetrievalService):
        """Initialize retrieval agent.
        
        Args:
            retrieval_service: Retrieval service instance
        """
        self.retrieval_service = retrieval_service
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Retrieve evidence from multiple sources.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with retrieved documents
        """
        start_time = datetime.now()
        logger.info("retrieval_agent_started", queries=len(state.get("queries", [])))
        
        state["execution_trace"].append(ExecutionStep(
            agent="RetrievalAgent",
            status="started",
            timestamp=start_time
        ))
        
        try:
            queries = state.get("queries", [])
            
            if not queries:
                logger.warning("no_queries_to_retrieve")
                state["retrieved_documents"] = []
                return state
            
            # Retrieve evidence from all sources
            results = await self.retrieval_service.retrieve_evidence(
                queries=queries,
                max_results_per_query=3
            )
            
            # Convert to Evidence objects
            evidence_list = []
            for result in results:
                evidence = Evidence(
                    text=result.get("text", ""),
                    source=result.get("source", ""),
                    title=result.get("title", ""),
                    url=result.get("url"),
                    snippet=result.get("snippet")
                )
                evidence_list.append(evidence)
            
            state["retrieved_documents"] = evidence_list
            
            # Update execution trace
            duration = (datetime.now() - start_time).total_seconds() * 1000
            state["execution_trace"][-1] = ExecutionStep(
                agent="RetrievalAgent",
                status="completed",
                timestamp=datetime.now(),
                duration_ms=duration,
                details={"num_documents": len(evidence_list)}
            )
            
            logger.info(
                "retrieval_agent_completed",
                num_documents=len(evidence_list),
                duration_ms=duration
            )
            
        except Exception as e:
            logger.error("retrieval_agent_failed", error=str(e))
            state["execution_trace"][-1] = ExecutionStep(
                agent="RetrievalAgent",
                status="failed",
                timestamp=datetime.now(),
                details={"error": str(e)}
            )
            state["error"] = f"RetrievalAgent failed: {str(e)}"
        
        return state
