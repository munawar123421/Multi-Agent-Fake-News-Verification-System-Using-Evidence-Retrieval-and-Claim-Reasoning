"""Ranking Agent: Rank evidence by relevance."""
from datetime import datetime
import structlog
from backend.graph.state import WorkflowState, ExecutionStep, Evidence
from backend.services.ranking_service import RankingService

logger = structlog.get_logger()


class RankingAgent:
    """Agent responsible for ranking evidence."""
    
    def __init__(self, ranking_service: RankingService):
        """Initialize ranking agent.
        
        Args:
            ranking_service: Ranking service instance
        """
        self.ranking_service = ranking_service
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Rank evidence by relevance to claim.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with ranked evidence
        """
        start_time = datetime.now()
        logger.info("ranking_agent_started", documents=len(state.get("retrieved_documents", [])))
        
        state["execution_trace"].append(ExecutionStep(
            agent="RankingAgent",
            status="started",
            timestamp=start_time
        ))
        
        try:
            claim = state.get("clean_claim", "")
            documents = state.get("retrieved_documents", [])
            
            if not documents:
                logger.warning("no_documents_to_rank")
                state["ranked_evidence"] = []
                return state
            
            # Convert Evidence objects to dicts for ranking
            doc_dicts = [doc.model_dump() if hasattr(doc, 'model_dump') else doc for doc in documents]
            
            # Rank evidence
            ranked_results = self.ranking_service.rank_evidence(
                claim=claim,
                evidence_list=doc_dicts,
                top_k=5
            )
            
            # Convert back to Evidence objects
            ranked_evidence = []
            for result in ranked_results:
                evidence = Evidence(**result)
                ranked_evidence.append(evidence)
            
            state["ranked_evidence"] = ranked_evidence
            
            # Update execution trace
            duration = (datetime.now() - start_time).total_seconds() * 1000
            state["execution_trace"][-1] = ExecutionStep(
                agent="RankingAgent",
                status="completed",
                timestamp=datetime.now(),
                duration_ms=duration,
                details={
                    "num_ranked": len(ranked_evidence),
                    "top_score": ranked_evidence[0].relevance_score if ranked_evidence else 0
                }
            )
            
            logger.info(
                "ranking_agent_completed",
                num_ranked=len(ranked_evidence),
                duration_ms=duration
            )
            
        except Exception as e:
            logger.error("ranking_agent_failed", error=str(e))
            state["execution_trace"][-1] = ExecutionStep(
                agent="RankingAgent",
                status="failed",
                timestamp=datetime.now(),
                details={"error": str(e)}
            )
            state["error"] = f"RankingAgent failed: {str(e)}"
        
        return state
