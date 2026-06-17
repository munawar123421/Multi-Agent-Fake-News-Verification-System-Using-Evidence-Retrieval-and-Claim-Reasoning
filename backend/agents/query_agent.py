"""Query Agent: Generate search queries for evidence retrieval."""
from datetime import datetime
from typing import List
import structlog
from backend.graph.state import WorkflowState, ExecutionStep
from backend.services.llm_service import LLMService

logger = structlog.get_logger()


class QueryAgent:
    """Agent responsible for generating search queries."""
    
    def __init__(self, llm_service: LLMService):
        """Initialize query agent.
        
        Args:
            llm_service: LLM service instance
        """
        self.llm_service = llm_service
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Generate search queries for evidence retrieval.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with queries
        """
        start_time = datetime.now()
        logger.info("query_agent_started", claim=state.get("clean_claim"))
        
        state["execution_trace"].append(ExecutionStep(
            agent="QueryAgent",
            status="started",
            timestamp=start_time
        ))
        
        try:
            system_prompt = """You are a search query generation expert. Your task is to generate diverse search queries to find relevant evidence for fact-checking a claim.

Your responsibilities:
1. Generate 7-10 diverse search queries
2. Include the primary query (direct claim search)
3. Include supporting queries (related concepts, entities)
4. Include alternative wordings and perspectives
5. Include queries for potential counter-evidence
6. Keep queries concise and search-engine friendly

Return queries in a numbered list format, one per line."""

            prompt = f"""Generate 7-10 diverse search queries to find evidence for fact-checking this claim:

Claim: {state["clean_claim"]}

Search queries:"""

            response = await self.llm_service.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=500
            )
            
            # Parse queries from response
            queries = self._parse_queries(response)
            
            state["queries"] = queries
            
            # Update execution trace
            duration = (datetime.now() - start_time).total_seconds() * 1000
            state["execution_trace"][-1] = ExecutionStep(
                agent="QueryAgent",
                status="completed",
                timestamp=datetime.now(),
                duration_ms=duration,
                details={"num_queries": len(queries), "queries": queries}
            )
            
            logger.info(
                "query_agent_completed",
                num_queries=len(queries),
                duration_ms=duration
            )
            
        except Exception as e:
            logger.error("query_agent_failed", error=str(e))
            state["execution_trace"][-1] = ExecutionStep(
                agent="QueryAgent",
                status="failed",
                timestamp=datetime.now(),
                details={"error": str(e)}
            )
            state["error"] = f"QueryAgent failed: {str(e)}"
        
        return state
    
    def _parse_queries(self, response: str) -> List[str]:
        """Parse queries from LLM response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of parsed queries
        """
        lines = response.strip().split('\n')
        queries = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering (e.g., "1.", "1)", etc.)
            if line[0].isdigit():
                # Find first non-digit, non-punctuation character
                for i, char in enumerate(line):
                    if char.isalpha() or char == '"':
                        line = line[i:].strip()
                        break
            
            # Remove quotes
            line = line.strip('"\'')
            
            if line and len(line) > 3:
                queries.append(line)
        
        # Ensure we have at least the original claim as a query
        if not queries:
            queries = [state.get("clean_claim", state.get("claim", ""))]
        
        return queries[:10]  # Limit to 10 queries
