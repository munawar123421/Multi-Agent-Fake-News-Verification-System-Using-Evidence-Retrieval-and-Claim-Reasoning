"""Claim Agent: Extract and normalize factual claims."""
from datetime import datetime
import structlog
from backend.graph.state import WorkflowState, ExecutionStep
from backend.services.llm_service import LLMService

logger = structlog.get_logger()


class ClaimAgent:
    """Agent responsible for extracting and normalizing claims."""
    
    def __init__(self, llm_service: LLMService):
        """Initialize claim agent.
        
        Args:
            llm_service: LLM service instance
        """
        self.llm_service = llm_service
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Process the claim and extract clean factual statement.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with clean_claim
        """
        start_time = datetime.now()
        logger.info("claim_agent_started", claim=state["claim"])
        
        # Add execution trace
        if "execution_trace" not in state:
            state["execution_trace"] = []
        
        state["execution_trace"].append(ExecutionStep(
            agent="ClaimAgent",
            status="started",
            timestamp=start_time
        ))
        
        try:
            system_prompt = """You are a claim extraction expert. Your task is to extract and normalize factual claims from user input.

Your responsibilities:
1. Remove noise, filler words, and unnecessary context
2. Extract the core factual claim
3. Normalize wording while preserving the original meaning
4. Ensure the claim is clear and verifiable
5. Keep the claim concise (1-2 sentences)

Return ONLY the normalized claim, nothing else."""

            prompt = f"""Extract and normalize the factual claim from the following text:

Text: {state["claim"]}

Normalized claim:"""

            clean_claim = await self.llm_service.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=200
            )
            
            state["clean_claim"] = clean_claim.strip()
            
            # Update execution trace
            duration = (datetime.now() - start_time).total_seconds() * 1000
            state["execution_trace"][-1] = ExecutionStep(
                agent="ClaimAgent",
                status="completed",
                timestamp=datetime.now(),
                duration_ms=duration,
                details={"clean_claim": state["clean_claim"]}
            )
            
            logger.info(
                "claim_agent_completed",
                clean_claim=state["clean_claim"],
                duration_ms=duration
            )
            
        except Exception as e:
            logger.error("claim_agent_failed", error=str(e))
            state["execution_trace"][-1] = ExecutionStep(
                agent="ClaimAgent",
                status="failed",
                timestamp=datetime.now(),
                details={"error": str(e)}
            )
            state["error"] = f"ClaimAgent failed: {str(e)}"
        
        return state
