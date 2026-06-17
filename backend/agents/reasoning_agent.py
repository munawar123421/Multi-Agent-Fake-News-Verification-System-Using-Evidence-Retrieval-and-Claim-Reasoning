"""Reasoning Agent: Perform fact verification with evidence."""
from datetime import datetime
import re
import structlog
from backend.graph.state import WorkflowState, ExecutionStep
from backend.services.llm_service import LLMService

logger = structlog.get_logger()


class ReasoningAgent:
    """Agent responsible for fact verification reasoning."""
    
    def __init__(self, llm_service: LLMService):
        """Initialize reasoning agent.
        
        Args:
            llm_service: LLM service instance
        """
        self.llm_service = llm_service
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Perform fact verification with evidence-based reasoning.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with verdict, confidence, and reasoning
        """
        start_time = datetime.now()
        logger.info("reasoning_agent_started")
        
        state["execution_trace"].append(ExecutionStep(
            agent="ReasoningAgent",
            status="started",
            timestamp=start_time
        ))
        
        try:
            claim = state.get("clean_claim", "")
            evidence = state.get("ranked_evidence", [])
            
            if not evidence:
                logger.warning("no_evidence_for_reasoning")
                state["verdict"] = "NOT_ENOUGH_INFO"
                state["confidence"] = 50.0
                state["reasoning"] = "No evidence found to verify the claim."
                return state
            
            # Format evidence for prompt
            evidence_text = self._format_evidence(evidence)
            
            system_prompt = """You are a fact-checking expert. Your task is to verify claims based on provided evidence.

Your responsibilities:
1. Analyze the claim and all provided evidence carefully
2. Determine the relationship between the claim and evidence
3. Make an evidence-grounded decision
4. NEVER hallucinate facts - only use provided evidence
5. Be objective and unbiased

You must return your verdict in this EXACT format:

VERDICT: [SUPPORT or REFUTE or NOT_ENOUGH_INFO]
CONFIDENCE: [number between 0-100]
REASONING: [Your detailed reasoning here]

Verdict definitions:
- SUPPORT: Evidence strongly confirms the claim is true
- REFUTE: Evidence strongly contradicts the claim
- NOT_ENOUGH_INFO: Evidence is insufficient, unclear, or contradictory

Base confidence on:
- Quality and authority of sources
- Consistency across evidence
- Relevance of evidence to claim
- Presence of counter-evidence"""

            prompt = f"""Verify the following claim using the provided evidence:

CLAIM: {claim}

EVIDENCE:
{evidence_text}

Your verification (use exact format):"""

            response = await self.llm_service.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=800
            )
            
            # Parse response
            verdict, confidence, reasoning = self._parse_reasoning_response(response)
            
            state["verdict"] = verdict
            state["confidence"] = confidence
            state["reasoning"] = reasoning
            
            # Update execution trace
            duration = (datetime.now() - start_time).total_seconds() * 1000
            state["execution_trace"][-1] = ExecutionStep(
                agent="ReasoningAgent",
                status="completed",
                timestamp=datetime.now(),
                duration_ms=duration,
                details={
                    "verdict": verdict,
                    "confidence": confidence
                }
            )
            
            logger.info(
                "reasoning_agent_completed",
                verdict=verdict,
                confidence=confidence,
                duration_ms=duration
            )
            
        except Exception as e:
            logger.error("reasoning_agent_failed", error=str(e))
            state["execution_trace"][-1] = ExecutionStep(
                agent="ReasoningAgent",
                status="failed",
                timestamp=datetime.now(),
                details={"error": str(e)}
            )
            state["error"] = f"ReasoningAgent failed: {str(e)}"
        
        return state
    
    def _format_evidence(self, evidence: list) -> str:
        """Format evidence for prompt."""
        formatted = []
        for i, ev in enumerate(evidence, 1):
            ev_dict = ev.model_dump() if hasattr(ev, 'model_dump') else ev
            text = ev_dict.get("text", "") or ev_dict.get("snippet", "")
            source = ev_dict.get("source", "Unknown")
            title = ev_dict.get("title", "")
            score = ev_dict.get("relevance_score", 0)
            
            formatted.append(
                f"Evidence {i} (Relevance: {score:.2f}):\n"
                f"Source: {source} - {title}\n"
                f"Content: {text[:500]}...\n"
            )
        
        return "\n".join(formatted)
    
    def _parse_reasoning_response(self, response: str) -> tuple:
        """Parse reasoning response."""
        verdict = "NOT_ENOUGH_INFO"
        confidence = 50.0
        reasoning = response
        
        # Extract verdict
        verdict_match = re.search(r'VERDICT:\s*(SUPPORT|REFUTE|NOT_ENOUGH_INFO)', response, re.IGNORECASE)
        if verdict_match:
            verdict = verdict_match.group(1).upper().replace(" ", "_")
        
        # Extract confidence
        confidence_match = re.search(r'CONFIDENCE:\s*(\d+(?:\.\d+)?)', response)
        if confidence_match:
            confidence = float(confidence_match.group(1))
            confidence = max(0, min(100, confidence))  # Clamp to 0-100
        
        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        return verdict, confidence, reasoning
