"""Explanation Agent: Generate human-readable explanations."""
from datetime import datetime
import structlog
from backend.graph.state import WorkflowState, ExecutionStep
from backend.services.llm_service import LLMService

logger = structlog.get_logger()


class ExplanationAgent:
    """Agent responsible for generating explanations."""
    
    def __init__(self, llm_service: LLMService):
        """Initialize explanation agent.
        
        Args:
            llm_service: LLM service instance
        """
        self.llm_service = llm_service
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Generate human-readable explanation.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with explanation and sources
        """
        start_time = datetime.now()
        logger.info("explanation_agent_started")
        
        state["execution_trace"].append(ExecutionStep(
            agent="ExplanationAgent",
            status="started",
            timestamp=start_time
        ))
        
        try:
            claim = state.get("clean_claim", "")
            verdict = state.get("verdict", "NOT_ENOUGH_INFO")
            confidence = state.get("confidence", 50.0)
            reasoning = state.get("reasoning", "")
            evidence = state.get("ranked_evidence", [])
            
            # Extract sources
            sources = []
            for ev in evidence[:5]:
                ev_dict = ev.model_dump() if hasattr(ev, 'model_dump') else ev
                if ev_dict.get("url"):
                    sources.append({
                        "title": ev_dict.get("title", ""),
                        "url": ev_dict.get("url", ""),
                        "source": ev_dict.get("source", "")
                    })
            
            state["sources"] = sources
            
            # Format evidence references
            evidence_summary = self._format_evidence_summary(evidence)
            
            system_prompt = """You are an expert at explaining fact-checking results to general audiences.

Your responsibilities:
1. Explain the verdict in clear, simple language
2. Reference the key evidence that led to the conclusion
3. Include the confidence score and what it means
4. Be objective and educational
5. Keep the explanation concise (3-5 sentences)
6. Use accessible language (avoid jargon)

Do NOT include any prefixes like "Explanation:" or formatting markers. Just provide the explanation text."""

            prompt = f"""Generate a clear explanation for this fact-check result:

CLAIM: {claim}

VERDICT: {verdict}
CONFIDENCE: {confidence:.1f}%

REASONING: {reasoning}

KEY EVIDENCE:
{evidence_summary}

Explanation:"""

            explanation = await self.llm_service.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.5,
                max_tokens=400
            )
            
            state["explanation"] = explanation.strip()
            
            # Update execution trace
            duration = (datetime.now() - start_time).total_seconds() * 1000
            state["execution_trace"][-1] = ExecutionStep(
                agent="ExplanationAgent",
                status="completed",
                timestamp=datetime.now(),
                duration_ms=duration,
                details={"num_sources": len(sources)}
            )
            
            logger.info(
                "explanation_agent_completed",
                num_sources=len(sources),
                duration_ms=duration
            )
            
        except Exception as e:
            logger.error("explanation_agent_failed", error=str(e))
            state["execution_trace"][-1] = ExecutionStep(
                agent="ExplanationAgent",
                status="failed",
                timestamp=datetime.now(),
                details={"error": str(e)}
            )
            state["error"] = f"ExplanationAgent failed: {str(e)}"
        
        return state
    
    def _format_evidence_summary(self, evidence: list) -> str:
        """Format evidence summary for prompt."""
        formatted = []
        for i, ev in enumerate(evidence[:3], 1):
            ev_dict = ev.model_dump() if hasattr(ev, 'model_dump') else ev
            source = ev_dict.get("source", "Unknown")
            title = ev_dict.get("title", "")
            snippet = ev_dict.get("snippet", "") or ev_dict.get("text", "")[:200]
            
            formatted.append(f"{i}. {source} - {title}: {snippet}...")
        
        return "\n".join(formatted)
