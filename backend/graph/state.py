"""Shared state object for the multi-agent workflow."""
from typing import List, Dict, Any, Optional, TypedDict, Literal
from datetime import datetime
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """Evidence document with metadata."""
    text: str
    source: str
    title: str
    url: Optional[str] = None
    relevance_score: Optional[float] = None
    snippet: Optional[str] = None


class ExecutionStep(BaseModel):
    """Single step in agent execution trace."""
    agent: str
    status: Literal["started", "completed", "failed"]
    timestamp: datetime
    duration_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class WorkflowState(TypedDict, total=False):
    """Shared state for the multi-agent fake news verification workflow.
    
    All agents read from and write to this shared state object.
    """
    # Original input
    claim: str
    
    # Claim Agent output
    clean_claim: str
    
    # Query Agent output
    queries: List[str]
    
    # Retrieval Agent output
    retrieved_documents: List[Evidence]
    
    # Ranking Agent output
    ranked_evidence: List[Evidence]
    
    # Reasoning Agent output
    verdict: Literal["SUPPORT", "REFUTE", "NOT_ENOUGH_INFO"]
    confidence: float  # 0-100
    reasoning: str
    
    # Explanation Agent output
    explanation: str
    
    # Metadata
    sources: List[Dict[str, str]]
    execution_trace: List[ExecutionStep]
    
    # Error handling
    error: Optional[str]


class VerificationRequest(BaseModel):
    """API request model for claim verification."""
    claim: str = Field(..., min_length=5, max_length=1000, description="The claim to verify")


class VerificationResponse(BaseModel):
    """API response model for verification results."""
    claim: str
    clean_claim: str
    verdict: Literal["SUPPORT", "REFUTE", "NOT_ENOUGH_INFO"]
    confidence: float
    evidence: List[Evidence]
    reasoning: str
    explanation: str
    sources: List[Dict[str, str]]
    execution_trace: List[ExecutionStep]
    
    class Config:
        json_schema_extra = {
            "example": {
                "claim": "COVID vaccines contain microchips",
                "clean_claim": "COVID-19 vaccines contain microchips",
                "verdict": "REFUTE",
                "confidence": 94.5,
                "evidence": [
                    {
                        "text": "COVID-19 vaccines do not contain microchips...",
                        "source": "CDC",
                        "title": "Myths and Facts about COVID-19 Vaccines",
                        "relevance_score": 0.95
                    }
                ],
                "reasoning": "Multiple authoritative sources confirm...",
                "explanation": "This claim is false because...",
                "sources": [{"title": "CDC", "url": "https://..."}],
                "execution_trace": []
            }
        }
