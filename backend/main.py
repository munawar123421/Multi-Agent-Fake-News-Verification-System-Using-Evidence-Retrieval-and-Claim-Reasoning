"""FastAPI application for fake news verification system."""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import structlog
from dotenv import load_dotenv

from backend.graph.state import VerificationRequest, VerificationResponse, Evidence
from backend.graph.workflow import VerificationWorkflow
from backend.services.llm_service import LLMService

# Load environment variables
load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

# Global workflow instance
workflow: VerificationWorkflow = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global workflow
    
    # Startup
    logger.info("application_starting")
    llm_service = LLMService()
    workflow = VerificationWorkflow(llm_service)
    logger.info("application_ready")
    
    yield
    
    # Shutdown
    logger.info("application_shutting_down")


# Create FastAPI app
app = FastAPI(
    title="Multi-Agent Fake News Verification System",
    description="Production-quality fact-checking system using LangGraph",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Multi-Agent Fake News Verification System",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "fake-news-verification",
        "llm_provider": os.getenv("LLM_PROVIDER", "gemini")
    }


@app.post("/verify", response_model=VerificationResponse)
async def verify_claim(request: VerificationRequest):
    """Verify a claim using the multi-agent system.
    
    Args:
        request: Verification request with claim
        
    Returns:
        Verification results with verdict, evidence, and explanation
    """
    try:
        logger.info("verification_request_received", claim=request.claim)
        
        # Execute workflow
        result = await workflow.verify_claim(request.claim)
        
        # Check for errors
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Build response
        response = VerificationResponse(
            claim=result["claim"],
            clean_claim=result.get("clean_claim", result["claim"]),
            verdict=result.get("verdict", "NOT_ENOUGH_INFO"),
            confidence=result.get("confidence", 0.0),
            evidence=result.get("ranked_evidence", []),
            reasoning=result.get("reasoning", ""),
            explanation=result.get("explanation", ""),
            sources=result.get("sources", []),
            execution_trace=result.get("execution_trace", [])
        )
        
        logger.info("verification_completed", verdict=response.verdict)
        
        return response
        
    except Exception as e:
        logger.error("verification_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    return {
        "total_verifications": 0,
        "average_confidence": 0.0,
        "verdict_distribution": {
            "SUPPORT": 0,
            "REFUTE": 0,
            "NOT_ENOUGH_INFO": 0
        }
    }


@app.get("/graph")
async def get_graph():
    """Get workflow graph visualization."""
    return {
        "mermaid": workflow.get_graph_visualization()
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
