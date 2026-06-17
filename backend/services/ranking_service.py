"""Ranking service for evidence relevance scoring."""
from typing import List, Dict, Any
import numpy as np
import structlog
from backend.services.embedding_service import EmbeddingService

logger = structlog.get_logger()


class RankingService:
    """Service for ranking evidence by relevance."""
    
    def __init__(self):
        """Initialize ranking service."""
        self.embedding_service = EmbeddingService()
    
    def rank_evidence(
        self,
        claim: str,
        evidence_list: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Rank evidence by relevance to claim.
        
        Args:
            claim: The claim to verify
            evidence_list: List of evidence documents
            top_k: Number of top results to return
            
        Returns:
            Top-k ranked evidence with relevance scores
        """
        if not evidence_list:
            logger.warning("no_evidence_to_rank")
            return []
        
        # Extract texts for embedding
        texts = [ev.get("text", "") or ev.get("snippet", "") for ev in evidence_list]
        
        # Compute similarities
        similarities = self.embedding_service.compute_similarities(claim, texts)
        
        # Add scores to evidence
        ranked_evidence = []
        for evidence, score in zip(evidence_list, similarities):
            evidence_with_score = evidence.copy()
            evidence_with_score["relevance_score"] = float(score)
            ranked_evidence.append(evidence_with_score)
        
        # Sort by relevance score
        ranked_evidence.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Return top-k
        top_evidence = ranked_evidence[:top_k]
        
        logger.info(
            "evidence_ranked",
            total_evidence=len(evidence_list),
            top_k=top_k,
            top_score=top_evidence[0]["relevance_score"] if top_evidence else 0
        )
        
        return top_evidence
