"""Services package."""
# Import only when needed to avoid circular imports and loading issues

__all__ = [
    "LLMService",
    "EmbeddingService",
    "RetrievalService",
    "RankingService",
]

def __getattr__(name):
    """Lazy load services to avoid import issues."""
    if name == "LLMService":
        from backend.services.llm_service import LLMService
        return LLMService
    elif name == "EmbeddingService":
        from backend.services.embedding_service import EmbeddingService
        return EmbeddingService
    elif name == "RetrievalService":
        from backend.services.retrieval_service import RetrievalService
        return RetrievalService
    elif name == "RankingService":
        from backend.services.ranking_service import RankingService
        return RankingService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
