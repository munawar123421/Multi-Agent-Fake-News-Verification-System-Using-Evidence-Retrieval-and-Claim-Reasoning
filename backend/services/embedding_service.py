"""Embedding service using sentence-transformers."""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import structlog

logger = structlog.get_logger()


class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding service.
        
        Args:
            model_name: Name of the sentence-transformer model
        """
        self.model_name = model_name
        logger.info("loading_embedding_model", model=model_name)
        self.model = SentenceTransformer(model_name)
        logger.info("embedding_model_loaded", model=model_name)
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embeddings
    
    def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        embeddings = self.embed([text1, text2])
        
        # Cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return float(similarity)
    
    def compute_similarities(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """Compute similarities between query and multiple documents.
        
        Args:
            query: Query text
            documents: List of document texts
            
        Returns:
            List of similarity scores
        """
        query_embedding = self.embed(query)
        doc_embeddings = self.embed(documents)
        
        # Compute cosine similarities
        similarities = []
        for doc_emb in doc_embeddings:
            similarity = np.dot(query_embedding[0], doc_emb) / (
                np.linalg.norm(query_embedding[0]) * np.linalg.norm(doc_emb)
            )
            similarities.append(float(similarity))
        
        return similarities
