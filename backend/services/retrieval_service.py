"""Retrieval service for evidence gathering from multiple sources."""
import os
from typing import List, Dict, Any, Optional
import structlog
from dotenv import load_dotenv

load_dotenv()
logger = structlog.get_logger()


class RetrievalService:
    """Service for retrieving evidence from multiple sources."""
    
    def __init__(self):
        """Initialize retrieval service."""
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        if self.tavily_api_key:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
                logger.info("tavily_client_initialized")
            except Exception as e:
                logger.warning("tavily_init_failed", error=str(e))
                self.tavily_client = None
        else:
            logger.warning("tavily_api_key_not_found")
            self.tavily_client = None
    
    async def search_tavily(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search using Tavily API.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of search results
        """
        if not self.tavily_client:
            logger.warning("tavily_client_not_available")
            return []
        
        try:
            response = self.tavily_client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced"
            )
            
            results = []
            for result in response.get("results", []):
                results.append({
                    "text": result.get("content", ""),
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "source": "Tavily Search",
                    "snippet": result.get("content", "")[:300]
                })
            
            logger.info("tavily_search_completed", query=query, results=len(results))
            return results
            
        except Exception as e:
            logger.error("tavily_search_failed", error=str(e), query=query)
            return []
    
    async def search_wikipedia(
        self,
        query: str,
        max_results: int = 3
    ) -> List[Dict[str, Any]]:
        """Search Wikipedia.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of Wikipedia articles
        """
        try:
            import wikipedia
            wikipedia.set_lang("en")
            
            # Search for pages
            search_results = wikipedia.search(query, results=max_results)
            
            results = []
            for title in search_results[:max_results]:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    
                    # Get first few paragraphs
                    content = page.content
                    snippet = content[:500] if len(content) > 500 else content
                    
                    results.append({
                        "text": content,
                        "title": page.title,
                        "url": page.url,
                        "source": "Wikipedia",
                        "snippet": snippet
                    })
                    
                except wikipedia.exceptions.DisambiguationError:
                    logger.warning("wikipedia_disambiguation", title=title)
                    continue
                except wikipedia.exceptions.PageError:
                    logger.warning("wikipedia_page_not_found", title=title)
                    continue
                except Exception as e:
                    logger.error("wikipedia_page_error", title=title, error=str(e))
                    continue
            
            logger.info("wikipedia_search_completed", query=query, results=len(results))
            return results
            
        except Exception as e:
            logger.error("wikipedia_search_failed", error=str(e), query=query)
            return []
    
    async def retrieve_evidence(
        self,
        queries: List[str],
        max_results_per_query: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve evidence from all sources.
        
        Args:
            queries: List of search queries
            max_results_per_query: Maximum results per query per source
            
        Returns:
            Combined and deduplicated results
        """
        all_results = []
        seen_urls = set()
        
        for query in queries:
            # Search Tavily
            tavily_results = await self.search_tavily(query, max_results_per_query)
            for result in tavily_results:
                url = result.get("url", "")
                if url and url not in seen_urls:
                    all_results.append(result)
                    seen_urls.add(url)
            
            # Search Wikipedia
            wiki_results = await self.search_wikipedia(query, max_results_per_query)
            for result in wiki_results:
                url = result.get("url", "")
                if url and url not in seen_urls:
                    all_results.append(result)
                    seen_urls.add(url)
        
        logger.info(
            "evidence_retrieval_completed",
            total_queries=len(queries),
            total_results=len(all_results)
        )
        
        return all_results
