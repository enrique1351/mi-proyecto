"""
News Aggregator Module
======================

Handles news collection and processing:
- Web scraping from financial news sources
- News sentiment analysis
- Event detection for market-moving news
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import pandas as pd

logger = logging.getLogger(__name__)


class NewsSource(ABC):
    """Base class for news sources"""
    
    def __init__(self, name: str):
        self.name = name
        self.last_fetch_time: Optional[datetime] = None
        self.articles_fetched = 0
    
    @abstractmethod
    def fetch_news(self, keywords: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch news articles"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this news source"""
        return {
            "name": self.name,
            "last_fetch": self.last_fetch_time,
            "articles_fetched": self.articles_fetched
        }


class MockNewsSource(NewsSource):
    """Mock news source for testing"""
    
    def __init__(self):
        super().__init__("MockNews")
    
    def fetch_news(self, keywords: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Generate mock news articles"""
        articles = []
        
        for i in range(limit):
            article = {
                "title": f"Market Update: {keywords[0] if keywords else 'General'} - Article {i+1}",
                "content": f"Mock content about {keywords[0] if keywords else 'markets'}",
                "source": self.name,
                "timestamp": datetime.now() - timedelta(hours=i),
                "url": f"https://example.com/article-{i+1}",
                "sentiment": 0.0  # Neutral
            }
            articles.append(article)
        
        self.articles_fetched += len(articles)
        self.last_fetch_time = datetime.now()
        
        logger.info(f"Fetched {len(articles)} mock articles for keywords: {keywords}")
        return articles


class NewsAggregator:
    """
    Aggregates news from multiple sources
    
    Supports:
    - Multiple news sources
    - Keyword filtering
    - Deduplication
    - Time-based filtering
    """
    
    def __init__(self):
        self.sources: Dict[str, NewsSource] = {}
        self.articles_cache: List[Dict[str, Any]] = []
        logger.info("NewsAggregator initialized")
    
    def add_source(self, source: NewsSource):
        """Add a news source"""
        self.sources[source.name] = source
        logger.info(f"News source '{source.name}' added")
    
    def fetch_all_news(self, keywords: List[str], limit_per_source: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch news from all sources
        
        Args:
            keywords: Keywords to search for
            limit_per_source: Max articles per source
            
        Returns:
            List of news articles
        """
        all_articles = []
        
        for source_name, source in self.sources.items():
            try:
                articles = source.fetch_news(keywords, limit_per_source)
                all_articles.extend(articles)
                logger.info(f"Fetched {len(articles)} articles from {source_name}")
            except Exception as e:
                logger.error(f"Error fetching news from {source_name}: {e}")
        
        # Update cache
        self.articles_cache = all_articles
        
        return all_articles
    
    def filter_by_time(self, articles: List[Dict[str, Any]], 
                       hours_back: int = 24) -> List[Dict[str, Any]]:
        """Filter articles by time"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        filtered = [a for a in articles if a.get('timestamp', datetime.min) > cutoff_time]
        logger.info(f"Filtered to {len(filtered)} articles within last {hours_back} hours")
        return filtered
    
    def filter_by_sentiment(self, articles: List[Dict[str, Any]], 
                           min_sentiment: float = -1.0,
                           max_sentiment: float = 1.0) -> List[Dict[str, Any]]:
        """Filter articles by sentiment score"""
        filtered = [a for a in articles 
                   if min_sentiment <= a.get('sentiment', 0.0) <= max_sentiment]
        logger.info(f"Filtered to {len(filtered)} articles by sentiment")
        return filtered
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all sources"""
        stats = {
            "total_sources": len(self.sources),
            "cached_articles": len(self.articles_cache),
            "sources": {name: source.get_stats() for name, source in self.sources.items()}
        }
        return stats
    
    def to_dataframe(self, articles: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        """Convert articles to DataFrame"""
        if articles is None:
            articles = self.articles_cache
        
        if not articles:
            return pd.DataFrame()
        
        return pd.DataFrame(articles)


class SentimentAnalyzer:
    """
    Sentiment analysis for news articles
    
    TODO: Integrate with NLP models (BERT, etc.)
    """
    
    def __init__(self):
        self.model = None  # Placeholder for ML model
        logger.info("SentimentAnalyzer initialized (mock mode)")
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score between -1.0 (negative) and 1.0 (positive)
        """
        # TODO: Implement actual sentiment analysis with BERT or similar
        # For now, return neutral
        return 0.0
    
    def analyze_batch(self, texts: List[str]) -> List[float]:
        """Analyze sentiment for multiple texts"""
        return [self.analyze_sentiment(text) for text in texts]
    
    def add_sentiment_to_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add sentiment scores to articles"""
        for article in articles:
            content = article.get('content', '') or article.get('title', '')
            article['sentiment'] = self.analyze_sentiment(content)
        
        logger.info(f"Added sentiment scores to {len(articles)} articles")
        return articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    aggregator = NewsAggregator()
    aggregator.add_source(MockNewsSource())
    
    # Fetch news
    articles = aggregator.fetch_all_news(keywords=["Bitcoin", "crypto"], limit_per_source=5)
    
    # Analyze sentiment
    sentiment_analyzer = SentimentAnalyzer()
    articles = sentiment_analyzer.add_sentiment_to_articles(articles)
    
    # Convert to DataFrame
    df = aggregator.to_dataframe(articles)
    print(f"\nFetched {len(df)} articles")
    print(df[['title', 'timestamp', 'sentiment']].head())
    
    print("\nStatistics:", aggregator.get_statistics())
