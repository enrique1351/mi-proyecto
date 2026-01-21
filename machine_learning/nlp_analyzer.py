"""
NLP Module for News and Sentiment Analysis
===========================================

Uses BERT and other NLP models for:
- News sentiment analysis
- Political event detection
- Market sentiment from text
"""

import logging
from typing import List, Dict, Optional
import pandas as pd


logger = logging.getLogger(__name__)


class NewsAnalyzer:
    """
    Analyzes news and text for market sentiment.
    
    Uses:
    - BERT for sentiment analysis
    - Named Entity Recognition
    - Topic modeling
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Initialize news analyzer.
        
        Args:
            model_name: Pretrained model name
        """
        self.model_name = model_name
        self.sentiment_pipeline = None
        self.ner_pipeline = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models."""
        try:
            from transformers import pipeline
            
            # Sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert"  # Financial BERT
            )
            logger.info("✅ Sentiment analysis model loaded")
            
            # NER pipeline for entity extraction
            self.ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER"
            )
            logger.info("✅ NER model loaded")
            
        except ImportError:
            logger.error("transformers package not installed. Install with: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with sentiment and confidence
        """
        try:
            # Truncate long text
            if len(text) > 512:
                text = text[:512]
            
            result = self.sentiment_pipeline(text)[0]
            
            return {
                'sentiment': result['label'].lower(),  # positive, negative, neutral
                'confidence': result['score'],
                'text': text
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze sentiment: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'text': text
            }
    
    def analyze_news_batch(self, news_list: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment for multiple news articles.
        
        Args:
            news_list: List of news texts
            
        Returns:
            DataFrame with sentiment analysis results
        """
        try:
            results = []
            
            for news in news_list:
                sentiment_data = self.analyze_sentiment(news)
                results.append(sentiment_data)
            
            df = pd.DataFrame(results)
            
            # Calculate aggregate sentiment score
            sentiment_map = {
                'positive': 1,
                'neutral': 0,
                'negative': -1
            }
            
            df['sentiment_score'] = df.apply(
                lambda row: sentiment_map[row['sentiment']] * row['confidence'],
                axis=1
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to analyze news batch: {e}")
            return pd.DataFrame()
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of entities with type and text
        """
        try:
            entities = self.ner_pipeline(text)
            
            # Group consecutive tokens
            grouped_entities = []
            current_entity = None
            
            for entity in entities:
                if entity['entity'].startswith('B-'):
                    if current_entity:
                        grouped_entities.append(current_entity)
                    current_entity = {
                        'type': entity['entity'][2:],
                        'text': entity['word'],
                        'score': entity['score']
                    }
                elif entity['entity'].startswith('I-') and current_entity:
                    current_entity['text'] += ' ' + entity['word']
                    current_entity['score'] = (current_entity['score'] + entity['score']) / 2
            
            if current_entity:
                grouped_entities.append(current_entity)
            
            return grouped_entities
            
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            return []
    
    def get_market_sentiment(
        self,
        news_list: List[str],
        entities_of_interest: Optional[List[str]] = None
    ) -> Dict:
        """
        Calculate overall market sentiment from news.
        
        Args:
            news_list: List of news articles
            entities_of_interest: Optional list of entities to focus on (e.g., ['Bitcoin', 'Fed'])
            
        Returns:
            Dict with sentiment metrics
        """
        try:
            # Analyze all news
            df = self.analyze_news_batch(news_list)
            
            if df.empty:
                return {
                    'overall_sentiment': 'neutral',
                    'sentiment_score': 0.0,
                    'confidence': 0.0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0
                }
            
            # Filter by entities if specified
            if entities_of_interest:
                filtered_indices = []
                for idx, text in enumerate(news_list):
                    entities = self.extract_entities(text)
                    entity_texts = [e['text'].lower() for e in entities]
                    
                    if any(entity.lower() in ' '.join(entity_texts) for entity in entities_of_interest):
                        filtered_indices.append(idx)
                
                if filtered_indices:
                    df = df.iloc[filtered_indices]
            
            # Calculate metrics
            sentiment_counts = df['sentiment'].value_counts().to_dict()
            avg_score = df['sentiment_score'].mean()
            avg_confidence = df['confidence'].mean()
            
            # Determine overall sentiment
            if avg_score > 0.2:
                overall = 'positive'
            elif avg_score < -0.2:
                overall = 'negative'
            else:
                overall = 'neutral'
            
            return {
                'overall_sentiment': overall,
                'sentiment_score': float(avg_score),
                'confidence': float(avg_confidence),
                'positive_count': sentiment_counts.get('positive', 0),
                'negative_count': sentiment_counts.get('negative', 0),
                'neutral_count': sentiment_counts.get('neutral', 0),
                'total_articles': len(df)
            }
            
        except Exception as e:
            logger.error(f"Failed to get market sentiment: {e}")
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_articles': 0
            }


class NewsAggregator:
    """
    Aggregates news from multiple sources.
    """
    
    def __init__(self, api_keys: Dict[str, str]):
        """
        Initialize news aggregator.
        
        Args:
            api_keys: Dict with API keys for news sources
        """
        self.api_keys = api_keys
    
    def fetch_crypto_news(self, limit: int = 10) -> List[Dict]:
        """
        Fetch cryptocurrency news.
        
        Args:
            limit: Number of articles to fetch
            
        Returns:
            List of news articles
        """
        try:
            import requests
            
            # Example: CryptoPanic API
            if 'cryptopanic' in self.api_keys:
                response = requests.get(
                    "https://cryptopanic.com/api/v1/posts/",
                    params={
                        'auth_token': self.api_keys['cryptopanic'],
                        'kind': 'news'
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    results = response.json().get('results', [])[:limit]
                    return [
                        {
                            'title': item['title'],
                            'url': item['url'],
                            'published_at': item['published_at'],
                            'source': item['source']['title']
                        }
                        for item in results
                    ]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to fetch crypto news: {e}")
            return []
    
    def fetch_stock_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Fetch stock-specific news.
        
        Args:
            symbol: Stock symbol
            limit: Number of articles
            
        Returns:
            List of news articles
        """
        try:
            import requests
            
            # Example: NewsAPI
            if 'newsapi' in self.api_keys:
                response = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        'apiKey': self.api_keys['newsapi'],
                        'q': symbol,
                        'language': 'en',
                        'sortBy': 'publishedAt',
                        'pageSize': limit
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    articles = response.json().get('articles', [])
                    return [
                        {
                            'title': article['title'],
                            'description': article['description'],
                            'url': article['url'],
                            'published_at': article['publishedAt'],
                            'source': article['source']['name']
                        }
                        for article in articles
                    ]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to fetch stock news: {e}")
            return []
