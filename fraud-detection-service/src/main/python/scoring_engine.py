"""
Real-time scoring engine with caching and batch processing for fraud detection
"""

import os
import logging
import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.pyfunc
from sklearn.preprocessing import StandardScaler

# Async libraries
import aioredis
import asyncpg
from motor.motor_asyncio import AsyncIOMotorClient
import httpx

# Monitoring and observability
from prometheus_client import Counter, Histogram, Gauge, Summary
import structlog

# Custom imports
from feature_pipeline import FeaturePipeline, FeatureConfig

# Configure logging
logger = structlog.get_logger()

# Prometheus metrics
SCORING_REQUESTS_TOTAL = Counter('fraud_scoring_requests_total', 'Total scoring requests', ['model_version', 'request_type'])
SCORING_LATENCY = Histogram('fraud_scoring_duration_seconds', 'Scoring latency', ['model_version', 'request_type'])
CACHE_OPERATIONS = Counter('scoring_cache_operations_total', 'Cache operations', ['operation', 'result'])
BATCH_PROCESSING_QUEUE_SIZE = Gauge('batch_processing_queue_size', 'Batch processing queue size')
MODEL_PREDICTIONS_SUMMARY = Summary('model_predictions', 'Model prediction scores')
SCORING_ERRORS = Counter('fraud_scoring_errors_total', 'Scoring errors', ['error_type'])


class ScoringMode(Enum):
    """Scoring mode enumeration"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"


class CacheStrategy(Enum):
    """Cache strategy enumeration"""
    NONE = "none"
    AGGRESSIVE = "aggressive"
    SELECTIVE = "selective"
    ADAPTIVE = "adaptive"


@dataclass
class ScoringRequest:
    """Scoring request data structure"""
    request_id: str
    user_id: int
    transaction_data: Dict[str, Any]
    context: Dict[str, Any]
    mode: ScoringMode
    priority: int = 1  # 1=low, 2=medium, 3=high
    created_at: datetime = None
    timeout_seconds: float = 5.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ScoringResult:
    """Scoring result data structure"""
    request_id: str
    fraud_probability: float
    fraud_prediction: bool
    risk_level: str
    confidence_score: float
    model_version: str
    features_used: int
    processing_time_ms: float
    cache_hit: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ModelCache:
    """Advanced model result caching with multiple strategies"""
    
    def __init__(self, redis_client: aioredis.Redis, strategy: CacheStrategy = CacheStrategy.SELECTIVE):
        self.redis_client = redis_client
        self.strategy = strategy
        self.hit_ratio_threshold = 0.7
        self.default_ttl = 300  # 5 minutes
        
    async def get_cached_result(self, cache_key: str) -> Optional[ScoringResult]:
        """Get cached scoring result"""
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                result_dict = json.loads(cached_data)
                result = ScoringResult(**result_dict)
                result.cache_hit = True
                CACHE_OPERATIONS.labels(operation='get', result='hit').inc()
                return result
            else:
                CACHE_OPERATIONS.labels(operation='get', result='miss').inc()
                return None
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            CACHE_OPERATIONS.labels(operation='get', result='error').inc()
            return None
    
    async def set_cached_result(self, cache_key: str, result: ScoringResult, ttl: Optional[int] = None):
        """Cache scoring result"""
        try:
            if not self._should_cache(result):
                return
            
            ttl = ttl or self._calculate_ttl(result)
            result_dict = asdict(result)
            result_dict['cache_hit'] = False  # Reset for storage
            
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(result_dict, default=str)
            )
            CACHE_OPERATIONS.labels(operation='set', result='success').inc()
            
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
            CACHE_OPERATIONS.labels(operation='set', result='error').inc()
    
    def _should_cache(self, result: ScoringResult) -> bool:
        """Determine if result should be cached based on strategy"""
        if self.strategy == CacheStrategy.NONE:
            return False
        elif self.strategy == CacheStrategy.AGGRESSIVE:
            return True
        elif self.strategy == CacheStrategy.SELECTIVE:
            # Cache high-confidence predictions
            return result.confidence_score > 0.8
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Cache based on hit ratio and confidence
            return result.confidence_score > 0.7
        
        return False
    
    def _calculate_ttl(self, result: ScoringResult) -> int:
        """Calculate TTL based on result characteristics"""
        base_ttl = self.default_ttl
        
        # Adjust based on confidence
        if result.confidence_score > 0.9:
            return base_ttl * 2  # Cache longer for high confidence
        elif result.confidence_score < 0.6:
            return base_ttl // 2  # Cache shorter for low confidence
        
        # Adjust based on risk level
        if result.risk_level == 'high':
            return base_ttl // 4  # Cache very briefly for high risk
        elif result.risk_level == 'low':
            return base_ttl * 3  # Cache longer for low risk
        
        return base_ttl
    
    async def invalidate_user_cache(self, user_id: int):
        """Invalidate all cache entries for a user"""
        try:
            pattern = f"score:*:user:{user_id}:*"
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
                CACHE_OPERATIONS.labels(operation='invalidate', result='success').inc()
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
            CACHE_OPERATIONS.labels(operation='invalidate', result='error').inc()
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        try:
            info = await self.redis_client.info('memory')
            return {
                'memory_usage_mb': info.get('used_memory', 0) / 1024 / 1024,
                'total_keys': await self.redis_client.dbsize(),
                'strategy': self.strategy.value,
                'default_ttl': self.default_ttl
            }
        except Exception as e:
            logger.warning(f"Failed to get cache statistics: {e}")
            return {}


class BatchProcessor:
    """Batch processing engine for high-throughput scoring"""
    
    def __init__(self, 
                 scoring_engine: 'ScoringEngine',
                 batch_size: int = 100,
                 max_wait_time: float = 1.0,
                 max_workers: int = 4):
        
        self.scoring_engine = scoring_engine
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_workers = max_workers
        
        self.request_queue = asyncio.Queue()
        self.result_handlers = {}
        self.processing = False
        self.batch_tasks = []
        
    async def start_processing(self):
        """Start batch processing"""
        if self.processing:
            return
        
        self.processing = True
        
        # Start batch collection and processing tasks
        for i in range(self.max_workers):
            task = asyncio.create_task(self._batch_processor_loop())
            self.batch_tasks.append(task)
        
        logger.info(f"Batch processor started with {self.max_workers} workers")
    
    async def stop_processing(self):
        """Stop batch processing"""
        self.processing = False
        
        # Cancel all tasks
        for task in self.batch_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.batch_tasks, return_exceptions=True)
        
        logger.info("Batch processor stopped")
    
    async def submit_request(self, request: ScoringRequest) -> ScoringResult:
        """Submit request for batch processing"""
        if not self.processing:
            raise RuntimeError("Batch processor not started")
        
        # Create future for result
        future = asyncio.Future()
        self.result_handlers[request.request_id] = future
        
        # Add to queue
        await self.request_queue.put(request)
        BATCH_PROCESSING_QUEUE_SIZE.set(self.request_queue.qsize())
        
        # Wait for result with timeout
        try:
            result = await asyncio.wait_for(future, timeout=request.timeout_seconds)
            return result
        except asyncio.TimeoutError:
            # Cleanup
            self.result_handlers.pop(request.request_id, None)
            raise
    
    async def _batch_processor_loop(self):
        """Main batch processing loop"""
        while self.processing:
            try:
                batch_requests = []
                start_time = time.time()
                
                # Collect requests for batch
                while (len(batch_requests) < self.batch_size and 
                       (time.time() - start_time) < self.max_wait_time):
                    try:
                        request = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=0.1
                        )
                        batch_requests.append(request)
                    except asyncio.TimeoutError:
                        continue
                
                if not batch_requests:
                    continue
                
                BATCH_PROCESSING_QUEUE_SIZE.set(self.request_queue.qsize())
                
                # Process batch
                await self._process_batch(batch_requests)
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                SCORING_ERRORS.labels(error_type='batch_processing').inc()
    
    async def _process_batch(self, requests: List[ScoringRequest]):
        """Process a batch of requests"""
        try:
            logger.info(f"Processing batch of {len(requests)} requests")
            
            # Prepare batch data
            contexts = []
            for request in requests:
                context = {
                    'user_id': request.user_id,
                    'transaction': request.transaction_data,
                    **request.context
                }
                contexts.append(context)
            
            # Extract features for batch
            features_list = []
            for context in contexts:
                features_df = await self.scoring_engine.feature_pipeline.extract_features(context)
                features_list.append(features_df.iloc[0].to_dict())
            
            batch_features_df = pd.DataFrame(features_list)
            
            # Preprocess features
            batch_features_df = await self.scoring_engine.feature_pipeline.preprocess_features(batch_features_df)
            
            # Make batch prediction
            predictions = await self.scoring_engine._predict_batch(batch_features_df)
            
            # Create results
            for i, request in enumerate(requests):
                result = ScoringResult(
                    request_id=request.request_id,
                    fraud_probability=predictions['probabilities'][i],
                    fraud_prediction=predictions['predictions'][i],
                    risk_level=predictions['risk_levels'][i],
                    confidence_score=predictions['confidence_scores'][i],
                    model_version=self.scoring_engine.model_version,
                    features_used=len(batch_features_df.columns),
                    processing_time_ms=predictions['processing_time_ms'] / len(requests)
                )
                
                # Return result to handler
                future = self.result_handlers.pop(request.request_id, None)
                if future and not future.done():
                    future.set_result(result)
                
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            
            # Set error for all requests
            for request in requests:
                future = self.result_handlers.pop(request.request_id, None)
                if future and not future.done():
                    error_result = ScoringResult(
                        request_id=request.request_id,
                        fraud_probability=0.5,
                        fraud_prediction=False,
                        risk_level='unknown',
                        confidence_score=0.0,
                        model_version='unknown',
                        features_used=0,
                        processing_time_ms=0.0,
                        error=str(e)
                    )
                    future.set_result(error_result)


class ScoringEngine:
    """Main scoring engine with real-time and batch capabilities"""
    
    def __init__(self, 
                 model_path: str = None,
                 redis_client: aioredis.Redis = None,
                 db_client: AsyncIOMotorClient = None,
                 feature_config: FeatureConfig = None):
        
        # Core components
        self.model = None
        self.model_version = "unknown"
        self.model_loaded = False
        
        # Feature pipeline
        self.feature_config = feature_config or FeatureConfig()
        self.feature_pipeline = None
        
        # Caching
        self.redis_client = redis_client
        self.model_cache = ModelCache(redis_client) if redis_client else None
        
        # Database
        self.db_client = db_client
        
        # Batch processing
        self.batch_processor = BatchProcessor(self)
        
        # Performance tracking
        self.performance_metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'avg_latency_ms': 0.0,
            'error_rate': 0.0
        }
        
        # Threading for CPU-intensive operations
        self.thread_executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self, model_path: str = None):
        """Initialize scoring engine"""
        try:
            # Load model
            await self.load_model(model_path)
            
            # Initialize feature pipeline
            if self.redis_client and self.db_client:
                self.feature_pipeline = FeaturePipeline(
                    config=self.feature_config,
                    redis_client=self.redis_client,
                    db_client=self.db_client
                )
            
            # Start batch processor
            await self.batch_processor.start_processing()
            
            logger.info("Scoring engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize scoring engine: {e}")
            raise
    
    async def load_model(self, model_path: str = None):
        """Load ML model"""
        try:
            if model_path:
                # Load from file
                self.model = joblib.load(model_path)
                self.model_version = "file_model"
            else:
                # Load from MLflow
                model_uri = os.getenv('MODEL_URI', 'models:/fraud_detection_model/latest')
                self.model = mlflow.pyfunc.load_model(model_uri)
                self.model_version = model_uri.split('/')[-1]
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully: {self.model_version}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    async def score_single(self, request: ScoringRequest) -> ScoringResult:
        """Score single transaction in real-time"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if self.model_cache:
                cached_result = await self.model_cache.get_cached_result(cache_key)
                if cached_result:
                    SCORING_REQUESTS_TOTAL.labels(
                        model_version=self.model_version,
                        request_type='real_time_cached'
                    ).inc()
                    return cached_result
            
            # Extract features
            context = {
                'user_id': request.user_id,
                'transaction': request.transaction_data,
                **request.context
            }
            
            features_df = await self.feature_pipeline.extract_features(context)
            features_df = await self.feature_pipeline.preprocess_features(features_df)
            
            # Make prediction
            prediction = await self._predict_single(features_df)
            
            # Create result
            result = ScoringResult(
                request_id=request.request_id,
                fraud_probability=prediction['probability'],
                fraud_prediction=prediction['prediction'],
                risk_level=prediction['risk_level'],
                confidence_score=prediction['confidence'],
                model_version=self.model_version,
                features_used=len(features_df.columns),
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
            # Cache result
            if self.model_cache:
                await self.model_cache.set_cached_result(cache_key, result)
            
            # Update context for future features
            await self.feature_pipeline.update_user_context(context)
            
            # Record metrics
            SCORING_REQUESTS_TOTAL.labels(
                model_version=self.model_version,
                request_type='real_time'
            ).inc()
            
            SCORING_LATENCY.labels(
                model_version=self.model_version,
                request_type='real_time'
            ).observe(time.time() - start_time)
            
            MODEL_PREDICTIONS_SUMMARY.observe(result.fraud_probability)
            
            return result
            
        except Exception as e:
            logger.error(f"Scoring failed for request {request.request_id}: {e}")
            SCORING_ERRORS.labels(error_type='real_time_scoring').inc()
            
            return ScoringResult(
                request_id=request.request_id,
                fraud_probability=0.5,
                fraud_prediction=False,
                risk_level='unknown',
                confidence_score=0.0,
                model_version=self.model_version,
                features_used=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def score_batch(self, requests: List[ScoringRequest]) -> List[ScoringResult]:
        """Score batch of transactions"""
        if not requests:
            return []
        
        # Submit all requests to batch processor
        tasks = []
        for request in requests:
            task = asyncio.create_task(self.batch_processor.submit_request(request))
            tasks.append(task)
        
        # Wait for all results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = ScoringResult(
                    request_id=requests[i].request_id,
                    fraud_probability=0.5,
                    fraud_prediction=False,
                    risk_level='unknown',
                    confidence_score=0.0,
                    model_version=self.model_version,
                    features_used=0,
                    processing_time_ms=0.0,
                    error=str(result)
                )
                final_results.append(error_result)
            else:
                final_results.append(result)
        
        return final_results
    
    async def _predict_single(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Make single prediction"""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Run prediction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            prediction = await loop.run_in_executor(
                self.thread_executor,
                self._sync_predict,
                features_df
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    async def _predict_batch(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Make batch predictions"""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Run batch prediction in thread pool
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(
                self.thread_executor,
                self._sync_predict_batch,
                features_df
            )
            
            return predictions
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def _sync_predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Synchronous prediction for single record"""
        start_time = time.time()
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_df)
            probability = probabilities[0, 1]  # Fraud class probability
        else:
            prediction = self.model.predict(features_df)
            probability = float(prediction[0])
        
        # Binary prediction
        binary_prediction = probability > 0.5
        
        # Confidence score
        confidence = abs(probability - 0.5) * 2
        
        # Risk level
        if probability < 0.3:
            risk_level = 'low'
        elif probability < 0.7:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'probability': probability,
            'prediction': binary_prediction,
            'confidence': confidence,
            'risk_level': risk_level,
            'processing_time_ms': processing_time
        }
    
    def _sync_predict_batch(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Synchronous batch prediction"""
        start_time = time.time()
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_df)[:, 1]
        else:
            predictions = self.model.predict(features_df)
            probabilities = predictions.astype(float)
        
        # Binary predictions
        binary_predictions = (probabilities > 0.5).tolist()
        
        # Confidence scores
        confidence_scores = (np.abs(probabilities - 0.5) * 2).tolist()
        
        # Risk levels
        risk_levels = []
        for prob in probabilities:
            if prob < 0.3:
                risk_levels.append('low')
            elif prob < 0.7:
                risk_levels.append('medium')
            else:
                risk_levels.append('high')
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'probabilities': probabilities.tolist(),
            'predictions': binary_predictions,
            'confidence_scores': confidence_scores,
            'risk_levels': risk_levels,
            'processing_time_ms': processing_time
        }
    
    def _generate_cache_key(self, request: ScoringRequest) -> str:
        """Generate cache key for scoring request"""
        key_components = [
            str(request.user_id),
            str(request.transaction_data.get('amount', '')),
            str(request.transaction_data.get('merchant_id', '')),
            str(request.transaction_data.get('timestamp', '')),
            self.model_version
        ]
        
        key_str = '|'.join(key_components)
        return f"score:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            cache_stats = {}
            if self.model_cache:
                cache_stats = await self.model_cache.get_cache_statistics()
            
            return {
                'model_version': self.model_version,
                'model_loaded': self.model_loaded,
                'performance_metrics': self.performance_metrics,
                'cache_statistics': cache_stats,
                'batch_queue_size': self.batch_processor.request_queue.qsize(),
                'feature_pipeline_stats': self.feature_pipeline.get_feature_statistics() if self.feature_pipeline else {}
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for scoring engine"""
        try:
            # Test model prediction
            test_features = pd.DataFrame([{
                'amount': 100.0,
                'hour_of_day': 12,
                'is_weekend': False,
                'payment_method': 'credit_card'
            }])
            
            test_prediction = await self._predict_single(test_features)
            
            return {
                'status': 'healthy',
                'model_loaded': self.model_loaded,
                'model_version': self.model_version,
                'test_prediction_successful': True,
                'batch_processor_running': self.batch_processor.processing,
                'feature_pipeline_available': self.feature_pipeline is not None,
                'cache_available': self.model_cache is not None
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'model_loaded': self.model_loaded,
                'model_version': self.model_version
            }
    
    async def shutdown(self):
        """Shutdown scoring engine"""
        try:
            # Stop batch processor
            await self.batch_processor.stop_processing()
            
            # Shutdown thread executor
            self.thread_executor.shutdown(wait=True)
            
            logger.info("Scoring engine shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Factory function
async def create_scoring_engine(
    model_path: str = None,
    redis_url: str = "redis://localhost:6379",
    mongodb_url: str = "mongodb://localhost:27017",
    feature_config: FeatureConfig = None
) -> ScoringEngine:
    """Factory function to create and initialize scoring engine"""
    
    # Initialize clients
    redis_client = aioredis.from_url(redis_url)
    db_client = AsyncIOMotorClient(mongodb_url)
    
    # Create scoring engine
    engine = ScoringEngine(
        model_path=model_path,
        redis_client=redis_client,
        db_client=db_client,
        feature_config=feature_config
    )
    
    # Initialize
    await engine.initialize(model_path)
    
    return engine


# Example usage and testing
async def test_scoring_engine():
    """Test scoring engine functionality"""
    
    # Create engine
    engine = await create_scoring_engine()
    
    # Test single scoring
    test_request = ScoringRequest(
        request_id="test_001",
        user_id=12345,
        transaction_data={
            "amount": 150.0,
            "currency": "USD",
            "merchant_id": "merchant_001",
            "payment_method": "credit_card",
            "timestamp": datetime.now().isoformat()
        },
        context={
            "ip_address": "192.168.1.1",
            "device_id": "device_001"
        },
        mode=ScoringMode.REAL_TIME
    )
    
    result = await engine.score_single(test_request)
    print(f"Single scoring result: {result}")
    
    # Test batch scoring
    batch_requests = []
    for i in range(10):
        req = ScoringRequest(
            request_id=f"batch_test_{i:03d}",
            user_id=12345 + i,
            transaction_data={
                "amount": 100.0 + i * 10,
                "currency": "USD",
                "merchant_id": f"merchant_{i:03d}",
                "payment_method": "credit_card",
                "timestamp": datetime.now().isoformat()
            },
            context={},
            mode=ScoringMode.BATCH
        )
        batch_requests.append(req)
    
    batch_results = await engine.score_batch(batch_requests)
    print(f"Batch scoring completed: {len(batch_results)} results")
    
    # Get performance metrics
    metrics = await engine.get_performance_metrics()
    print(f"Performance metrics: {metrics}")
    
    # Health check
    health = await engine.health_check()
    print(f"Health check: {health}")
    
    # Shutdown
    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(test_scoring_engine())
