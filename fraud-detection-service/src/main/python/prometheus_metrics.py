"""
Custom Prometheus metrics for IntelliFlow Fraud Detection Service
"""

import time
import logging
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from prometheus_client.multiprocess import MultiProcessCollector
import os

logger = logging.getLogger(__name__)


class FraudDetectionMetrics:
    """
    Comprehensive Prometheus metrics for fraud detection service
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics with optional custom registry"""
        self.registry = registry or CollectorRegistry()
        self._initialize_metrics()
        
    def _initialize_metrics(self):
        """Initialize all Prometheus metrics"""
        
        # Business Metrics
        self.fraud_predictions_total = Counter(
            'fraud_predictions_total',
            'Total number of fraud predictions made',
            ['model_version', 'prediction', 'risk_level'],
            registry=self.registry
        )
        
        self.fraud_detection_accuracy = Gauge(
            'fraud_detection_accuracy',
            'Current fraud detection model accuracy',
            ['model_version', 'metric_type'],
            registry=self.registry
        )
        
        self.transaction_processing_total = Counter(
            'transaction_processing_total',
            'Total number of transactions processed',
            ['processing_type', 'status'],
            registry=self.registry
        )
        
        # Performance Metrics
        self.fraud_prediction_duration_seconds = Histogram(
            'fraud_prediction_duration_seconds',
            'Time spent making fraud predictions',
            ['model_type', 'feature_count'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0],
            registry=self.registry
        )
        
        self.feature_extraction_duration_seconds = Histogram(
            'feature_extraction_duration_seconds',
            'Time spent extracting features',
            ['feature_type', 'source'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        self.model_loading_duration_seconds = Histogram(
            'model_loading_duration_seconds',
            'Time spent loading ML models',
            ['model_type', 'model_version'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        # System Metrics
        self.active_model_info = Info(
            'active_model_info',
            'Information about currently active models',
            registry=self.registry
        )
        
        self.model_memory_usage_bytes = Gauge(
            'model_memory_usage_bytes',
            'Memory usage of loaded models',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        self.feature_cache_operations_total = Counter(
            'feature_cache_operations_total',
            'Total feature cache operations',
            ['operation', 'result'],
            registry=self.registry
        )
        
        self.database_operations_total = Counter(
            'database_operations_total',
            'Total database operations',
            ['operation', 'table', 'status'],
            registry=self.registry
        )
        
        # Error Metrics
        self.prediction_errors_total = Counter(
            'prediction_errors_total',
            'Total prediction errors',
            ['error_type', 'model_version'],
            registry=self.registry
        )
        
        self.system_errors_total = Counter(
            'system_errors_total',
            'Total system errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Queue Metrics
        self.prediction_queue_size = Gauge(
            'prediction_queue_size',
            'Current size of prediction queue',
            ['queue_type'],
            registry=self.registry
        )
        
        self.batch_processing_size = Histogram(
            'batch_processing_size',
            'Size of batches processed',
            ['processing_type'],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
            registry=self.registry
        )
        
        # Model Performance Metrics
        self.model_drift_score = Gauge(
            'model_drift_score',
            'Model drift detection score',
            ['model_version', 'drift_type'],
            registry=self.registry
        )
        
        self.feature_importance_scores = Gauge(
            'feature_importance_scores',
            'Feature importance scores from models',
            ['feature_name', 'model_version'],
            registry=self.registry
        )
        
        # API Metrics
        self.api_requests_total = Counter(
            'api_requests_total',
            'Total API requests',
            ['endpoint', 'method', 'status_code'],
            registry=self.registry
        )
        
        self.api_request_duration_seconds = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['endpoint', 'method'],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0],
            registry=self.registry
        )
        
        # Data Quality Metrics
        self.data_quality_score = Gauge(
            'data_quality_score',
            'Data quality score for inputs',
            ['data_source', 'quality_dimension'],
            registry=self.registry
        )
        
        self.missing_features_total = Counter(
            'missing_features_total',
            'Total count of missing features',
            ['feature_name', 'source'],
            registry=self.registry
        )
        
        # A/B Testing Metrics
        self.ab_test_assignments_total = Counter(
            'ab_test_assignments_total',
            'Total A/B test assignments',
            ['experiment_id', 'variant'],
            registry=self.registry
        )
        
        self.ab_test_conversions_total = Counter(
            'ab_test_conversions_total',
            'Total A/B test conversions',
            ['experiment_id', 'variant', 'conversion_type'],
            registry=self.registry
        )
        
    # Business Metric Methods
    
    def record_fraud_prediction(self, model_version: str, prediction: bool, 
                              risk_level: str, duration: float, model_type: str = "default"):
        """Record a fraud prediction"""
        prediction_str = "fraud" if prediction else "legitimate"
        
        self.fraud_predictions_total.labels(
            model_version=model_version,
            prediction=prediction_str,
            risk_level=risk_level
        ).inc()
        
        self.fraud_prediction_duration_seconds.labels(
            model_type=model_type,
            feature_count="unknown"  # Can be updated with actual feature count
        ).observe(duration)
    
    def record_batch_prediction(self, batch_size: int, processing_time: float, 
                              results: List[Dict[str, Any]]):
        """Record batch prediction metrics"""
        self.batch_processing_size.labels(
            processing_type="fraud_detection"
        ).observe(batch_size)
        
        # Process individual predictions in batch
        for result in results:
            model_version = result.get('model_version', 'unknown')
            prediction = result.get('fraud_prediction', False)
            risk_level = result.get('risk_level', 'unknown')
            
            self.record_fraud_prediction(model_version, prediction, risk_level, 0)
    
    def update_model_accuracy(self, model_version: str, accuracy: float, 
                            precision: float, recall: float, f1_score: float):
        """Update model performance metrics"""
        self.fraud_detection_accuracy.labels(
            model_version=model_version,
            metric_type="accuracy"
        ).set(accuracy)
        
        self.fraud_detection_accuracy.labels(
            model_version=model_version,
            metric_type="precision"
        ).set(precision)
        
        self.fraud_detection_accuracy.labels(
            model_version=model_version,
            metric_type="recall"
        ).set(recall)
        
        self.fraud_detection_accuracy.labels(
            model_version=model_version,
            metric_type="f1_score"
        ).set(f1_score)
    
    def record_feature_extraction(self, feature_type: str, source: str, duration: float):
        """Record feature extraction time"""
        self.feature_extraction_duration_seconds.labels(
            feature_type=feature_type,
            source=source
        ).observe(duration)
    
    def record_model_loading(self, model_type: str, model_version: str, duration: float):
        """Record model loading time"""
        self.model_loading_duration_seconds.labels(
            model_type=model_type,
            model_version=model_version
        ).observe(duration)
    
    # Cache and Database Methods
    
    def record_cache_operation(self, operation: str, result: str):
        """Record cache operation (hit/miss/error)"""
        self.feature_cache_operations_total.labels(
            operation=operation,
            result=result
        ).inc()
    
    def record_database_operation(self, operation: str, table: str, status: str):
        """Record database operation"""
        self.database_operations_total.labels(
            operation=operation,
            table=table,
            status=status
        ).inc()
    
    # Error Tracking Methods
    
    def record_prediction_error(self, error_type: str, model_version: str):
        """Record prediction error"""
        self.prediction_errors_total.labels(
            error_type=error_type,
            model_version=model_version
        ).inc()
    
    def record_system_error(self, error_type: str, component: str):
        """Record system error"""
        self.system_errors_total.labels(
            error_type=error_type,
            component=component
        ).inc()
    
    # API Metrics Methods
    
    def record_api_request(self, endpoint: str, method: str, status_code: int, duration: float):
        """Record API request metrics"""
        self.api_requests_total.labels(
            endpoint=endpoint,
            method=method,
            status_code=str(status_code)
        ).inc()
        
        self.api_request_duration_seconds.labels(
            endpoint=endpoint,
            method=method
        ).observe(duration)
    
    # System State Methods
    
    def set_active_model_info(self, model_info: Dict[str, str]):
        """Set active model information"""
        self.active_model_info.info(model_info)
    
    def set_model_memory_usage(self, model_name: str, model_version: str, memory_bytes: int):
        """Set model memory usage"""
        self.model_memory_usage_bytes.labels(
            model_name=model_name,
            model_version=model_version
        ).set(memory_bytes)
    
    def set_queue_size(self, queue_type: str, size: int):
        """Set queue size"""
        self.prediction_queue_size.labels(queue_type=queue_type).set(size)
    
    def set_data_quality_score(self, data_source: str, quality_dimension: str, score: float):
        """Set data quality score"""
        self.data_quality_score.labels(
            data_source=data_source,
            quality_dimension=quality_dimension
        ).set(score)
    
    def record_missing_feature(self, feature_name: str, source: str):
        """Record missing feature"""
        self.missing_features_total.labels(
            feature_name=feature_name,
            source=source
        ).inc()
    
    # A/B Testing Methods
    
    def record_ab_test_assignment(self, experiment_id: str, variant: str):
        """Record A/B test assignment"""
        self.ab_test_assignments_total.labels(
            experiment_id=experiment_id,
            variant=variant
        ).inc()
    
    def record_ab_test_conversion(self, experiment_id: str, variant: str, conversion_type: str):
        """Record A/B test conversion"""
        self.ab_test_conversions_total.labels(
            experiment_id=experiment_id,
            variant=variant,
            conversion_type=conversion_type
        ).inc()
    
    # Context Managers for Timing
    
    @contextmanager
    def time_prediction(self, model_type: str = "default", feature_count: str = "unknown"):
        """Context manager for timing predictions"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.fraud_prediction_duration_seconds.labels(
                model_type=model_type,
                feature_count=feature_count
            ).observe(duration)
    
    @contextmanager
    def time_feature_extraction(self, feature_type: str, source: str):
        """Context manager for timing feature extraction"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_feature_extraction(feature_type, source, duration)
    
    @contextmanager
    def time_api_request(self, endpoint: str, method: str):
        """Context manager for timing API requests"""
        start_time = time.time()
        status_code = 200
        try:
            yield
        except Exception:
            status_code = 500
            raise
        finally:
            duration = time.time() - start_time
            self.record_api_request(endpoint, method, status_code, duration)
    
    # Advanced Metrics Methods
    
    def set_model_drift_score(self, model_version: str, drift_type: str, score: float):
        """Set model drift score"""
        self.model_drift_score.labels(
            model_version=model_version,
            drift_type=drift_type
        ).set(score)
    
    def set_feature_importance(self, feature_name: str, model_version: str, importance: float):
        """Set feature importance score"""
        self.feature_importance_scores.labels(
            feature_name=feature_name,
            model_version=model_version
        ).set(importance)
    
    def update_transaction_processing(self, processing_type: str, status: str, count: int = 1):
        """Update transaction processing metrics"""
        self.transaction_processing_total.labels(
            processing_type=processing_type,
            status=status
        ).inc(count)
    
    # Utility Methods
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry)
    
    def get_content_type(self) -> str:
        """Get content type for metrics endpoint"""
        return CONTENT_TYPE_LATEST
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        logger.warning("Resetting all metrics - this should only be used in testing!")
        # Note: Prometheus metrics cannot be easily reset in production
        # This method is primarily for testing purposes
        pass


# Global metrics instance
_metrics_instance: Optional[FraudDetectionMetrics] = None


def get_metrics() -> FraudDetectionMetrics:
    """Get the global metrics instance"""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = FraudDetectionMetrics()
    return _metrics_instance


def init_metrics(registry: Optional[CollectorRegistry] = None) -> FraudDetectionMetrics:
    """Initialize metrics with custom registry"""
    global _metrics_instance
    _metrics_instance = FraudDetectionMetrics(registry)
    return _metrics_instance


# Decorator for automatic metric collection
def track_prediction_time(model_type: str = "default"):
    """Decorator to automatically track prediction time"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            with metrics.time_prediction(model_type):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def track_feature_extraction_time(feature_type: str, source: str):
    """Decorator to automatically track feature extraction time"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            with metrics.time_feature_extraction(feature_type, source):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def track_api_request_time(endpoint: str, method: str):
    """Decorator to automatically track API request time"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            with metrics.time_api_request(endpoint, method):
                return func(*args, **kwargs)
        return wrapper
    return decorator