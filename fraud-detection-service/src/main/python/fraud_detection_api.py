"""
FastAPI application for fraud detection model serving
"""

import os
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
import json
import hashlib

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# Pydantic models
from pydantic import BaseModel, Field, validator
from pydantic.types import UUID4

# ML and data processing
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.pyfunc
from sklearn.preprocessing import StandardScaler

# Caching and storage
import redis
import aioredis
from functools import lru_cache

# Monitoring and observability
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import structlog

# Custom imports
import sys
sys.path.append('../../../ml-platform/src/fraud_detection')
sys.path.append('../../../ml-platform/feature_store')

from data_processor import FraudDataProcessor
from feature_store import FraudFeatureStore

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()

# Prometheus metrics
PREDICTION_COUNTER = Counter('fraud_predictions_total', 'Total fraud predictions', ['model_version', 'prediction'])
PREDICTION_LATENCY = Histogram('fraud_prediction_duration_seconds', 'Prediction latency', ['model_version'])
MODEL_LOAD_TIME = Histogram('model_load_duration_seconds', 'Model loading time')
CACHE_HIT_COUNTER = Counter('cache_hits_total', 'Cache hits', ['cache_type'])
CACHE_MISS_COUNTER = Counter('cache_misses_total', 'Cache misses', ['cache_type'])
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active requests')

# LLM metrics
LLM_INVOCATIONS = Counter('llm_verifications_total', 'Total LLM secondary verifications', ['model'])
LLM_ERRORS = Counter('llm_verification_errors_total', 'LLM verification errors', ['reason'])
LLM_LATENCY = Histogram('llm_verification_duration_seconds', 'LLM verification latency', ['model'])
AGREEMENT_COUNTER = Counter('llm_ml_agreement_total', 'Agreement between ML and LLM', ['outcome'])

# Security
security = HTTPBearer()

# Global variables for model and components
model_manager = None
feature_store = None
redis_client = None
llm_client = None


# Pydantic models for API
class TransactionInput(BaseModel):
    """Input model for single transaction scoring"""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: int = Field(..., description="User identifier")
    merchant_id: str = Field(..., description="Merchant identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    currency: str = Field(default="USD", description="Currency code")
    payment_method: str = Field(..., description="Payment method")
    merchant_category: str = Field(..., description="Merchant category")
    user_country: str = Field(..., description="User country")
    merchant_country: str = Field(..., description="Merchant country")
    device_id: Optional[str] = Field(None, description="Device identifier")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Transaction timestamp")
    
    # Additional context
    channel: Optional[str] = Field("online", description="Transaction channel")
    ip_address: Optional[str] = Field(None, description="User IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v
    
    @validator('currency')
    def validate_currency(cls, v):
        valid_currencies = ['USD', 'EUR', 'GBP', 'CAD', 'AUD', 'JPY']
        if v.upper() not in valid_currencies:
            raise ValueError(f'Currency must be one of {valid_currencies}')
        return v.upper()


class BatchTransactionInput(BaseModel):
    """Input model for batch transaction scoring"""
    transactions: List[TransactionInput] = Field(..., description="List of transactions to score")
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    
    @validator('transactions')
    def validate_transactions(cls, v):
        if len(v) == 0:
            raise ValueError('At least one transaction is required')
        if len(v) > 1000:
            raise ValueError('Maximum 1000 transactions per batch')
        return v


class FraudScore(BaseModel):
    """Fraud score output model"""
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1, description="Fraud probability (0-1)")
    fraud_prediction: bool = Field(..., description="Binary fraud prediction")
    risk_level: str = Field(..., description="Risk level (low/medium/high)")
    model_version: str = Field(..., description="Model version used")
    features_used: int = Field(..., description="Number of features used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    confidence_score: float = Field(..., ge=0, le=1, description="Prediction confidence")
    # Optional LLM-assisted verification
    llm_used: Optional[bool] = Field(default=False, description="Whether LLM secondary verification was used")
    llm_score: Optional[float] = Field(default=None, ge=0, le=1, description="LLM-derived risk score (0-1)")
    llm_reason: Optional[str] = Field(default=None, description="LLM explanation for risk assessment")
    llm_fraud_prediction: Optional[bool] = Field(default=None, description="LLM binary fraud prediction (True=fraud)")
    
    
class BatchFraudScore(BaseModel):
    """Batch fraud score output model"""
    batch_id: str
    results: List[FraudScore]
    total_processed: int
    total_fraud_detected: int
    processing_time_ms: float
    model_version: str


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: datetime
    version: str
    model_status: str
    feature_store_status: str
    cache_status: str
    uptime_seconds: float


class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    model_version: str
    model_type: str
    features_count: int
    training_date: datetime
    performance_metrics: Dict[str, float]
    model_size_mb: float


# Application lifespan events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    logger.info("Starting Fraud Detection Service...")
    
    # Initialize components
    await initialize_components()
    
    logger.info("Fraud Detection Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Fraud Detection Service...")
    await cleanup_components()
    logger.info("Fraud Detection Service stopped")


# Create FastAPI app
app = FastAPI(
    title="Fraud Detection Service",
    description="Real-time fraud detection API with ML model serving",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup time tracking
app.startup_time = time.time()


class ModelManager:
    """Manages ML model loading and inference"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_version = None
        self.model_info = {}
        self.model_loaded = False
        
    async def load_model(self, model_uri: str = None):
        """Load ML model and preprocessor"""
        try:
            start_time = time.time()
            
            # Default model URI
            if model_uri is None:
                model_uri = os.getenv('MODEL_URI', 'models:/fraud_detection_model/latest')
            
            logger.info(f"Loading model from {model_uri}")
            
            # Load model using MLflow
            self.model = mlflow.pyfunc.load_model(model_uri)
            
            # Load preprocessor
            processor_path = os.getenv('PROCESSOR_PATH', '../../../ml-platform/models/fraud_processor.joblib')
            if os.path.exists(processor_path):
                self.processor = FraudDataProcessor.load_processor(processor_path)
            else:
                logger.warning("Processor not found, using default configuration")
                self.processor = FraudDataProcessor()
            
            # Extract model metadata
            self.model_version = self._extract_model_version(model_uri)
            self.model_info = await self._get_model_info()
            
            load_time = time.time() - start_time
            MODEL_LOAD_TIME.observe(load_time)
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully in {load_time:.2f}s, version: {self.model_version}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _extract_model_version(self, model_uri: str) -> str:
        """Extract model version from URI"""
        try:
            if 'models:/' in model_uri:
                parts = model_uri.split('/')
                return parts[-1] if parts[-1] != 'latest' else 'latest'
            return 'unknown'
        except:
            return 'unknown'
    
    async def _get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata"""
        try:
            return {
                'model_type': 'fraud_detection',
                'features_count': getattr(self.processor, 'feature_names', []) and len(self.processor.feature_names) or 0,
                'training_date': datetime.now(),  # Would come from MLflow metadata
                'performance_metrics': {
                    'auc': 0.95,  # Would come from MLflow
                    'precision': 0.92,
                    'recall': 0.89,
                    'f1_score': 0.90
                }
            }
        except Exception as e:
            logger.warning(f"Could not retrieve model info: {e}")
            return {}
    
    async def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Make fraud prediction"""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            start_time = time.time()
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[:, 1]
            else:
                # Fallback for models without predict_proba
                predictions = self.model.predict(features)
                probabilities = predictions.astype(float)
            
            # Binary predictions (threshold = 0.5)
            binary_predictions = (probabilities > 0.5).astype(bool)
            
            # Calculate confidence (distance from threshold)
            confidence_scores = np.abs(probabilities - 0.5) * 2
            
            # Determine risk levels
            risk_levels = []
            for prob in probabilities:
                if prob < 0.3:
                    risk_levels.append('low')
                elif prob < 0.7:
                    risk_levels.append('medium')
                else:
                    risk_levels.append('high')
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                'probabilities': probabilities.tolist(),
                'predictions': binary_predictions.tolist(),
                'confidence_scores': confidence_scores.tolist(),
                'risk_levels': risk_levels,
                'processing_time_ms': processing_time,
                'features_count': len(features.columns)
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}")


class CacheManager:
    """Manages Redis caching for features and predictions"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.default_ttl = 300  # 5 minutes
    
    async def get_cached_features(self, cache_key: str) -> Optional[Dict]:
        """Get cached features"""
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                CACHE_HIT_COUNTER.labels(cache_type='features').inc()
                return json.loads(cached_data)
            else:
                CACHE_MISS_COUNTER.labels(cache_type='features').inc()
                return None
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None
    
    async def set_cached_features(self, cache_key: str, features: Dict, ttl: int = None):
        """Cache features"""
        try:
            ttl = ttl or self.default_ttl
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(features, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
    
    async def get_cached_prediction(self, cache_key: str) -> Optional[Dict]:
        """Get cached prediction"""
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                CACHE_HIT_COUNTER.labels(cache_type='predictions').inc()
                return json.loads(cached_data)
            else:
                CACHE_MISS_COUNTER.labels(cache_type='predictions').inc()
                return None
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None
    
    async def set_cached_prediction(self, cache_key: str, prediction: Dict, ttl: int = 60):
        """Cache prediction result"""
        try:
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(prediction, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")


# Initialize global components
async def initialize_components():
    """Initialize all service components"""
    global model_manager, feature_store, redis_client, llm_client
    
    try:
        # Initialize Redis
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        redis_client = aioredis.from_url(redis_url)
        
        # Initialize model manager
        model_manager = ModelManager()
        await model_manager.load_model()
        
        # Initialize feature store
        feature_store = FraudFeatureStore()
        
        # Initialize LLM client (optional)
        if os.getenv('LLM_ENABLED', 'false').lower() in ('1', 'true', 'yes'):
            llm_client = LLMClient(
                base_url=os.getenv('OLLAMA_URL', 'http://localhost:11434'),
                model=os.getenv('OLLAMA_MODEL', 'mistral')
            )
            logger.info("LLM client initialized", model=os.getenv('OLLAMA_MODEL', 'mistral'))
        else:
            logger.info("LLM client disabled (set LLM_ENABLED=true to enable)")
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Component initialization failed: {e}")
        raise


async def cleanup_components():
    """Cleanup components on shutdown"""
    global redis_client
    
    if redis_client:
        await redis_client.close()


# Dependency functions
async def get_model_manager() -> ModelManager:
    """Get model manager dependency"""
    if not model_manager or not model_manager.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_manager


async def get_cache_manager() -> CacheManager:
    """Get cache manager dependency"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Cache not available")
    return CacheManager(redis_client)


# ------------------------
# LLM Verification Support
# ------------------------

class LLMClient:
    """Minimal client for Ollama-hosted LLMs (e.g., Mistral)."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout_seconds = float(os.getenv('LLM_TIMEOUT_SECONDS', '0.6'))

    async def verify_transaction(self, transaction: 'TransactionInput', features: Dict[str, Any]) -> Dict[str, Any]:
        import httpx
        start_time = time.time()
        prompt = self._build_prompt(transaction, features)
        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 200}
            }
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                text = data.get('response', '')
                parsed = self._parse_llm_response(text)
                LLM_LATENCY.labels(self.model).observe(time.time() - start_time)
                LLM_INVOCATIONS.labels(self.model).inc()
                return parsed
        except Exception as e:
            LLM_ERRORS.labels(reason=type(e).__name__).inc()
            logger.warning("LLM verification failed", error=str(e))
            return {"llm_score": None, "llm_reason": None}

    def _build_prompt(self, transaction: 'TransactionInput', features: Dict[str, Any]) -> str:
        summary = {
            "transaction_id": transaction.transaction_id,
            "amount": transaction.amount,
            "currency": transaction.currency,
            "payment_method": transaction.payment_method,
            "merchant_category": transaction.merchant_category,
            "user_country": transaction.user_country,
            "merchant_country": transaction.merchant_country,
            "channel": transaction.channel,
            "timestamp": transaction.timestamp.isoformat(),
        }
        feature_keys = [
            'risk_score', 'txn_count_24h', 'txn_amount_sum_24h', 'unique_merchants_24h',
            'declined_txn_ratio_7d', 'device_reputation_score', 'merchant_risk_category',
            'fraud_rate_30d', 'transaction_amount', 'is_international', 'hour_of_day', 'day_of_week'
        ]
        compact_features = {k: features.get(k) for k in feature_keys if k in features}
        instruction = (
            "You are a fraud detection assistant. Given the transaction summary and compact feature set, "
            "assess fraud risk. Respond in STRICT JSON with keys: risk_score (0-1 float), reason (short)."
        )
        prompt = (
            f"{instruction}\n\n"
            f"Transaction: {json.dumps(summary)}\n"
            f"Features: {json.dumps(compact_features)}\n"
            f"JSON: "
        )
        return prompt

    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                obj = json.loads(text[start:end + 1])
                score = obj.get('risk_score')
                reason = obj.get('reason')
                if isinstance(score, (int, float)):
                    score = max(0.0, min(1.0, float(score)))
                else:
                    score = None
                if reason is not None:
                    reason = str(reason)[:300]
                return {"llm_score": score, "llm_reason": reason}
        except Exception:
            pass
        return {"llm_score": None, "llm_reason": None}


def _should_use_llm(primary_prob: float, transaction_amount: float) -> bool:
    if os.getenv('LLM_ENABLED', 'false').lower() not in ('1', 'true', 'yes'):
        return False
    try:
        band_low = float(os.getenv('LLM_TRIGGER_MIN_PROB', '0.4'))
        band_high = float(os.getenv('LLM_TRIGGER_MAX_PROB', '0.6'))
        min_amount = float(os.getenv('LLM_TRIGGER_MIN_AMOUNT', '500.0'))
    except Exception:
        band_low, band_high, min_amount = 0.4, 0.6, 500.0
    return (band_low <= primary_prob <= band_high) or (transaction_amount >= min_amount)


def _combine_scores(primary_prob: float, llm_score: Optional[float]) -> float:
    if llm_score is None:
        return primary_prob
    weight = float(os.getenv('LLM_WEIGHT', '0.3'))
    weight = max(0.0, min(0.5, weight))
    return float((1.0 - weight) * primary_prob + weight * llm_score)


def _llm_binary_from_score(llm_score: Optional[float]) -> Optional[bool]:
    if llm_score is None:
        return None
    try:
        threshold = float(os.getenv('LLM_PASS_THRESHOLD', '0.5'))
    except Exception:
        threshold = 0.5
    return bool(llm_score > threshold)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify JWT token (simplified for demo)"""
    # In production, implement proper JWT validation
    token = credentials.credentials
    if not token or token == "invalid":
        raise HTTPException(status_code=401, detail="Invalid token")
    return token


# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request metrics"""
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    try:
        response = await call_next(request)
        return response
    finally:
        ACTIVE_REQUESTS.dec()
        processing_time = time.time() - start_time
        logger.info(
            f"Request processed",
            path=request.url.path,
            method=request.method,
            processing_time=processing_time
        )


# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - app.startup_time
    
    # Check component status
    model_status = "healthy" if model_manager and model_manager.model_loaded else "unhealthy"
    feature_store_status = "healthy" if feature_store else "unhealthy"
    cache_status = "healthy" if redis_client else "unhealthy"
    
    overall_status = "healthy" if all([
        model_status == "healthy",
        feature_store_status == "healthy",
        cache_status == "healthy"
    ]) else "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(),
        version="1.0.0",
        model_status=model_status,
        feature_store_status=feature_store_status,
        cache_status=cache_status,
        uptime_seconds=uptime
    )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info(
    model_mgr: ModelManager = Depends(get_model_manager)
):
    """Get model information"""
    model_info = model_mgr.model_info
    
    return ModelInfo(
        model_name="fraud_detection_model",
        model_version=model_mgr.model_version,
        model_type=model_info.get('model_type', 'unknown'),
        features_count=model_info.get('features_count', 0),
        training_date=model_info.get('training_date', datetime.now()),
        performance_metrics=model_info.get('performance_metrics', {}),
        model_size_mb=0.0  # Would be calculated from actual model
    )


@app.post("/predict", response_model=FraudScore)
async def predict_fraud(
    transaction: TransactionInput,
    background_tasks: BackgroundTasks,
    model_mgr: ModelManager = Depends(get_model_manager),
    cache_mgr: CacheManager = Depends(get_cache_manager),
    token: str = Depends(verify_token)
):
    """Predict fraud for a single transaction"""
    
    start_time = time.time()
    
    try:
        # Create cache key
        transaction_hash = hashlib.md5(
            f"{transaction.transaction_id}_{transaction.amount}_{transaction.timestamp}".encode()
        ).hexdigest()
        cache_key = f"prediction:{transaction_hash}"
        
        # Check cache first
        cached_result = await cache_mgr.get_cached_prediction(cache_key)
        if cached_result:
            logger.info(f"Returning cached prediction for {transaction.transaction_id}")
            return FraudScore(**cached_result)
        
        # Extract features
        features_df = await extract_features_single(transaction, feature_store, cache_mgr)
        
        # Preprocess features
        if model_mgr.processor:
            features_df = model_mgr.processor.transform(features_df)
        
        # Make prediction
        prediction_result = await model_mgr.predict(features_df)
        
        # LLM verification (required if enabled): final decision is AND of ML and LLM non-fraud
        llm_used = False
        llm_score = None
        llm_reason = None
        llm_pred: Optional[bool] = None
        primary_prob = prediction_result['probabilities'][0]
        ml_pred = bool(primary_prob > 0.5)
        final_prob = primary_prob
        require_llm = os.getenv('LLM_REQUIRED', 'true').lower() in ('1', 'true', 'yes')
        fail_closed = os.getenv('LLM_FAIL_CLOSED', 'true').lower() in ('1', 'true', 'yes')
        if _should_use_llm(primary_prob, transaction.amount) and llm_client is not None:
            llm_used = True
            try:
                llm_result = await llm_client.verify_transaction(transaction, features_df.iloc[0].to_dict())
                llm_score = llm_result.get('llm_score')
                llm_reason = llm_result.get('llm_reason')
                llm_pred = _llm_binary_from_score(llm_score)
                # keep final_prob as ML prob to reflect primary model, or optionally combine for score only
                final_prob = _combine_scores(primary_prob, llm_score) if llm_score is not None else primary_prob
            except Exception as e:
                logger.warning("LLM verification failed", error=str(e))
                llm_used = True
                llm_score = None
                llm_reason = None
                llm_pred = None
        else:
            # LLM not used (disabled or condition not met)
            llm_used = False
            llm_pred = None

        # Final decision: require both ML and LLM to agree on non-fraud when LLM is required/used
        # Interpretation: fraud_prediction=True means flagged as fraud
        if llm_used and llm_pred is not None:
            # AND policy: approve (non-fraud) only if both are non-fraud
            # So fraud = ML OR LLM
            final_pred = bool(ml_pred or llm_pred)
            AGREEMENT_COUNTER.labels(outcome='agree' if ml_pred == llm_pred else 'disagree').inc()
        elif require_llm:
            # LLM required but not available; fail-closed -> treat as fraud
            final_pred = True if fail_closed else ml_pred
            AGREEMENT_COUNTER.labels(outcome='llm_missing').inc()
        else:
            # LLM not required; use ML decision
            final_pred = ml_pred

        final_risk = 'low' if final_prob < 0.3 else ('medium' if final_prob < 0.7 else 'high')

        # Create response
        fraud_score = FraudScore(
            transaction_id=transaction.transaction_id,
            fraud_probability=final_prob,
            fraud_prediction=bool(final_pred),
            risk_level=final_risk,
            model_version=model_mgr.model_version,
            features_used=prediction_result['features_count'],
            processing_time_ms=prediction_result['processing_time_ms'],
            confidence_score=prediction_result['confidence_scores'][0],
            llm_used=llm_used,
            llm_score=llm_score,
            llm_reason=llm_reason,
            llm_fraud_prediction=llm_pred
        )
        
        # Cache result
        background_tasks.add_task(
            cache_mgr.set_cached_prediction,
            cache_key,
            fraud_score.dict()
        )
        
        # Update metrics
        PREDICTION_COUNTER.labels(
            model_version=model_mgr.model_version,
            prediction=str(fraud_score.fraud_prediction)
        ).inc()
        
        PREDICTION_LATENCY.labels(model_version=model_mgr.model_version).observe(
            time.time() - start_time
        )
        
        # Log prediction
        logger.info(
            f"Prediction completed",
            transaction_id=transaction.transaction_id,
            fraud_probability=fraud_score.fraud_probability,
            processing_time=fraud_score.processing_time_ms
        )
        
        return fraud_score
        
    except Exception as e:
        logger.error(f"Prediction failed for {transaction.transaction_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchFraudScore)
async def predict_fraud_batch(
    batch_input: BatchTransactionInput,
    background_tasks: BackgroundTasks,
    model_mgr: ModelManager = Depends(get_model_manager),
    cache_mgr: CacheManager = Depends(get_cache_manager),
    token: str = Depends(verify_token)
):
    """Predict fraud for a batch of transactions"""
    
    start_time = time.time()
    batch_id = batch_input.batch_id or f"batch_{int(time.time())}"
    
    try:
        logger.info(f"Processing batch {batch_id} with {len(batch_input.transactions)} transactions")
        
        # Extract features for all transactions
        features_df = await extract_features_batch(batch_input.transactions, feature_store, cache_mgr)
        
        # Preprocess features
        if model_mgr.processor:
            features_df = model_mgr.processor.transform(features_df)
        
        # Make batch prediction
        prediction_result = await model_mgr.predict(features_df)
        
        # Create individual results
        results = []
        for i, transaction in enumerate(batch_input.transactions):
            fraud_score = FraudScore(
                transaction_id=transaction.transaction_id,
                fraud_probability=prediction_result['probabilities'][i],
                fraud_prediction=prediction_result['predictions'][i],
                risk_level=prediction_result['risk_levels'][i],
                model_version=model_mgr.model_version,
                features_used=prediction_result['features_count'],
                processing_time_ms=prediction_result['processing_time_ms'] / len(batch_input.transactions),
                confidence_score=prediction_result['confidence_scores'][i]
            )
            results.append(fraud_score)
        
        # Count fraud predictions
        fraud_count = sum(1 for result in results if result.fraud_prediction)
        
        batch_result = BatchFraudScore(
            batch_id=batch_id,
            results=results,
            total_processed=len(results),
            total_fraud_detected=fraud_count,
            processing_time_ms=(time.time() - start_time) * 1000,
            model_version=model_mgr.model_version
        )
        
        # Update metrics
        for result in results:
            PREDICTION_COUNTER.labels(
                model_version=model_mgr.model_version,
                prediction=str(result.fraud_prediction)
            ).inc()
        
        logger.info(
            f"Batch prediction completed",
            batch_id=batch_id,
            total_processed=len(results),
            fraud_detected=fraud_count,
            processing_time=batch_result.processing_time_ms
        )
        
        return batch_result
        
    except Exception as e:
        logger.error(f"Batch prediction failed for {batch_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/model/reload")
async def reload_model(
    model_uri: Optional[str] = None,
    token: str = Depends(verify_token)
):
    """Reload the ML model"""
    try:
        global model_manager
        
        logger.info(f"Reloading model from {model_uri}")
        
        await model_manager.load_model(model_uri)
        
        return {"status": "success", "message": "Model reloaded successfully"}
        
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


# Feature extraction functions
async def extract_features_single(
    transaction: TransactionInput,
    feature_store: FraudFeatureStore,
    cache_mgr: CacheManager
) -> pd.DataFrame:
    """Extract features for a single transaction"""
    
    # Create feature cache key
    feature_cache_key = f"features:{transaction.user_id}:{transaction.merchant_id}:{transaction.device_id}"
    
    # Check cached features
    cached_features = await cache_mgr.get_cached_features(feature_cache_key)
    
    if cached_features:
        # Use cached features and add transaction-specific ones
        features = cached_features.copy()
    else:
        # Extract features from feature store
        entity_rows = [{
            'user_id': transaction.user_id,
            'merchant_id': transaction.merchant_id,
            'device_id': transaction.device_id or 'unknown',
            'transaction_id': transaction.transaction_id
        }]
        
        try:
            online_features = feature_store.get_online_features(entity_rows, "realtime_fraud_service")
            features = {k: v[0] if isinstance(v, list) and len(v) > 0 else v 
                       for k, v in online_features.items()}
        except Exception as e:
            logger.warning(f"Feature store unavailable, using default features: {e}")
            features = _get_default_features()
        
        # Cache features
        await cache_mgr.set_cached_features(feature_cache_key, features, ttl=300)
    
    # Add transaction-specific features
    features.update({
        'transaction_amount': transaction.amount,
        'currency': transaction.currency,
        'payment_method': transaction.payment_method,
        'merchant_category': transaction.merchant_category,
        'is_international': transaction.user_country != transaction.merchant_country,
        'hour_of_day': transaction.timestamp.hour,
        'day_of_week': transaction.timestamp.weekday(),
        'is_weekend': transaction.timestamp.weekday() >= 5,
        'is_business_hours': 9 <= transaction.timestamp.hour <= 17,
    })
    
    # Convert to DataFrame
    return pd.DataFrame([features])


async def extract_features_batch(
    transactions: List[TransactionInput],
    feature_store: FraudFeatureStore,
    cache_mgr: CacheManager
) -> pd.DataFrame:
    """Extract features for a batch of transactions"""
    
    features_list = []
    
    # Process in smaller batches to avoid overwhelming the feature store
    batch_size = 100
    for i in range(0, len(transactions), batch_size):
        batch = transactions[i:i + batch_size]
        
        # Extract features for this batch
        batch_features = []
        for transaction in batch:
            features_df = await extract_features_single(transaction, feature_store, cache_mgr)
            batch_features.append(features_df.iloc[0].to_dict())
        
        features_list.extend(batch_features)
    
    return pd.DataFrame(features_list)


def _get_default_features() -> Dict[str, Any]:
    """Get default features when feature store is unavailable"""
    return {
        'risk_score': 0.5,
        'txn_count_24h': 5,
        'txn_amount_sum_24h': 500.0,
        'unique_merchants_24h': 3,
        'declined_txn_ratio_7d': 0.1,
        'device_reputation_score': 0.8,
        'merchant_risk_category': 'medium',
        'fraud_rate_30d': 0.02
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().isoformat()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.now().isoformat()}
    )


if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("WORKERS", 1))
    log_level = os.getenv("LOG_LEVEL", "info")
    
    # Run server
    uvicorn.run(
        "fraud_detection_api:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=False
    )
