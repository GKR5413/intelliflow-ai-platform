"""
Advanced feature extraction and preprocessing pipelines for fraud detection
"""

import os
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import json
import hashlib

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import joblib

# Async database and cache clients
import aioredis
import asyncpg
from motor.motor_asyncio import AsyncIOMotorClient

# Monitoring
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Configuration
logger = structlog.get_logger()

# Metrics
FEATURE_EXTRACTION_LATENCY = Histogram('feature_extraction_duration_seconds', 'Feature extraction latency', ['feature_type'])
FEATURE_CACHE_HITS = Counter('feature_cache_hits_total', 'Feature cache hits', ['feature_type'])
FEATURE_CACHE_MISSES = Counter('feature_cache_misses_total', 'Feature cache misses', ['feature_type'])
FEATURE_COMPUTATION_ERRORS = Counter('feature_computation_errors_total', 'Feature computation errors', ['feature_type'])


@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    enable_real_time_features: bool = True
    enable_historical_features: bool = True
    enable_behavioral_features: bool = True
    enable_network_features: bool = True
    
    # Cache settings
    cache_ttl_seconds: int = 300
    enable_feature_caching: bool = True
    
    # Computation windows
    velocity_windows: List[str] = None
    aggregation_windows: List[str] = None
    
    # Feature selection
    max_features: Optional[int] = None
    feature_selection_method: str = 'mutual_info'
    
    # Preprocessing
    scaling_method: str = 'robust'  # 'standard', 'minmax', 'robust'
    handle_missing: str = 'median'  # 'mean', 'median', 'knn'
    
    def __post_init__(self):
        if self.velocity_windows is None:
            self.velocity_windows = ['1m', '5m', '15m', '1h', '6h', '24h']
        if self.aggregation_windows is None:
            self.aggregation_windows = ['1h', '6h', '24h', '7d', '30d']


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors"""
    
    @abstractmethod
    async def extract_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from context"""
        pass
    
    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Get list of feature names this extractor produces"""
        pass


class TransactionFeatureExtractor(FeatureExtractor):
    """Extract transaction-level features"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        
    async def extract_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract transaction features"""
        transaction = context.get('transaction', {})
        
        features = {}
        
        # Basic transaction features
        features.update(self._extract_basic_features(transaction))
        
        # Amount-based features
        features.update(self._extract_amount_features(transaction))
        
        # Time-based features
        features.update(self._extract_time_features(transaction))
        
        # Geographic features
        features.update(self._extract_geographic_features(transaction))
        
        # Payment method features
        features.update(self._extract_payment_features(transaction))
        
        return features
    
    def _extract_basic_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic transaction features"""
        return {
            'amount': float(transaction.get('amount', 0)),
            'currency': transaction.get('currency', 'USD'),
            'payment_method': transaction.get('payment_method', 'unknown'),
            'merchant_category': transaction.get('merchant_category', 'unknown'),
            'channel': transaction.get('channel', 'online'),
        }
    
    def _extract_amount_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Extract amount-based features"""
        amount = float(transaction.get('amount', 0))
        
        return {
            'amount_log': np.log1p(amount),
            'amount_sqrt': np.sqrt(amount),
            'amount_category': self._categorize_amount(amount),
            'is_round_amount': self._is_round_amount(amount),
            'amount_digits': len(str(int(amount))),
        }
    
    def _extract_time_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Extract time-based features"""
        timestamp = transaction.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.now()
        
        return {
            'hour_of_day': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'day_of_month': timestamp.day,
            'month': timestamp.month,
            'quarter': (timestamp.month - 1) // 3 + 1,
            'is_weekend': timestamp.weekday() >= 5,
            'is_holiday': self._is_holiday(timestamp),
            'is_business_hours': 9 <= timestamp.hour <= 17,
            'is_night_time': timestamp.hour < 6 or timestamp.hour > 22,
            'time_since_midnight': timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second,
        }
    
    def _extract_geographic_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Extract geographic features"""
        user_country = transaction.get('user_country', 'unknown')
        merchant_country = transaction.get('merchant_country', 'unknown')
        
        return {
            'user_country': user_country,
            'merchant_country': merchant_country,
            'is_international': user_country != merchant_country,
            'is_high_risk_country': self._is_high_risk_country(merchant_country),
            'country_risk_score': self._get_country_risk_score(merchant_country),
        }
    
    def _extract_payment_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Extract payment method features"""
        payment_method = transaction.get('payment_method', 'unknown')
        
        return {
            'payment_method': payment_method,
            'is_card_payment': payment_method in ['credit_card', 'debit_card'],
            'is_digital_payment': payment_method in ['digital_wallet', 'cryptocurrency'],
            'is_high_risk_payment': payment_method in ['cryptocurrency', 'prepaid_card'],
            'payment_risk_score': self._get_payment_risk_score(payment_method),
        }
    
    def _categorize_amount(self, amount: float) -> str:
        """Categorize transaction amount"""
        if amount <= 10:
            return 'micro'
        elif amount <= 50:
            return 'small'
        elif amount <= 200:
            return 'medium'
        elif amount <= 1000:
            return 'large'
        elif amount <= 5000:
            return 'very_large'
        else:
            return 'extreme'
    
    def _is_round_amount(self, amount: float) -> bool:
        """Check if amount is a round number"""
        return amount % 10 == 0 or amount % 100 == 0
    
    def _is_holiday(self, date: datetime) -> bool:
        """Check if date is a holiday (simplified)"""
        # This would typically check against a holiday calendar
        holidays = [
            (1, 1),   # New Year's Day
            (7, 4),   # July 4th
            (12, 25), # Christmas
        ]
        return (date.month, date.day) in holidays
    
    def _is_high_risk_country(self, country: str) -> bool:
        """Check if country is high risk"""
        high_risk_countries = ['XX', 'YY', 'ZZ']  # Placeholder
        return country in high_risk_countries
    
    def _get_country_risk_score(self, country: str) -> float:
        """Get risk score for country"""
        # This would come from a risk database
        risk_scores = {
            'US': 0.1, 'CA': 0.1, 'UK': 0.15, 'FR': 0.15, 'DE': 0.1,
            'CN': 0.3, 'RU': 0.4, 'XX': 0.8
        }
        return risk_scores.get(country, 0.5)
    
    def _get_payment_risk_score(self, payment_method: str) -> float:
        """Get risk score for payment method"""
        risk_scores = {
            'credit_card': 0.2,
            'debit_card': 0.1,
            'bank_transfer': 0.05,
            'digital_wallet': 0.3,
            'cryptocurrency': 0.8,
            'prepaid_card': 0.6
        }
        return risk_scores.get(payment_method, 0.5)
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names"""
        return [
            'amount', 'currency', 'payment_method', 'merchant_category', 'channel',
            'amount_log', 'amount_sqrt', 'amount_category', 'is_round_amount', 'amount_digits',
            'hour_of_day', 'day_of_week', 'day_of_month', 'month', 'quarter',
            'is_weekend', 'is_holiday', 'is_business_hours', 'is_night_time', 'time_since_midnight',
            'user_country', 'merchant_country', 'is_international', 'is_high_risk_country', 'country_risk_score',
            'is_card_payment', 'is_digital_payment', 'is_high_risk_payment', 'payment_risk_score'
        ]


class BehavioralFeatureExtractor(FeatureExtractor):
    """Extract user behavioral features"""
    
    def __init__(self, config: FeatureConfig, db_client: AsyncIOMotorClient):
        self.config = config
        self.db_client = db_client
        self.db = db_client.fraud_detection
        
    async def extract_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract behavioral features"""
        user_id = context.get('user_id')
        if not user_id:
            return {}
        
        features = {}
        
        # Velocity features
        if self.config.enable_behavioral_features:
            features.update(await self._extract_velocity_features(user_id))
            features.update(await self._extract_spending_patterns(user_id))
            features.update(await self._extract_merchant_patterns(user_id))
            features.update(await self._extract_device_patterns(user_id, context.get('device_id')))
        
        return features
    
    async def _extract_velocity_features(self, user_id: int) -> Dict[str, Any]:
        """Extract transaction velocity features"""
        features = {}
        current_time = datetime.now()
        
        for window in self.config.velocity_windows:
            window_delta = self._parse_time_window(window)
            start_time = current_time - window_delta
            
            # Count transactions in window
            count = await self.db.transactions.count_documents({
                'user_id': user_id,
                'timestamp': {'$gte': start_time, '$lt': current_time}
            })
            
            # Sum amount in window
            pipeline = [
                {'$match': {
                    'user_id': user_id,
                    'timestamp': {'$gte': start_time, '$lt': current_time}
                }},
                {'$group': {
                    '_id': None,
                    'total_amount': {'$sum': '$amount'},
                    'avg_amount': {'$avg': '$amount'},
                    'max_amount': {'$max': '$amount'}
                }}
            ]
            
            result = await self.db.transactions.aggregate(pipeline).to_list(1)
            if result:
                amount_stats = result[0]
                features.update({
                    f'txn_count_{window}': count,
                    f'txn_amount_sum_{window}': amount_stats.get('total_amount', 0),
                    f'txn_amount_avg_{window}': amount_stats.get('avg_amount', 0),
                    f'txn_amount_max_{window}': amount_stats.get('max_amount', 0),
                })
            else:
                features.update({
                    f'txn_count_{window}': count,
                    f'txn_amount_sum_{window}': 0,
                    f'txn_amount_avg_{window}': 0,
                    f'txn_amount_max_{window}': 0,
                })
        
        return features
    
    async def _extract_spending_patterns(self, user_id: int) -> Dict[str, Any]:
        """Extract spending pattern features"""
        features = {}
        current_time = datetime.now()
        
        # Last 30 days for pattern analysis
        start_time = current_time - timedelta(days=30)
        
        # Spending by time of day
        pipeline = [
            {'$match': {
                'user_id': user_id,
                'timestamp': {'$gte': start_time}
            }},
            {'$group': {
                '_id': {'$hour': '$timestamp'},
                'count': {'$sum': 1},
                'amount': {'$sum': '$amount'}
            }}
        ]
        
        hourly_stats = await self.db.transactions.aggregate(pipeline).to_list(24)
        
        # Calculate patterns
        night_txns = sum(stat['count'] for stat in hourly_stats if stat['_id'] < 6 or stat['_id'] > 22)
        business_txns = sum(stat['count'] for stat in hourly_stats if 9 <= stat['_id'] <= 17)
        total_txns = sum(stat['count'] for stat in hourly_stats)
        
        features.update({
            'night_txn_ratio_30d': night_txns / max(total_txns, 1),
            'business_txn_ratio_30d': business_txns / max(total_txns, 1),
        })
        
        # Spending by day of week
        pipeline = [
            {'$match': {
                'user_id': user_id,
                'timestamp': {'$gte': start_time}
            }},
            {'$group': {
                '_id': {'$dayOfWeek': '$timestamp'},
                'count': {'$sum': 1},
                'amount': {'$sum': '$amount'}
            }}
        ]
        
        daily_stats = await self.db.transactions.aggregate(pipeline).to_list(7)
        
        weekend_txns = sum(stat['count'] for stat in daily_stats if stat['_id'] in [1, 7])  # Sunday=1, Saturday=7
        
        features.update({
            'weekend_txn_ratio_30d': weekend_txns / max(total_txns, 1),
        })
        
        return features
    
    async def _extract_merchant_patterns(self, user_id: int) -> Dict[str, Any]:
        """Extract merchant interaction patterns"""
        features = {}
        current_time = datetime.now()
        
        for window in ['24h', '7d', '30d']:
            window_delta = self._parse_time_window(window)
            start_time = current_time - window_delta
            
            # Unique merchants
            pipeline = [
                {'$match': {
                    'user_id': user_id,
                    'timestamp': {'$gte': start_time}
                }},
                {'$group': {
                    '_id': '$merchant_id'
                }}
            ]
            
            unique_merchants = len(await self.db.transactions.aggregate(pipeline).to_list(None))
            features[f'unique_merchants_{window}'] = unique_merchants
            
            # Unique categories
            pipeline = [
                {'$match': {
                    'user_id': user_id,
                    'timestamp': {'$gte': start_time}
                }},
                {'$group': {
                    '_id': '$merchant_category'
                }}
            ]
            
            unique_categories = len(await self.db.transactions.aggregate(pipeline).to_list(None))
            features[f'unique_categories_{window}'] = unique_categories
        
        return features
    
    async def _extract_device_patterns(self, user_id: int, device_id: Optional[str]) -> Dict[str, Any]:
        """Extract device usage patterns"""
        features = {}
        current_time = datetime.now()
        
        if not device_id:
            return features
        
        # Device history
        start_time = current_time - timedelta(days=30)
        
        # Unique devices used
        pipeline = [
            {'$match': {
                'user_id': user_id,
                'timestamp': {'$gte': start_time}
            }},
            {'$group': {
                '_id': '$device_id'
            }}
        ]
        
        unique_devices = len(await self.db.transactions.aggregate(pipeline).to_list(None))
        features['unique_devices_30d'] = unique_devices
        
        # Is this device new for this user?
        first_seen = await self.db.transactions.find_one(
            {'user_id': user_id, 'device_id': device_id},
            sort=[('timestamp', 1)]
        )
        
        if first_seen:
            days_since_first_seen = (current_time - first_seen['timestamp']).days
            features['device_age_days'] = days_since_first_seen
            features['is_new_device'] = days_since_first_seen <= 1
        else:
            features['device_age_days'] = 0
            features['is_new_device'] = True
        
        return features
    
    def _parse_time_window(self, window: str) -> timedelta:
        """Parse time window string to timedelta"""
        if window.endswith('m'):
            return timedelta(minutes=int(window[:-1]))
        elif window.endswith('h'):
            return timedelta(hours=int(window[:-1]))
        elif window.endswith('d'):
            return timedelta(days=int(window[:-1]))
        else:
            raise ValueError(f"Invalid time window: {window}")
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names"""
        names = []
        
        # Velocity features
        for window in self.config.velocity_windows:
            names.extend([
                f'txn_count_{window}',
                f'txn_amount_sum_{window}',
                f'txn_amount_avg_{window}',
                f'txn_amount_max_{window}'
            ])
        
        # Pattern features
        names.extend([
            'night_txn_ratio_30d',
            'business_txn_ratio_30d', 
            'weekend_txn_ratio_30d'
        ])
        
        # Merchant patterns
        for window in ['24h', '7d', '30d']:
            names.extend([
                f'unique_merchants_{window}',
                f'unique_categories_{window}'
            ])
        
        # Device patterns
        names.extend([
            'unique_devices_30d',
            'device_age_days',
            'is_new_device'
        ])
        
        return names


class NetworkFeatureExtractor(FeatureExtractor):
    """Extract network-based features"""
    
    def __init__(self, config: FeatureConfig, redis_client: aioredis.Redis):
        self.config = config
        self.redis_client = redis_client
        
    async def extract_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract network features"""
        features = {}
        
        if not self.config.enable_network_features:
            return features
        
        ip_address = context.get('ip_address')
        device_id = context.get('device_id')
        user_id = context.get('user_id')
        
        if ip_address:
            features.update(await self._extract_ip_features(ip_address))
        
        if device_id and user_id:
            features.update(await self._extract_device_network_features(device_id, user_id))
        
        return features
    
    async def _extract_ip_features(self, ip_address: str) -> Dict[str, Any]:
        """Extract IP-based features"""
        features = {}
        
        # Check IP reputation (cached)
        cache_key = f"ip_reputation:{ip_address}"
        cached_rep = await self.redis_client.get(cache_key)
        
        if cached_rep:
            reputation = json.loads(cached_rep)
        else:
            # Would call external IP reputation service
            reputation = await self._get_ip_reputation(ip_address)
            await self.redis_client.setex(cache_key, 3600, json.dumps(reputation))
        
        features.update({
            'ip_reputation_score': reputation.get('score', 0.5),
            'is_vpn': reputation.get('is_vpn', False),
            'is_tor': reputation.get('is_tor', False),
            'is_proxy': reputation.get('is_proxy', False),
            'ip_country': reputation.get('country', 'unknown'),
            'ip_asn': reputation.get('asn', 'unknown'),
        })
        
        # Check how many users used this IP recently
        user_count_key = f"ip_users:{ip_address}"
        user_count = await self.redis_client.scard(user_count_key)
        features['ip_user_count_24h'] = user_count
        
        return features
    
    async def _extract_device_network_features(self, device_id: str, user_id: int) -> Dict[str, Any]:
        """Extract device network features"""
        features = {}
        
        # Check device sharing
        device_users_key = f"device_users:{device_id}"
        user_count = await self.redis_client.scard(device_users_key)
        features['device_user_count'] = user_count
        features['is_shared_device'] = user_count > 1
        
        # Check location consistency
        location_key = f"user_locations:{user_id}"
        location_count = await self.redis_client.scard(location_key)
        features['location_diversity_7d'] = location_count
        
        return features
    
    async def _get_ip_reputation(self, ip_address: str) -> Dict[str, Any]:
        """Get IP reputation from external service (mock)"""
        # This would integrate with services like VirusTotal, AbuseIPDB, etc.
        return {
            'score': np.random.uniform(0, 1),
            'is_vpn': np.random.choice([True, False], p=[0.05, 0.95]),
            'is_tor': np.random.choice([True, False], p=[0.01, 0.99]),
            'is_proxy': np.random.choice([True, False], p=[0.03, 0.97]),
            'country': np.random.choice(['US', 'CA', 'UK', 'FR', 'DE', 'CN']),
            'asn': f"AS{np.random.randint(1000, 99999)}"
        }
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names"""
        return [
            'ip_reputation_score', 'is_vpn', 'is_tor', 'is_proxy',
            'ip_country', 'ip_asn', 'ip_user_count_24h',
            'device_user_count', 'is_shared_device', 'location_diversity_7d'
        ]


class FeaturePipeline:
    """Main feature extraction and preprocessing pipeline"""
    
    def __init__(self, 
                 config: FeatureConfig,
                 redis_client: aioredis.Redis,
                 db_client: AsyncIOMotorClient,
                 postgres_pool: asyncpg.Pool = None):
        
        self.config = config
        self.redis_client = redis_client
        self.db_client = db_client
        self.postgres_pool = postgres_pool
        
        # Initialize extractors
        self.extractors = {
            'transaction': TransactionFeatureExtractor(config),
            'behavioral': BehavioralFeatureExtractor(config, db_client),
            'network': NetworkFeatureExtractor(config, redis_client)
        }
        
        # Preprocessing components
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        self.pca = None
        
        # Feature metadata
        self.feature_names = []
        self.feature_importance = {}
        
    async def extract_features(self, context: Dict[str, Any]) -> pd.DataFrame:
        """Extract all features for given context"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(context)
            if self.config.enable_feature_caching:
                cached_features = await self._get_cached_features(cache_key)
                if cached_features is not None:
                    FEATURE_CACHE_HITS.labels(feature_type='all').inc()
                    return cached_features
            
            FEATURE_CACHE_MISSES.labels(feature_type='all').inc()
            
            # Extract features from all extractors
            all_features = {}
            
            for extractor_name, extractor in self.extractors.items():
                try:
                    extractor_start = time.time()
                    features = await extractor.extract_features(context)
                    all_features.update(features)
                    
                    FEATURE_EXTRACTION_LATENCY.labels(feature_type=extractor_name).observe(
                        time.time() - extractor_start
                    )
                    
                except Exception as e:
                    logger.error(f"Feature extraction failed for {extractor_name}: {e}")
                    FEATURE_COMPUTATION_ERRORS.labels(feature_type=extractor_name).inc()
                    # Continue with other extractors
            
            # Convert to DataFrame
            features_df = pd.DataFrame([all_features])
            
            # Cache features
            if self.config.enable_feature_caching:
                await self._cache_features(cache_key, features_df)
            
            total_time = time.time() - start_time
            FEATURE_EXTRACTION_LATENCY.labels(feature_type='total').observe(total_time)
            
            logger.info(f"Feature extraction completed in {total_time:.3f}s, {len(features_df.columns)} features")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            FEATURE_COMPUTATION_ERRORS.labels(feature_type='pipeline').inc()
            raise
    
    async def preprocess_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess extracted features"""
        
        # Handle missing values
        if self.imputer is None:
            if self.config.handle_missing == 'knn':
                self.imputer = KNNImputer(n_neighbors=5)
            else:
                strategy = 'median' if self.config.handle_missing == 'median' else 'mean'
                self.imputer = SimpleImputer(strategy=strategy)
            
            # Fit imputer on numeric columns only
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                self.imputer.fit(features_df[numeric_cols])
        
        # Apply imputation
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0 and self.imputer is not None:
            features_df[numeric_cols] = self.imputer.transform(features_df[numeric_cols])
        
        # Handle categorical variables
        features_df = self._encode_categorical_features(features_df)
        
        # Feature scaling
        features_df = self._scale_features(features_df)
        
        # Feature selection
        if self.config.max_features and self.config.max_features < len(features_df.columns):
            features_df = self._select_features(features_df)
        
        return features_df
    
    def _encode_categorical_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        categorical_cols = features_df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # Simple label encoding for now
            unique_vals = features_df[col].unique()
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            features_df[col] = features_df[col].map(mapping).fillna(-1)
        
        return features_df
    
    def _scale_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return features_df
        
        if self.scaler is None:
            if self.config.scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif self.config.scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            else:  # robust
                self.scaler = RobustScaler()
            
            self.scaler.fit(features_df[numeric_cols])
        
        features_df[numeric_cols] = self.scaler.transform(features_df[numeric_cols])
        
        return features_df
    
    def _select_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Select most important features"""
        if self.feature_selector is None:
            # Use mutual information for feature selection (unsupervised)
            # In practice, this would be fitted during training
            self.feature_selector = SelectKBest(k=self.config.max_features)
            # For now, just return the dataframe as-is since we don't have labels
            return features_df
        
        selected_features = self.feature_selector.transform(features_df)
        selected_columns = features_df.columns[self.feature_selector.get_support()]
        
        return pd.DataFrame(selected_features, columns=selected_columns, index=features_df.index)
    
    def _generate_cache_key(self, context: Dict[str, Any]) -> str:
        """Generate cache key for feature context"""
        # Create a hash from relevant context fields
        key_components = [
            str(context.get('user_id', '')),
            str(context.get('merchant_id', '')),
            str(context.get('device_id', '')),
            str(context.get('transaction', {}).get('amount', '')),
            str(context.get('transaction', {}).get('timestamp', '')),
        ]
        
        key_str = '|'.join(key_components)
        return f"features:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    async def _get_cached_features(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get cached features"""
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                features_dict = json.loads(cached_data)
                return pd.DataFrame([features_dict])
            return None
        except Exception as e:
            logger.warning(f"Failed to get cached features: {e}")
            return None
    
    async def _cache_features(self, cache_key: str, features_df: pd.DataFrame):
        """Cache features"""
        try:
            features_dict = features_df.iloc[0].to_dict()
            await self.redis_client.setex(
                cache_key,
                self.config.cache_ttl_seconds,
                json.dumps(features_dict, default=str)
            )
        except Exception as e:
            logger.warning(f"Failed to cache features: {e}")
    
    async def update_user_context(self, context: Dict[str, Any]):
        """Update user context for future feature extraction"""
        user_id = context.get('user_id')
        device_id = context.get('device_id')
        ip_address = context.get('ip_address')
        
        if not any([user_id, device_id, ip_address]):
            return
        
        current_time = datetime.now()
        
        # Update device-user mapping
        if device_id and user_id:
            device_users_key = f"device_users:{device_id}"
            await self.redis_client.sadd(device_users_key, user_id)
            await self.redis_client.expire(device_users_key, 86400 * 30)  # 30 days
        
        # Update IP-user mapping
        if ip_address and user_id:
            ip_users_key = f"ip_users:{ip_address}"
            await self.redis_client.sadd(ip_users_key, user_id)
            await self.redis_client.expire(ip_users_key, 86400)  # 24 hours
        
        # Update user location
        if user_id and context.get('transaction', {}).get('user_country'):
            location_key = f"user_locations:{user_id}"
            await self.redis_client.sadd(location_key, context['transaction']['user_country'])
            await self.redis_client.expire(location_key, 86400 * 7)  # 7 days
    
    def save_preprocessors(self, filepath: str):
        """Save preprocessing components"""
        preprocessors = {
            'config': asdict(self.config),
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(preprocessors, filepath)
        logger.info(f"Preprocessors saved to {filepath}")
    
    @classmethod
    def load_preprocessors(cls, filepath: str, redis_client: aioredis.Redis, 
                          db_client: AsyncIOMotorClient) -> 'FeaturePipeline':
        """Load preprocessing components"""
        preprocessors = joblib.load(filepath)
        
        config = FeatureConfig(**preprocessors['config'])
        pipeline = cls(config, redis_client, db_client)
        
        pipeline.scaler = preprocessors.get('scaler')
        pipeline.imputer = preprocessors.get('imputer')
        pipeline.feature_selector = preprocessors.get('feature_selector')
        pipeline.feature_names = preprocessors.get('feature_names', [])
        pipeline.feature_importance = preprocessors.get('feature_importance', {})
        
        logger.info(f"Preprocessors loaded from {filepath}")
        return pipeline
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get feature pipeline statistics"""
        return {
            'total_extractors': len(self.extractors),
            'feature_count': len(self.feature_names),
            'config': asdict(self.config),
            'preprocessing_components': {
                'scaler': type(self.scaler).__name__ if self.scaler else None,
                'imputer': type(self.imputer).__name__ if self.imputer else None,
                'feature_selector': type(self.feature_selector).__name__ if self.feature_selector else None
            }
        }
