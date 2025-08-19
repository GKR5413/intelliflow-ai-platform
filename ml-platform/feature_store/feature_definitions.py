"""
Feature definitions for fraud detection using Feast
"""

from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from feast.types import Float32, Float64, Int32, Int64, String, Bool, UnixTimestamp
import os

# Define entities
user_entity = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="User identifier"
)

transaction_entity = Entity(
    name="transaction_id", 
    value_type=ValueType.STRING,
    description="Transaction identifier"
)

merchant_entity = Entity(
    name="merchant_id",
    value_type=ValueType.STRING, 
    description="Merchant identifier"
)

device_entity = Entity(
    name="device_id",
    value_type=ValueType.STRING,
    description="Device identifier"
)

# Data sources - in production these would point to actual data warehouses
base_path = os.path.dirname(os.path.abspath(__file__))

# User profile features source
user_profile_source = FileSource(
    path=os.path.join(base_path, "data/user_profiles.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)

# Transaction features source  
transaction_features_source = FileSource(
    path=os.path.join(base_path, "data/transaction_features.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)

# User behavior aggregations source
user_behavior_source = FileSource(
    path=os.path.join(base_path, "data/user_behavior_aggregations.parquet"),
    timestamp_field="event_timestamp", 
    created_timestamp_column="created_timestamp"
)

# Merchant features source
merchant_features_source = FileSource(
    path=os.path.join(base_path, "data/merchant_features.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)

# Device features source
device_features_source = FileSource(
    path=os.path.join(base_path, "data/device_features.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)

# Real-time transaction features source (for streaming)
realtime_transaction_source = FileSource(
    path=os.path.join(base_path, "data/realtime_transactions.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)

# Feature Views

# User profile features - relatively static user information
user_profile_fv = FeatureView(
    name="user_profile_features",
    entities=["user_id"],
    ttl=timedelta(days=30),  # User profiles don't change frequently
    features=[
        Feature(name="user_age", dtype=Int32),
        Feature(name="account_age_days", dtype=Int32), 
        Feature(name="user_country", dtype=String),
        Feature(name="user_state", dtype=String),
        Feature(name="user_city", dtype=String),
        Feature(name="account_type", dtype=String),  # personal, business
        Feature(name="kyc_status", dtype=String),    # verified, pending, failed
        Feature(name="risk_score", dtype=Float32),   # User risk assessment
        Feature(name="account_balance", dtype=Float64),
        Feature(name="credit_limit", dtype=Float64),
        Feature(name="is_premium_user", dtype=Bool),
        Feature(name="user_segment", dtype=String),  # high_value, regular, new
    ],
    online=True,
    source=user_profile_source,
    tags={"team": "fraud_detection", "category": "user_profile"}
)

# User behavior aggregations - computed features over time windows
user_behavior_fv = FeatureView(
    name="user_behavior_features", 
    entities=["user_id"],
    ttl=timedelta(hours=6),  # Behavioral patterns change more frequently
    features=[
        # Transaction count features
        Feature(name="txn_count_1h", dtype=Int32),
        Feature(name="txn_count_6h", dtype=Int32), 
        Feature(name="txn_count_24h", dtype=Int32),
        Feature(name="txn_count_7d", dtype=Int32),
        Feature(name="txn_count_30d", dtype=Int32),
        
        # Transaction amount features
        Feature(name="txn_amount_sum_1h", dtype=Float64),
        Feature(name="txn_amount_sum_6h", dtype=Float64),
        Feature(name="txn_amount_sum_24h", dtype=Float64),
        Feature(name="txn_amount_sum_7d", dtype=Float64),
        Feature(name="txn_amount_sum_30d", dtype=Float64),
        
        Feature(name="txn_amount_avg_1h", dtype=Float64),
        Feature(name="txn_amount_avg_24h", dtype=Float64),
        Feature(name="txn_amount_avg_7d", dtype=Float64),
        
        Feature(name="txn_amount_max_1h", dtype=Float64),
        Feature(name="txn_amount_max_24h", dtype=Float64),
        Feature(name="txn_amount_max_7d", dtype=Float64),
        
        Feature(name="txn_amount_std_24h", dtype=Float64),
        Feature(name="txn_amount_std_7d", dtype=Float64),
        
        # Merchant diversity
        Feature(name="unique_merchants_24h", dtype=Int32),
        Feature(name="unique_merchants_7d", dtype=Int32),
        Feature(name="unique_merchants_30d", dtype=Int32),
        
        # Location diversity  
        Feature(name="unique_countries_24h", dtype=Int32),
        Feature(name="unique_countries_7d", dtype=Int32),
        Feature(name="unique_cities_24h", dtype=Int32),
        Feature(name="unique_cities_7d", dtype=Int32),
        
        # Payment method diversity
        Feature(name="unique_payment_methods_24h", dtype=Int32),
        Feature(name="unique_payment_methods_7d", dtype=Int32),
        
        # Device diversity
        Feature(name="unique_devices_24h", dtype=Int32),
        Feature(name="unique_devices_7d", dtype=Int32),
        
        # Time-based patterns
        Feature(name="weekend_txn_ratio_7d", dtype=Float32),
        Feature(name="night_txn_ratio_7d", dtype=Float32),
        Feature(name="business_hours_txn_ratio_7d", dtype=Float32),
        
        # Declined transaction patterns
        Feature(name="declined_txn_count_24h", dtype=Int32),
        Feature(name="declined_txn_count_7d", dtype=Int32),
        Feature(name="declined_txn_ratio_7d", dtype=Float32),
        
        # High-risk behavior
        Feature(name="high_risk_merchant_count_7d", dtype=Int32),
        Feature(name="international_txn_count_7d", dtype=Int32),
        Feature(name="crypto_txn_count_7d", dtype=Int32),
        
        # Velocity features
        Feature(name="avg_time_between_txns_24h", dtype=Float32),  # in minutes
        Feature(name="min_time_between_txns_24h", dtype=Float32),
        
        # Historical fraud indicators
        Feature(name="fraud_reports_30d", dtype=Int32),
        Feature(name="chargebacks_30d", dtype=Int32),
        Feature(name="disputed_amount_30d", dtype=Float64),
    ],
    online=True,
    source=user_behavior_source,
    tags={"team": "fraud_detection", "category": "user_behavior"}
)

# Transaction-level features - features specific to each transaction
transaction_features_fv = FeatureView(
    name="transaction_features",
    entities=["transaction_id"],
    ttl=timedelta(days=7),  # Keep transaction features for a week
    features=[
        Feature(name="user_id", dtype=Int64),
        Feature(name="merchant_id", dtype=String),
        Feature(name="device_id", dtype=String),
        Feature(name="amount", dtype=Float64),
        Feature(name="currency", dtype=String),
        Feature(name="payment_method", dtype=String),
        Feature(name="merchant_category", dtype=String),
        Feature(name="merchant_country", dtype=String),
        Feature(name="user_country", dtype=String),
        Feature(name="is_international", dtype=Bool),
        Feature(name="hour_of_day", dtype=Int32),
        Feature(name="day_of_week", dtype=Int32),
        Feature(name="is_weekend", dtype=Bool),
        Feature(name="is_holiday", dtype=Bool),
        Feature(name="is_business_hours", dtype=Bool),
        Feature(name="amount_usd", dtype=Float64),  # Normalized to USD
        Feature(name="exchange_rate", dtype=Float64),
        
        # Transaction context
        Feature(name="channel", dtype=String),  # online, mobile, atm, pos
        Feature(name="entry_method", dtype=String),  # chip, swipe, contactless, manual
        Feature(name="auth_method", dtype=String),  # pin, signature, biometric
        
        # Risk indicators
        Feature(name="is_high_risk_merchant", dtype=Bool),
        Feature(name="is_high_risk_country", dtype=Bool),
        Feature(name="is_new_device", dtype=Bool),
        Feature(name="is_new_merchant", dtype=Bool),
        Feature(name="is_round_amount", dtype=Bool),
        
        # ML model scores (if available from previous models)
        Feature(name="prev_fraud_score", dtype=Float32),
        Feature(name="anomaly_score", dtype=Float32),
    ],
    online=True,
    source=transaction_features_source,
    tags={"team": "fraud_detection", "category": "transaction"}
)

# Merchant features - characteristics of merchants
merchant_features_fv = FeatureView(
    name="merchant_features",
    entities=["merchant_id"],
    ttl=timedelta(days=7),  # Merchant stats updated weekly
    features=[
        Feature(name="merchant_category", dtype=String),
        Feature(name="merchant_country", dtype=String),
        Feature(name="merchant_age_days", dtype=Int32),
        Feature(name="merchant_risk_category", dtype=String),  # low, medium, high
        
        # Merchant volume statistics
        Feature(name="avg_transaction_amount_30d", dtype=Float64),
        Feature(name="total_volume_30d", dtype=Float64),
        Feature(name="transaction_count_30d", dtype=Int32),
        Feature(name="unique_customers_30d", dtype=Int32),
        
        # Merchant fraud statistics
        Feature(name="fraud_rate_30d", dtype=Float32),
        Feature(name="chargeback_rate_30d", dtype=Float32),
        Feature(name="dispute_rate_30d", dtype=Float32),
        
        # Merchant patterns
        Feature(name="peak_hour", dtype=Int32),
        Feature(name="operates_weekends", dtype=Bool),
        Feature(name="operates_24h", dtype=Bool),
        Feature(name="seasonal_business", dtype=Bool),
        
        # Risk indicators
        Feature(name="mcc_risk_score", dtype=Float32),  # Merchant Category Code risk
        Feature(name="processing_history_months", dtype=Int32),
        Feature(name="compliance_score", dtype=Float32),
    ],
    online=True,
    source=merchant_features_source,
    tags={"team": "fraud_detection", "category": "merchant"}
)

# Device features - characteristics of devices used for transactions
device_features_fv = FeatureView(
    name="device_features",
    entities=["device_id"],
    ttl=timedelta(days=1),  # Device features change daily
    features=[
        Feature(name="device_type", dtype=String),  # mobile, desktop, tablet, atm
        Feature(name="operating_system", dtype=String),
        Feature(name="browser", dtype=String),
        Feature(name="user_agent", dtype=String),
        Feature(name="screen_resolution", dtype=String),
        Feature(name="timezone", dtype=String),
        Feature(name="language", dtype=String),
        
        # Device usage patterns
        Feature(name="first_seen_days_ago", dtype=Int32),
        Feature(name="total_users", dtype=Int32),  # How many users use this device
        Feature(name="total_transactions_30d", dtype=Int32),
        Feature(name="unique_merchants_30d", dtype=Int32),
        Feature(name="unique_countries_30d", dtype=Int32),
        
        # Device risk indicators
        Feature(name="is_mobile", dtype=Bool),
        Feature(name="is_jailbroken", dtype=Bool),
        Feature(name="is_emulator", dtype=Bool),
        Feature(name="is_vpn", dtype=Bool),
        Feature(name="is_tor", dtype=Bool),
        Feature(name="device_reputation_score", dtype=Float32),
        
        # Location-based features
        Feature(name="country_changes_30d", dtype=Int32),
        Feature(name="city_changes_30d", dtype=Int32),
        Feature(name="suspicious_location_changes", dtype=Int32),
        
        # Behavioral patterns
        Feature(name="avg_session_duration", dtype=Float32),
        Feature(name="typing_pattern_score", dtype=Float32),
        Feature(name="mouse_movement_score", dtype=Float32),
        Feature(name="touch_pattern_score", dtype=Float32),
    ],
    online=True,
    source=device_features_source,
    tags={"team": "fraud_detection", "category": "device"}
)

# Real-time transaction features - computed in real-time for immediate scoring
realtime_transaction_fv = FeatureView(
    name="realtime_transaction_features",
    entities=["transaction_id"],
    ttl=timedelta(minutes=30),  # Very short TTL for real-time features
    features=[
        # Velocity features computed in real-time
        Feature(name="velocity_1min", dtype=Int32),
        Feature(name="velocity_5min", dtype=Int32),
        Feature(name="velocity_15min", dtype=Int32),
        
        Feature(name="amount_velocity_1min", dtype=Float64),
        Feature(name="amount_velocity_5min", dtype=Float64),
        Feature(name="amount_velocity_15min", dtype=Float64),
        
        # Real-time anomaly detection scores
        Feature(name="realtime_anomaly_score", dtype=Float32),
        Feature(name="amount_anomaly_score", dtype=Float32),
        Feature(name="location_anomaly_score", dtype=Float32),
        Feature(name="time_anomaly_score", dtype=Float32),
        
        # Pattern matching scores
        Feature(name="merchant_pattern_score", dtype=Float32),
        Feature(name="amount_pattern_score", dtype=Float32),
        Feature(name="time_pattern_score", dtype=Float32),
        
        # Network analysis features
        Feature(name="network_risk_score", dtype=Float32),
        Feature(name="connected_fraud_score", dtype=Float32),
        
        # External data enrichment (if available)
        Feature(name="blacklist_score", dtype=Float32),
        Feature(name="whitelist_score", dtype=Float32),
        Feature(name="third_party_risk_score", dtype=Float32),
    ],
    online=True,
    source=realtime_transaction_source,
    tags={"team": "fraud_detection", "category": "realtime"}
)

# Feature service for model serving
from feast import FeatureService

# Define feature services - collections of features for specific use cases

# Fraud detection feature service - comprehensive feature set for fraud scoring
fraud_detection_fs = FeatureService(
    name="fraud_detection_service",
    features=[
        user_profile_fv,
        user_behavior_fv[["txn_count_1h", "txn_count_24h", "txn_amount_sum_24h", 
                         "txn_amount_avg_24h", "unique_merchants_24h", 
                         "declined_txn_ratio_7d", "high_risk_merchant_count_7d"]],
        transaction_features_fv,
        merchant_features_fv[["merchant_category", "merchant_risk_category", 
                             "fraud_rate_30d", "chargeback_rate_30d"]],
        device_features_fv[["device_type", "is_mobile", "is_vpn", 
                           "device_reputation_score", "country_changes_30d"]],
    ],
    tags={"use_case": "fraud_detection"}
)

# Real-time fraud scoring feature service - optimized for low latency
realtime_fraud_fs = FeatureService(
    name="realtime_fraud_service", 
    features=[
        user_profile_fv[["risk_score", "account_type", "user_segment"]],
        user_behavior_fv[["txn_count_1h", "txn_amount_sum_1h", 
                         "declined_txn_count_24h", "high_risk_merchant_count_7d"]],
        transaction_features_fv[["amount", "is_international", "is_high_risk_merchant",
                               "is_new_device", "hour_of_day", "is_weekend"]],
        merchant_features_fv[["merchant_risk_category", "fraud_rate_30d"]],
        device_features_fv[["is_vpn", "device_reputation_score"]],
        realtime_transaction_fv,
    ],
    tags={"use_case": "realtime_fraud", "latency": "low"}
)

# Model training feature service - comprehensive features for training
training_fs = FeatureService(
    name="fraud_training_service",
    features=[
        user_profile_fv,
        user_behavior_fv, 
        transaction_features_fv,
        merchant_features_fv,
        device_features_fv,
    ],
    tags={"use_case": "training"}
)

# Batch scoring feature service - for batch inference jobs
batch_scoring_fs = FeatureService(
    name="batch_fraud_service",
    features=[
        user_profile_fv,
        user_behavior_fv,
        transaction_features_fv,
        merchant_features_fv,
        device_features_fv,
    ],
    tags={"use_case": "batch_scoring"}
)

# List all feature views and services for easy import
all_feature_views = [
    user_profile_fv,
    user_behavior_fv,
    transaction_features_fv,
    merchant_features_fv,
    device_features_fv,
    realtime_transaction_fv,
]

all_entities = [
    user_entity,
    transaction_entity,
    merchant_entity,
    device_entity,
]

all_feature_services = [
    fraud_detection_fs,
    realtime_fraud_fs,
    training_fs,
    batch_scoring_fs,
]
