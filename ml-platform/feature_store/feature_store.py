  """
Feast feature store configuration and utilities for fraud detection
"""

import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging

from feast import FeatureStore, RepoConfig
from feast.data_source import DataSource
from feast.infra.offline_stores.file import FileOfflineStoreConfig
from feast.infra.online_stores.redis import RedisOnlineStoreConfig
from feast.repo_config import RegistryConfig

from feature_definitions import (
    all_feature_views, all_entities, all_feature_services,
    fraud_detection_fs, realtime_fraud_fs, training_fs, batch_scoring_fs
)

logger = logging.getLogger(__name__)


class FraudFeatureStore:
    """
    Fraud detection feature store implementation using Feast
    """
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._create_default_config()
        self.store = None
        self._initialize_store()
    
    def _create_default_config(self) -> str:
        """Create default feature store configuration"""
        
        config = {
            'project': 'fraud_detection',
            'registry': {
                'path': 'feature_store/data/registry.db',
                'cache_ttl_seconds': 60
            },
            'provider': 'local',
            'online_store': {
                'type': 'redis',
                'connection_string': 'redis://localhost:6379'
            },
            'offline_store': {
                'type': 'file'
            },
            'entity_key_serialization_version': 2
        }
        
        config_dir = 'feature_store'
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, 'feature_store.yaml')
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config_path
    
    def _initialize_store(self):
        """Initialize the Feast feature store"""
        try:
            # Change to feature store directory for Feast to work properly
            original_dir = os.getcwd()
            feature_store_dir = os.path.dirname(self.config_path)
            if feature_store_dir:
                os.chdir(feature_store_dir)
            
            self.store = FeatureStore(repo_path=".")
            
            # Change back to original directory
            os.chdir(original_dir)
            
            logger.info("Feature store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize feature store: {e}")
            raise
    
    def setup_feature_store(self):
        """Setup the feature store with all feature definitions"""
        try:
            logger.info("Setting up feature store...")
            
            # Create data directories
            data_dir = os.path.join(os.path.dirname(self.config_path), 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            # Generate sample data for demonstration
            self._generate_sample_data(data_dir)
            
            # Apply feature definitions to the store
            original_dir = os.getcwd()
            feature_store_dir = os.path.dirname(self.config_path)
            if feature_store_dir:
                os.chdir(feature_store_dir)
            
            self.store.apply(all_entities + all_feature_views + all_feature_services)
            
            os.chdir(original_dir)
            
            logger.info("Feature store setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup feature store: {e}")
            raise
    
    def _generate_sample_data(self, data_dir: str):
        """Generate sample data for feature store demonstration"""
        logger.info("Generating sample data...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate user profile data
        n_users = 1000
        user_data = {
            'user_id': range(1, n_users + 1),
            'user_age': np.random.randint(18, 80, n_users),
            'account_age_days': np.random.randint(30, 2000, n_users),
            'user_country': np.random.choice(['US', 'CA', 'UK', 'FR', 'DE', 'AU'], n_users),
            'user_state': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL'], n_users),
            'user_city': np.random.choice(['San Francisco', 'New York', 'London', 'Paris'], n_users),
            'account_type': np.random.choice(['personal', 'business'], n_users, p=[0.8, 0.2]),
            'kyc_status': np.random.choice(['verified', 'pending', 'failed'], n_users, p=[0.85, 0.1, 0.05]),
            'risk_score': np.random.normal(0.3, 0.2, n_users).clip(0, 1),
            'account_balance': np.random.lognormal(8, 1.5, n_users),
            'credit_limit': np.random.lognormal(9, 1, n_users),
            'is_premium_user': np.random.choice([True, False], n_users, p=[0.2, 0.8]),
            'user_segment': np.random.choice(['high_value', 'regular', 'new'], n_users, p=[0.1, 0.7, 0.2]),
            'event_timestamp': [datetime.now() - timedelta(minutes=np.random.randint(0, 1440)) for _ in range(n_users)],
            'created_timestamp': [datetime.now()] * n_users
        }
        
        user_df = pd.DataFrame(user_data)
        user_df.to_parquet(os.path.join(data_dir, 'user_profiles.parquet'), index=False)
        
        # Generate user behavior aggregations
        user_behavior_data = {
            'user_id': range(1, n_users + 1),
            'txn_count_1h': np.random.poisson(2, n_users),
            'txn_count_6h': np.random.poisson(8, n_users),
            'txn_count_24h': np.random.poisson(15, n_users),
            'txn_count_7d': np.random.poisson(50, n_users),
            'txn_count_30d': np.random.poisson(200, n_users),
            'txn_amount_sum_1h': np.random.lognormal(4, 1, n_users),
            'txn_amount_sum_6h': np.random.lognormal(5, 1, n_users),
            'txn_amount_sum_24h': np.random.lognormal(6, 1, n_users),
            'txn_amount_sum_7d': np.random.lognormal(7, 1, n_users),
            'txn_amount_sum_30d': np.random.lognormal(8, 1, n_users),
            'txn_amount_avg_1h': np.random.lognormal(3, 0.5, n_users),
            'txn_amount_avg_24h': np.random.lognormal(3.5, 0.5, n_users),
            'txn_amount_avg_7d': np.random.lognormal(3.8, 0.5, n_users),
            'txn_amount_max_1h': np.random.lognormal(4, 1, n_users),
            'txn_amount_max_24h': np.random.lognormal(5, 1, n_users),
            'txn_amount_max_7d': np.random.lognormal(6, 1, n_users),
            'txn_amount_std_24h': np.random.lognormal(2, 0.5, n_users),
            'txn_amount_std_7d': np.random.lognormal(2.5, 0.5, n_users),
            'unique_merchants_24h': np.random.randint(1, 10, n_users),
            'unique_merchants_7d': np.random.randint(3, 30, n_users),
            'unique_merchants_30d': np.random.randint(5, 100, n_users),
            'unique_countries_24h': np.random.randint(1, 3, n_users),
            'unique_countries_7d': np.random.randint(1, 5, n_users),
            'unique_cities_24h': np.random.randint(1, 4, n_users),
            'unique_cities_7d': np.random.randint(1, 8, n_users),
            'unique_payment_methods_24h': np.random.randint(1, 3, n_users),
            'unique_payment_methods_7d': np.random.randint(1, 4, n_users),
            'unique_devices_24h': np.random.randint(1, 2, n_users),
            'unique_devices_7d': np.random.randint(1, 3, n_users),
            'weekend_txn_ratio_7d': np.random.beta(2, 5, n_users),
            'night_txn_ratio_7d': np.random.beta(1, 9, n_users),
            'business_hours_txn_ratio_7d': np.random.beta(7, 3, n_users),
            'declined_txn_count_24h': np.random.poisson(1, n_users),
            'declined_txn_count_7d': np.random.poisson(3, n_users),
            'declined_txn_ratio_7d': np.random.beta(1, 19, n_users),
            'high_risk_merchant_count_7d': np.random.poisson(0.5, n_users),
            'international_txn_count_7d': np.random.poisson(2, n_users),
            'crypto_txn_count_7d': np.random.poisson(0.1, n_users),
            'avg_time_between_txns_24h': np.random.lognormal(4, 1, n_users),
            'min_time_between_txns_24h': np.random.lognormal(2, 1, n_users),
            'fraud_reports_30d': np.random.poisson(0.1, n_users),
            'chargebacks_30d': np.random.poisson(0.05, n_users),
            'disputed_amount_30d': np.random.lognormal(3, 2, n_users) * np.random.binomial(1, 0.02, n_users),
            'event_timestamp': [datetime.now() - timedelta(minutes=np.random.randint(0, 360)) for _ in range(n_users)],
            'created_timestamp': [datetime.now()] * n_users
        }
        
        behavior_df = pd.DataFrame(user_behavior_data)
        behavior_df.to_parquet(os.path.join(data_dir, 'user_behavior_aggregations.parquet'), index=False)
        
        # Generate transaction features
        n_transactions = 5000
        transaction_data = {
            'transaction_id': [f'txn_{i:06d}' for i in range(1, n_transactions + 1)],
            'user_id': np.random.randint(1, n_users + 1, n_transactions),
            'merchant_id': [f'merchant_{i:04d}' for i in np.random.randint(1, 500, n_transactions)],
            'device_id': [f'device_{i:06d}' for i in np.random.randint(1, 2000, n_transactions)],
            'amount': np.random.lognormal(4, 1.5, n_transactions),
            'currency': np.random.choice(['USD', 'EUR', 'GBP', 'CAD'], n_transactions, p=[0.6, 0.2, 0.1, 0.1]),
            'payment_method': np.random.choice(['credit_card', 'debit_card', 'bank_transfer', 'digital_wallet'], n_transactions),
            'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail', 'online', 'gambling'], n_transactions),
            'merchant_country': np.random.choice(['US', 'CA', 'UK', 'FR', 'DE'], n_transactions),
            'user_country': np.random.choice(['US', 'CA', 'UK', 'FR', 'DE'], n_transactions),
            'is_international': np.random.choice([True, False], n_transactions, p=[0.15, 0.85]),
            'hour_of_day': np.random.randint(0, 24, n_transactions),
            'day_of_week': np.random.randint(0, 7, n_transactions),
            'is_weekend': np.random.choice([True, False], n_transactions, p=[0.2, 0.8]),
            'is_holiday': np.random.choice([True, False], n_transactions, p=[0.05, 0.95]),
            'is_business_hours': np.random.choice([True, False], n_transactions, p=[0.7, 0.3]),
            'amount_usd': np.random.lognormal(4, 1.5, n_transactions),
            'exchange_rate': np.random.normal(1.0, 0.1, n_transactions).clip(0.5, 2.0),
            'channel': np.random.choice(['online', 'mobile', 'atm', 'pos'], n_transactions),
            'entry_method': np.random.choice(['chip', 'swipe', 'contactless', 'manual'], n_transactions),
            'auth_method': np.random.choice(['pin', 'signature', 'biometric'], n_transactions),
            'is_high_risk_merchant': np.random.choice([True, False], n_transactions, p=[0.1, 0.9]),
            'is_high_risk_country': np.random.choice([True, False], n_transactions, p=[0.05, 0.95]),
            'is_new_device': np.random.choice([True, False], n_transactions, p=[0.1, 0.9]),
            'is_new_merchant': np.random.choice([True, False], n_transactions, p=[0.15, 0.85]),
            'is_round_amount': np.random.choice([True, False], n_transactions, p=[0.2, 0.8]),
            'prev_fraud_score': np.random.beta(2, 8, n_transactions),
            'anomaly_score': np.random.beta(1, 9, n_transactions),
            'event_timestamp': [datetime.now() - timedelta(minutes=np.random.randint(0, 1440)) for _ in range(n_transactions)],
            'created_timestamp': [datetime.now()] * n_transactions
        }
        
        transaction_df = pd.DataFrame(transaction_data)
        transaction_df.to_parquet(os.path.join(data_dir, 'transaction_features.parquet'), index=False)
        
        # Generate merchant features
        n_merchants = 500
        merchant_data = {
            'merchant_id': [f'merchant_{i:04d}' for i in range(1, n_merchants + 1)],
            'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail', 'online', 'gambling'], n_merchants),
            'merchant_country': np.random.choice(['US', 'CA', 'UK', 'FR', 'DE'], n_merchants),
            'merchant_age_days': np.random.randint(30, 3000, n_merchants),
            'merchant_risk_category': np.random.choice(['low', 'medium', 'high'], n_merchants, p=[0.7, 0.25, 0.05]),
            'avg_transaction_amount_30d': np.random.lognormal(4, 1, n_merchants),
            'total_volume_30d': np.random.lognormal(10, 2, n_merchants),
            'transaction_count_30d': np.random.poisson(1000, n_merchants),
            'unique_customers_30d': np.random.poisson(200, n_merchants),
            'fraud_rate_30d': np.random.beta(1, 49, n_merchants),
            'chargeback_rate_30d': np.random.beta(1, 99, n_merchants),
            'dispute_rate_30d': np.random.beta(1, 199, n_merchants),
            'peak_hour': np.random.randint(9, 18, n_merchants),
            'operates_weekends': np.random.choice([True, False], n_merchants, p=[0.6, 0.4]),
            'operates_24h': np.random.choice([True, False], n_merchants, p=[0.2, 0.8]),
            'seasonal_business': np.random.choice([True, False], n_merchants, p=[0.3, 0.7]),
            'mcc_risk_score': np.random.beta(2, 8, n_merchants),
            'processing_history_months': np.random.randint(6, 120, n_merchants),
            'compliance_score': np.random.beta(8, 2, n_merchants),
            'event_timestamp': [datetime.now() - timedelta(hours=np.random.randint(0, 168)) for _ in range(n_merchants)],
            'created_timestamp': [datetime.now()] * n_merchants
        }
        
        merchant_df = pd.DataFrame(merchant_data)
        merchant_df.to_parquet(os.path.join(data_dir, 'merchant_features.parquet'), index=False)
        
        # Generate device features
        n_devices = 2000
        device_data = {
            'device_id': [f'device_{i:06d}' for i in range(1, n_devices + 1)],
            'device_type': np.random.choice(['mobile', 'desktop', 'tablet', 'atm'], n_devices, p=[0.6, 0.3, 0.08, 0.02]),
            'operating_system': np.random.choice(['iOS', 'Android', 'Windows', 'MacOS', 'Linux'], n_devices),
            'browser': np.random.choice(['Chrome', 'Safari', 'Firefox', 'Edge', 'Other'], n_devices),
            'user_agent': [f'UserAgent_{i}' for i in range(n_devices)],
            'screen_resolution': np.random.choice(['1920x1080', '1366x768', '1280x720', '2560x1440'], n_devices),
            'timezone': np.random.choice(['PST', 'EST', 'GMT', 'CET'], n_devices),
            'language': np.random.choice(['en', 'es', 'fr', 'de', 'zh'], n_devices),
            'first_seen_days_ago': np.random.randint(1, 365, n_devices),
            'total_users': np.random.randint(1, 5, n_devices),
            'total_transactions_30d': np.random.poisson(50, n_devices),
            'unique_merchants_30d': np.random.randint(1, 20, n_devices),
            'unique_countries_30d': np.random.randint(1, 3, n_devices),
            'is_mobile': np.random.choice([True, False], n_devices, p=[0.6, 0.4]),
            'is_jailbroken': np.random.choice([True, False], n_devices, p=[0.02, 0.98]),
            'is_emulator': np.random.choice([True, False], n_devices, p=[0.01, 0.99]),
            'is_vpn': np.random.choice([True, False], n_devices, p=[0.05, 0.95]),
            'is_tor': np.random.choice([True, False], n_devices, p=[0.001, 0.999]),
            'device_reputation_score': np.random.beta(8, 2, n_devices),
            'country_changes_30d': np.random.poisson(0.5, n_devices),
            'city_changes_30d': np.random.poisson(2, n_devices),
            'suspicious_location_changes': np.random.poisson(0.1, n_devices),
            'avg_session_duration': np.random.lognormal(4, 1, n_devices),
            'typing_pattern_score': np.random.beta(5, 5, n_devices),
            'mouse_movement_score': np.random.beta(5, 5, n_devices),
            'touch_pattern_score': np.random.beta(5, 5, n_devices),
            'event_timestamp': [datetime.now() - timedelta(hours=np.random.randint(0, 24)) for _ in range(n_devices)],
            'created_timestamp': [datetime.now()] * n_devices
        }
        
        device_df = pd.DataFrame(device_data)
        device_df.to_parquet(os.path.join(data_dir, 'device_features.parquet'), index=False)
        
        # Generate real-time transaction features
        realtime_data = {
            'transaction_id': [f'txn_{i:06d}' for i in range(1, n_transactions + 1)],
            'velocity_1min': np.random.poisson(1, n_transactions),
            'velocity_5min': np.random.poisson(3, n_transactions),
            'velocity_15min': np.random.poisson(8, n_transactions),
            'amount_velocity_1min': np.random.lognormal(3, 1, n_transactions),
            'amount_velocity_5min': np.random.lognormal(4, 1, n_transactions),
            'amount_velocity_15min': np.random.lognormal(5, 1, n_transactions),
            'realtime_anomaly_score': np.random.beta(1, 9, n_transactions),
            'amount_anomaly_score': np.random.beta(2, 8, n_transactions),
            'location_anomaly_score': np.random.beta(1, 19, n_transactions),
            'time_anomaly_score': np.random.beta(1, 9, n_transactions),
            'merchant_pattern_score': np.random.beta(5, 5, n_transactions),
            'amount_pattern_score': np.random.beta(5, 5, n_transactions),
            'time_pattern_score': np.random.beta(5, 5, n_transactions),
            'network_risk_score': np.random.beta(2, 8, n_transactions),
            'connected_fraud_score': np.random.beta(1, 19, n_transactions),
            'blacklist_score': np.random.beta(1, 99, n_transactions),
            'whitelist_score': np.random.beta(9, 1, n_transactions),
            'third_party_risk_score': np.random.beta(3, 7, n_transactions),
            'event_timestamp': [datetime.now() - timedelta(minutes=np.random.randint(0, 30)) for _ in range(n_transactions)],
            'created_timestamp': [datetime.now()] * n_transactions
        }
        
        realtime_df = pd.DataFrame(realtime_data)
        realtime_df.to_parquet(os.path.join(data_dir, 'realtime_transactions.parquet'), index=False)
        
        logger.info("Sample data generation completed")
    
    def get_training_features(self, entity_df: pd.DataFrame, feature_service_name: str = "fraud_training_service") -> pd.DataFrame:
        """Get features for model training"""
        try:
            logger.info(f"Retrieving training features using service: {feature_service_name}")
            
            # Get historical features
            training_df = self.store.get_historical_features(
                entity_df=entity_df,
                features=self.store.get_feature_service(feature_service_name)
            ).to_df()
            
            logger.info(f"Retrieved {len(training_df)} training samples with {len(training_df.columns)} features")
            return training_df
            
        except Exception as e:
            logger.error(f"Failed to get training features: {e}")
            raise
    
    def get_online_features(self, entity_rows: List[Dict], feature_service_name: str = "realtime_fraud_service") -> Dict:
        """Get features for real-time inference"""
        try:
            logger.info(f"Retrieving online features for {len(entity_rows)} entities")
            
            # Get online features
            online_features = self.store.get_online_features(
                features=self.store.get_feature_service(feature_service_name),
                entity_rows=entity_rows
            ).to_dict()
            
            logger.info(f"Retrieved online features with {len(online_features)} feature arrays")
            return online_features
            
        except Exception as e:
            logger.error(f"Failed to get online features: {e}")
            raise
    
    def materialize_features(self, start_date: datetime, end_date: datetime):
        """Materialize features to the online store"""
        try:
            logger.info(f"Materializing features from {start_date} to {end_date}")
            
            # Materialize features to online store
            self.store.materialize(start_date, end_date)
            
            logger.info("Feature materialization completed")
            
        except Exception as e:
            logger.error(f"Failed to materialize features: {e}")
            raise
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get statistics about the feature store"""
        try:
            stats = {
                'feature_views': len(all_feature_views),
                'entities': len(all_entities),
                'feature_services': len(all_feature_services),
                'total_features': sum(len(fv.features) for fv in all_feature_views)
            }
            
            # Get feature counts by category
            feature_counts = {}
            for fv in all_feature_views:
                category = fv.tags.get('category', 'unknown')
                feature_counts[category] = feature_counts.get(category, 0) + len(fv.features)
            
            stats['features_by_category'] = feature_counts
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get feature statistics: {e}")
            return {}


def create_feature_store_config() -> str:
    """Create a production-ready feature store configuration"""
    
    config = {
        'project': 'fraud_detection_prod',
        'registry': {
            'registry_type': 'sql',
            'path': 'postgresql://feast:feast@localhost:5432/feast',
            'cache_ttl_seconds': 60
        },
        'provider': 'aws',  # or 'gcp', 'azure'
        'online_store': {
            'type': 'redis',
            'connection_string': 'redis://redis-cluster.feast.svc.cluster.local:6379',
            'ssl': True
        },
        'offline_store': {
            'type': 'snowflake',  # or 'bigquery', 'redshift'
            'config': {
                'account': 'your_account',
                'user': 'feast_user',
                'password': 'feast_password',
                'role': 'FEAST_ROLE',
                'warehouse': 'FEAST_WH',
                'database': 'FEAST_DB'
            }
        },
        'batch_engine': {
            'type': 'spark',
            'config': {
                'spark.master': 'k8s://https://kubernetes.default.svc:443',
                'spark.executor.instances': '10',
                'spark.executor.cores': '4',
                'spark.executor.memory': '8g'
            }
        },
        'entity_key_serialization_version': 2,
        'flags': {
            'alpha_features': True,
            'go_feature_retrieval': True
        }
    }
    
    config_path = 'feature_store/feature_store_prod.yaml'
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize feature store
    feature_store = FraudFeatureStore()
    
    # Setup feature store with sample data
    feature_store.setup_feature_store()
    
    # Materialize features for the last day
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    feature_store.materialize_features(start_date, end_date)
    
    # Example: Get training features
    entity_df = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'transaction_id': ['txn_000001', 'txn_000002', 'txn_000003', 'txn_000004', 'txn_000005'],
        'event_timestamp': [datetime.now() - timedelta(hours=i) for i in range(5)]
    })
    
    training_features = feature_store.get_training_features(entity_df)
    print(f"Training features shape: {training_features.shape}")
    print(f"Training features columns: {list(training_features.columns)[:10]}...")
    
    # Example: Get online features for real-time prediction
    entity_rows = [
        {'user_id': 1, 'transaction_id': 'txn_000001', 'merchant_id': 'merchant_0001', 'device_id': 'device_000001'},
        {'user_id': 2, 'transaction_id': 'txn_000002', 'merchant_id': 'merchant_0002', 'device_id': 'device_000002'}
    ]
    
    online_features = feature_store.get_online_features(entity_rows)
    print(f"Online features retrieved for {len(entity_rows)} entities")
    
    # Get feature store statistics
    stats = feature_store.get_feature_statistics()
    print(f"Feature store statistics: {stats}")
    
    print("Feature store setup and testing completed!")
