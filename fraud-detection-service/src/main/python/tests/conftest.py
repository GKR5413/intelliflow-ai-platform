"""
Pytest configuration and fixtures for fraud detection service tests
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta
import redis
import tempfile
import os

# Test fixtures


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    mock_redis = Mock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.setex.return_value = True
    mock_redis.exists.return_value = False
    mock_redis.ping.return_value = True
    return mock_redis


@pytest.fixture
def mock_database():
    """Mock database connection"""
    mock_db = AsyncMock()
    mock_db.fetch.return_value = []
    mock_db.fetchrow.return_value = None
    mock_db.execute.return_value = None
    return mock_db


@pytest.fixture
def sample_transaction_data():
    """Sample transaction data for testing"""
    return {
        "transaction_id": "txn_test_001",
        "user_id": 12345,
        "amount": 99.99,
        "currency": "USD",
        "payment_method": "credit_card",
        "merchant_id": "merchant_123",
        "merchant_category": "grocery",
        "user_country": "US",
        "merchant_country": "US",
        "ip_address": "192.168.1.1",
        "device_fingerprint": "device_123",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "timestamp": datetime.now().isoformat(),
        "location_data": "San Francisco, CA"
    }


@pytest.fixture
def sample_features_dataframe():
    """Sample features DataFrame for testing"""
    data = {
        "amount": [99.99, 150.00, 25.50],
        "amount_log": [4.605, 5.011, 3.238],
        "amount_sqrt": [9.999, 12.247, 5.050],
        "is_weekend": [0, 1, 0],
        "is_night": [0, 0, 1],
        "is_international": [0, 0, 1],
        "user_velocity_1h": [1, 2, 1],
        "user_velocity_24h": [5, 8, 3],
        "merchant_risk_score": [0.1, 0.3, 0.8],
        "device_consistency": [1, 1, 0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_ml_model():
    """Mock machine learning model"""
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array([[0.85, 0.15], [0.25, 0.75], [0.95, 0.05]])
    mock_model.predict.return_value = np.array([0, 1, 0])
    mock_model.feature_names_in_ = ["amount", "amount_log", "amount_sqrt", "is_weekend", 
                                   "is_night", "is_international", "user_velocity_1h", 
                                   "user_velocity_24h", "merchant_risk_score", "device_consistency"]
    return mock_model


@pytest.fixture
def temporary_model_file():
    """Create a temporary model file for testing"""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    
    # Create a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
        joblib.dump(model, f.name)
        yield f.name
    
    # Clean up
    os.unlink(f.name)


@pytest.fixture
def mock_feature_store():
    """Mock feature store client"""
    mock_store = Mock()
    mock_store.get_online_features.return_value = {
        "user_velocity_1h": [1, 2, 1],
        "user_velocity_24h": [5, 8, 3],
        "user_avg_amount_7d": [75.50, 120.00, 45.25],
        "merchant_risk_score": [0.1, 0.3, 0.8],
        "device_consistency": [1, 1, 0]
    }
    return mock_store


@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        "model_path": "/tmp/test_model.joblib",
        "feature_cache_ttl": 300,
        "batch_size": 1000,
        "redis_url": "redis://localhost:6379/1",
        "database_url": "postgresql://test:test@localhost:5432/test_fraud_db",
        "log_level": "DEBUG",
        "monitoring_enabled": False
    }


@pytest.fixture
def performance_test_data():
    """Generate performance test data"""
    np.random.seed(42)
    n_samples = 1000
    
    data = []
    for i in range(n_samples):
        transaction = {
            "transaction_id": f"txn_perf_{i:06d}",
            "user_id": np.random.randint(1, 10000),
            "amount": np.random.exponential(50.0),
            "currency": np.random.choice(["USD", "EUR", "GBP"], p=[0.7, 0.2, 0.1]),
            "payment_method": np.random.choice(["credit_card", "debit_card", "paypal", "bank_transfer"]),
            "merchant_category": np.random.choice(["grocery", "gas_station", "restaurant", "online", "retail"]),
            "user_country": np.random.choice(["US", "GB", "CA", "DE", "FR"]),
            "merchant_country": np.random.choice(["US", "GB", "CA", "DE", "FR"]),
            "timestamp": (datetime.now() - timedelta(seconds=np.random.randint(0, 86400))).isoformat()
        }
        data.append(transaction)
    
    return data


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Cleanup test files after each test"""
    yield
    # Cleanup any test files created during tests
    test_files = [
        "/tmp/test_model.joblib",
        "/tmp/test_features.parquet",
        "/tmp/test_predictions.json"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            os.unlink(file_path)


# Test configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


# Test collection modifiers
def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their names"""
    for item in items:
        # Mark slow tests
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.integration)
        
        # Mark performance tests
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.performance)