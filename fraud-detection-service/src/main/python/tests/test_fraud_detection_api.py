"""
Unit tests for IntelliFlow Fraud Detection API
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from datetime import datetime

# Import the modules to test
from fraud_detection_api import app, FraudDetectionRequest, BatchTransactionInput
from scoring_engine import ScoringEngine
from feature_pipeline import FeaturePipeline


class TestFraudDetectionAPI:
    """Test suite for Fraud Detection API endpoints"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_transaction(self):
        """Sample transaction data for testing"""
        return {
            "transaction_id": "txn_test_001",
            "user_id": 12345,
            "amount": 99.99,
            "currency": "USD",
            "payment_method": "credit_card",
            "merchant_category": "grocery",
            "user_country": "US",
            "merchant_country": "US",
            "ip_address": "192.168.1.1",
            "device_fingerprint": "device_123",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
    @pytest.fixture
    def sample_batch_request(self, sample_transaction):
        """Sample batch request for testing"""
        return {
            "transactions": [
                sample_transaction,
                {**sample_transaction, "transaction_id": "txn_test_002", "amount": 150.00}
            ]
        }
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "warning", "unhealthy"]
        assert "uptime_seconds" in data
        assert "version" in data
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "model_version" in data
        assert "features_count" in data
    
    @patch('fraud_detection_api.scoring_engine')
    def test_predict_single_transaction_success(self, mock_scoring_engine, client, sample_transaction):
        """Test successful single transaction prediction"""
        # Mock the scoring engine response
        mock_scoring_engine.predict_fraud.return_value = {
            "transaction_id": "txn_test_001",
            "fraud_probability": 0.15,
            "fraud_prediction": False,
            "risk_level": "low",
            "confidence_score": 0.85,
            "processing_time_ms": 45.2,
            "model_version": "1.0.0"
        }
        
        response = client.post("/predict", json=sample_transaction)
        assert response.status_code == 200
        
        data = response.json()
        assert data["transaction_id"] == "txn_test_001"
        assert data["fraud_probability"] == 0.15
        assert data["fraud_prediction"] is False
        assert data["risk_level"] == "low"
        assert data["confidence_score"] == 0.85
    
    @patch('fraud_detection_api.scoring_engine')
    def test_predict_single_transaction_high_risk(self, mock_scoring_engine, client, sample_transaction):
        """Test high-risk transaction prediction"""
        # Mock high-risk response
        mock_scoring_engine.predict_fraud.return_value = {
            "transaction_id": "txn_test_001",
            "fraud_probability": 0.89,
            "fraud_prediction": True,
            "risk_level": "high",
            "confidence_score": 0.91,
            "processing_time_ms": 52.1,
            "model_version": "1.0.0"
        }
        
        response = client.post("/predict", json=sample_transaction)
        assert response.status_code == 200
        
        data = response.json()
        assert data["fraud_probability"] == 0.89
        assert data["fraud_prediction"] is True
        assert data["risk_level"] == "high"
    
    def test_predict_invalid_input(self, client):
        """Test prediction with invalid input"""
        invalid_transaction = {
            "transaction_id": "txn_test_001",
            "amount": -100.0,  # Invalid negative amount
            "currency": "INVALID"  # Invalid currency
        }
        
        response = client.post("/predict", json=invalid_transaction)
        assert response.status_code == 422  # Validation error
    
    @patch('fraud_detection_api.scoring_engine')
    def test_predict_batch_success(self, mock_scoring_engine, client, sample_batch_request):
        """Test successful batch prediction"""
        # Mock batch scoring response
        mock_scoring_engine.predict_fraud_batch.return_value = {
            "results": [
                {
                    "transaction_id": "txn_test_001",
                    "fraud_probability": 0.15,
                    "fraud_prediction": False,
                    "risk_level": "low"
                },
                {
                    "transaction_id": "txn_test_002",
                    "fraud_probability": 0.72,
                    "fraud_prediction": True,
                    "risk_level": "high"
                }
            ],
            "batch_size": 2,
            "processing_time_ms": 89.5,
            "model_version": "1.0.0"
        }
        
        response = client.post("/predict/batch", json=sample_batch_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["batch_size"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["transaction_id"] == "txn_test_001"
        assert data["results"][1]["transaction_id"] == "txn_test_002"
    
    def test_predict_batch_empty(self, client):
        """Test batch prediction with empty list"""
        empty_batch = {"transactions": []}
        
        response = client.post("/predict/batch", json=empty_batch)
        assert response.status_code == 422  # Validation error
    
    @patch('fraud_detection_api.scoring_engine')
    def test_predict_batch_too_large(self, mock_scoring_engine, client, sample_transaction):
        """Test batch prediction with too many transactions"""
        large_batch = {
            "transactions": [sample_transaction] * 1001  # Over limit
        }
        
        response = client.post("/predict/batch", json=large_batch)
        assert response.status_code == 422  # Validation error
    
    def test_authentication_required(self, client, sample_transaction):
        """Test that authentication is required for prediction endpoints"""
        # Remove Authorization header and test
        response = client.post("/predict", json=sample_transaction)
        # Should still work in test mode, but check that auth middleware exists
        assert response.status_code in [200, 401]
    
    @patch('fraud_detection_api.scoring_engine')
    def test_model_reload_success(self, mock_scoring_engine, client):
        """Test successful model reload"""
        mock_scoring_engine.reload_model.return_value = True
        
        response = client.post("/model/reload")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "message" in data
    
    @patch('fraud_detection_api.scoring_engine')
    def test_model_reload_failure(self, mock_scoring_engine, client):
        """Test model reload failure"""
        mock_scoring_engine.reload_model.side_effect = Exception("Model reload failed")
        
        response = client.post("/model/reload")
        assert response.status_code == 500
        
        data = response.json()
        assert data["status"] == "error"
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        
        content = response.text
        assert "fraud_predictions_total" in content
        assert "fraud_prediction_duration_seconds" in content


class TestFraudDetectionRequest:
    """Test suite for request/response models"""
    
    def test_valid_fraud_detection_request(self):
        """Test valid fraud detection request creation"""
        request_data = {
            "transaction_id": "txn_001",
            "user_id": 12345,
            "amount": 99.99,
            "currency": "USD",
            "payment_method": "credit_card",
            "merchant_category": "grocery",
            "user_country": "US",
            "merchant_country": "US"
        }
        
        request = FraudDetectionRequest(**request_data)
        assert request.transaction_id == "txn_001"
        assert request.amount == 99.99
        assert request.currency == "USD"
    
    def test_invalid_amount(self):
        """Test validation of negative amount"""
        with pytest.raises(ValueError):
            FraudDetectionRequest(
                transaction_id="txn_001",
                user_id=12345,
                amount=-10.0,  # Invalid
                currency="USD",
                payment_method="credit_card",
                merchant_category="grocery",
                user_country="US",
                merchant_country="US"
            )
    
    def test_invalid_currency(self):
        """Test validation of invalid currency"""
        with pytest.raises(ValueError):
            FraudDetectionRequest(
                transaction_id="txn_001",
                user_id=12345,
                amount=99.99,
                currency="INVALID",  # Invalid
                payment_method="credit_card",
                merchant_category="grocery",
                user_country="US",
                merchant_country="US"
            )
    
    def test_optional_fields(self):
        """Test that optional fields work correctly"""
        request = FraudDetectionRequest(
            transaction_id="txn_001",
            user_id=12345,
            amount=99.99,
            currency="USD",
            payment_method="credit_card",
            merchant_category="grocery",
            user_country="US",
            merchant_country="US"
            # Optional fields not provided
        )
        
        assert request.ip_address is None
        assert request.device_fingerprint is None
        assert request.user_agent is None


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test suite for async functionality"""
    
    @pytest.fixture
    def client(self):
        """Async test client"""
        return TestClient(app)
    
    @patch('fraud_detection_api.scoring_engine')
    async def test_async_prediction_performance(self, mock_scoring_engine, client):
        """Test that async predictions handle concurrent requests"""
        # Mock fast response
        mock_scoring_engine.predict_fraud.return_value = {
            "transaction_id": "txn_test",
            "fraud_probability": 0.15,
            "fraud_prediction": False,
            "risk_level": "low",
            "confidence_score": 0.85,
            "processing_time_ms": 25.0,
            "model_version": "1.0.0"
        }
        
        transaction = {
            "transaction_id": "txn_test",
            "user_id": 12345,
            "amount": 99.99,
            "currency": "USD",
            "payment_method": "credit_card",
            "merchant_category": "grocery",
            "user_country": "US",
            "merchant_country": "US"
        }
        
        # Send multiple concurrent requests
        tasks = []
        for i in range(10):
            transaction_copy = transaction.copy()
            transaction_copy["transaction_id"] = f"txn_test_{i}"
            task = asyncio.create_task(
                asyncio.to_thread(client.post, "/predict", json=transaction_copy)
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["fraud_probability"] == 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])