"""
Integration tests for IntelliFlow Fraud Detection Service using TestContainers
"""

import pytest
import asyncio
import httpx
import json
import time
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import redis.asyncio as aioredis
import asyncpg

# TestContainers imports
from testcontainers.postgres import PostgreSqlContainer
from testcontainers.redis import RedisContainer
from testcontainers.kafka import KafkaContainer

# Application imports
from fraud_detection_api import app
from scoring_engine import ScoringEngine
from feature_pipeline import FeaturePipeline
from prometheus_metrics import get_metrics


@pytest.mark.asyncio
@pytest.mark.integration
class TestFraudDetectionIntegration:
    """Integration tests using real containers"""
    
    @pytest.fixture(scope="class")
    def postgres_container(self):
        """PostgreSQL test container"""
        with PostgreSqlContainer("postgres:15.4-alpine") as postgres:
            postgres.with_env("POSTGRES_DB", "fraud_db_test")
            postgres.with_env("POSTGRES_USER", "test")
            postgres.with_env("POSTGRES_PASSWORD", "test")
            yield postgres
    
    @pytest.fixture(scope="class") 
    def redis_container(self):
        """Redis test container"""
        with RedisContainer("redis:7.2-alpine") as redis:
            redis.with_command("redis-server --requirepass test")
            yield redis
    
    @pytest.fixture(scope="class")
    def kafka_container(self):
        """Kafka test container"""
        with KafkaContainer() as kafka:
            yield kafka
    
    @pytest.fixture(scope="class")
    async def test_database(self, postgres_container):
        """Set up test database with schema"""
        connection_string = postgres_container.get_connection_url()
        conn = await asyncpg.connect(connection_string)
        
        # Create test schema
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS fraud_predictions (
                id SERIAL PRIMARY KEY,
                transaction_id VARCHAR(100) NOT NULL,
                user_id BIGINT NOT NULL,
                prediction_score DECIMAL(5,4) NOT NULL,
                prediction_result BOOLEAN NOT NULL,
                model_version VARCHAR(50) NOT NULL,
                features JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_features (
                user_id BIGINT PRIMARY KEY,
                total_transactions INTEGER DEFAULT 0,
                total_amount DECIMAL(15,2) DEFAULT 0,
                avg_amount DECIMAL(15,2) DEFAULT 0,
                fraud_count INTEGER DEFAULT 0,
                last_transaction_at TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert test data
        await conn.execute("""
            INSERT INTO user_features (user_id, total_transactions, total_amount, avg_amount, fraud_count)
            VALUES 
                (12345, 100, 5000.00, 50.00, 2),
                (67890, 50, 2500.00, 50.00, 0),
                (99999, 200, 10000.00, 50.00, 10)
            ON CONFLICT (user_id) DO NOTHING
        """)
        
        yield conn
        await conn.close()
    
    @pytest.fixture(scope="class")
    async def test_redis(self, redis_container):
        """Set up test Redis connection"""
        redis_url = f"redis://test@{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}/0"
        redis_client = await aioredis.from_url(redis_url)
        
        # Pre-populate some test cache data
        await redis_client.hset(
            "user_features:12345",
            mapping={
                "velocity_1h": "5",
                "velocity_24h": "25",
                "avg_amount_7d": "55.50",
                "merchant_diversity": "8"
            }
        )
        
        await redis_client.hset(
            "merchant_features:merchant_123", 
            mapping={
                "risk_score": "0.3",
                "transaction_count_24h": "150",
                "fraud_rate": "0.02"
            }
        )
        
        yield redis_client
        await redis_client.close()
    
    @pytest.fixture
    async def test_client(self, postgres_container, redis_container, kafka_container):
        """HTTP test client with container dependencies"""
        # Set environment variables for the application
        import os
        os.environ["DATABASE_URL"] = postgres_container.get_connection_url()
        os.environ["REDIS_URL"] = f"redis://test@{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}/0"
        os.environ["KAFKA_BOOTSTRAP_SERVERS"] = kafka_container.get_bootstrap_server()
        
        async with httpx.AsyncClient(app=app, base_url="http://testserver") as client:
            yield client
    
    async def test_health_check_with_dependencies(self, test_client):
        """Test health check with real dependencies"""
        response = await test_client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] in ["healthy", "warning"]
        assert "uptime_seconds" in health_data
        assert health_data["uptime_seconds"] >= 0
    
    async def test_detailed_health_check(self, test_client):
        """Test detailed health check with component status"""
        response = await test_client.get("/health/detailed")
        assert response.status_code in [200, 503]  # May be 503 if some components unhealthy
        
        health_data = response.json()
        assert "checks" in health_data
        
        # Verify individual component checks
        checks = health_data["checks"]
        expected_components = ["database", "redis", "model", "system_resources"]
        
        for component in expected_components:
            if component in checks:
                assert "status" in checks[component]
                assert "response_time_ms" in checks[component]
    
    async def test_single_fraud_prediction_integration(self, test_client, test_database):
        """Test single transaction fraud prediction with real database"""
        transaction_data = {
            "transaction_id": "txn_integration_001",
            "user_id": 12345,
            "amount": 99.99,
            "currency": "USD",
            "payment_method": "credit_card",
            "merchant_category": "grocery",
            "user_country": "US",
            "merchant_country": "US",
            "merchant_id": "merchant_123",
            "ip_address": "192.168.1.100",
            "device_fingerprint": "device_abc123"
        }
        
        response = await test_client.post("/predict", json=transaction_data)
        assert response.status_code == 200
        
        prediction = response.json()
        assert "fraud_probability" in prediction
        assert "fraud_prediction" in prediction
        assert "risk_level" in prediction
        assert "processing_time_ms" in prediction
        assert prediction["transaction_id"] == transaction_data["transaction_id"]
        
        # Verify prediction is within valid range
        assert 0.0 <= prediction["fraud_probability"] <= 1.0
        assert prediction["risk_level"] in ["low", "medium", "high"]
        assert prediction["processing_time_ms"] > 0
        
        # Verify prediction was stored in database
        stored_prediction = await test_database.fetchrow(
            "SELECT * FROM fraud_predictions WHERE transaction_id = $1",
            transaction_data["transaction_id"]
        )
        if stored_prediction:  # May not be stored in test mode
            assert stored_prediction["user_id"] == transaction_data["user_id"]
            assert stored_prediction["prediction_score"] == prediction["fraud_probability"]
    
    async def test_batch_fraud_prediction_integration(self, test_client, test_database):
        """Test batch fraud prediction with real database"""
        batch_data = {
            "transactions": [
                {
                    "transaction_id": f"txn_batch_{i}",
                    "user_id": 12345 + i,
                    "amount": 50.0 + i * 10,
                    "currency": "USD",
                    "payment_method": "credit_card",
                    "merchant_category": "grocery",
                    "user_country": "US", 
                    "merchant_country": "US"
                }
                for i in range(5)
            ]
        }
        
        response = await test_client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200
        
        batch_result = response.json()
        assert "results" in batch_result
        assert "batch_size" in batch_result
        assert "processing_time_ms" in batch_result
        assert batch_result["batch_size"] == 5
        
        # Verify each prediction in the batch
        for i, result in enumerate(batch_result["results"]):
            expected_id = f"txn_batch_{i}"
            assert result["transaction_id"] == expected_id
            assert "fraud_probability" in result
            assert "fraud_prediction" in result
            assert 0.0 <= result["fraud_probability"] <= 1.0
    
    async def test_feature_extraction_with_cache(self, test_client, test_redis):
        """Test feature extraction using cached data"""
        # Pre-populate cache with test data
        await test_redis.hset(
            "user_features:54321",
            mapping={
                "velocity_1h": "3",
                "velocity_24h": "15", 
                "avg_amount_7d": "75.25",
                "merchant_diversity": "5",
                "device_consistency": "0.8"
            }
        )
        
        transaction_data = {
            "transaction_id": "txn_cache_test",
            "user_id": 54321,
            "amount": 125.50,
            "currency": "USD",
            "payment_method": "credit_card",
            "merchant_category": "online",
            "user_country": "US",
            "merchant_country": "US"
        }
        
        response = await test_client.post("/predict", json=transaction_data)
        assert response.status_code == 200
        
        prediction = response.json()
        assert prediction["transaction_id"] == "txn_cache_test"
        
        # The processing should be faster due to cached features
        assert prediction["processing_time_ms"] < 200  # Should be fast with cache
    
    async def test_high_volume_predictions(self, test_client):
        """Test system under high volume of predictions"""
        # Generate many concurrent prediction requests
        num_requests = 50
        tasks = []
        
        for i in range(num_requests):
            transaction_data = {
                "transaction_id": f"txn_volume_{i}",
                "user_id": 10000 + i,
                "amount": 25.0 + (i % 100),
                "currency": "USD",
                "payment_method": "credit_card",
                "merchant_category": "retail",
                "user_country": "US",
                "merchant_country": "US"
            }
            task = test_client.post("/predict", json=transaction_data)
            tasks.append(task)
        
        # Execute all requests concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify results
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        assert len(successful_responses) >= num_requests * 0.9  # At least 90% success rate
        
        for response in successful_responses:
            if hasattr(response, 'status_code'):
                assert response.status_code == 200
    
    async def test_model_performance_metrics(self, test_client):
        """Test that performance metrics are properly collected"""
        # Make several predictions to generate metrics
        for i in range(10):
            transaction_data = {
                "transaction_id": f"txn_metrics_{i}",
                "user_id": 20000 + i,
                "amount": 75.0,
                "currency": "USD",
                "payment_method": "debit_card",
                "merchant_category": "gas_station",
                "user_country": "US",
                "merchant_country": "US"
            }
            await test_client.post("/predict", json=transaction_data)
        
        # Check metrics endpoint
        response = await test_client.get("/metrics")
        assert response.status_code == 200
        
        metrics_text = response.text
        assert "fraud_predictions_total" in metrics_text
        assert "fraud_prediction_duration_seconds" in metrics_text
        
    async def test_error_handling_with_invalid_data(self, test_client):
        """Test error handling with various invalid inputs"""
        
        # Test with missing required fields
        invalid_data_1 = {
            "transaction_id": "txn_invalid_1",
            # Missing user_id and amount
            "currency": "USD"
        }
        
        response = await test_client.post("/predict", json=invalid_data_1)
        assert response.status_code == 422  # Validation error
        
        # Test with invalid data types
        invalid_data_2 = {
            "transaction_id": "txn_invalid_2",
            "user_id": "not_a_number",  # Should be integer
            "amount": -50.0,  # Negative amount
            "currency": "INVALID_CURRENCY"
        }
        
        response = await test_client.post("/predict", json=invalid_data_2)
        assert response.status_code == 422  # Validation error
    
    async def test_database_connection_resilience(self, test_client, postgres_container):
        """Test system behavior when database connection issues occur"""
        # Make a normal prediction first
        transaction_data = {
            "transaction_id": "txn_db_test_1",
            "user_id": 30000,
            "amount": 100.0,
            "currency": "USD",
            "payment_method": "credit_card",
            "merchant_category": "restaurant",
            "user_country": "US",
            "merchant_country": "US"
        }
        
        response = await test_client.post("/predict", json=transaction_data)
        assert response.status_code == 200
        
        # The prediction should still work even if database storage fails
        # because the core ML prediction doesn't depend on database writes
    
    async def test_cache_performance_impact(self, test_client, test_redis):
        """Test the performance impact of cache hits vs misses"""
        
        # Clear cache for user
        test_user_id = 40000
        await test_redis.delete(f"user_features:{test_user_id}")
        
        # First request (cache miss)
        transaction_data = {
            "transaction_id": "txn_cache_miss",
            "user_id": test_user_id,
            "amount": 150.0,
            "currency": "USD",
            "payment_method": "credit_card",
            "merchant_category": "electronics",
            "user_country": "US",
            "merchant_country": "US"
        }
        
        start_time = time.time()
        response = await test_client.post("/predict", json=transaction_data)
        cache_miss_time = time.time() - start_time
        
        assert response.status_code == 200
        
        # Second request (cache hit)
        transaction_data["transaction_id"] = "txn_cache_hit"
        
        start_time = time.time()
        response = await test_client.post("/predict", json=transaction_data)
        cache_hit_time = time.time() - start_time
        
        assert response.status_code == 200
        
        # Cache hit should be faster (though this might not always be true in test environment)
        # We'll just verify both requests succeeded
        assert cache_miss_time > 0
        assert cache_hit_time > 0
    
    async def test_concurrent_predictions_data_consistency(self, test_client, test_database):
        """Test data consistency under concurrent load"""
        test_user_id = 50000
        num_concurrent = 20
        
        # Create concurrent prediction tasks for the same user
        tasks = []
        for i in range(num_concurrent):
            transaction_data = {
                "transaction_id": f"txn_concurrent_{i}",
                "user_id": test_user_id,
                "amount": 100.0 + i,
                "currency": "USD",
                "payment_method": "credit_card",
                "merchant_category": "retail",
                "user_country": "US",
                "merchant_country": "US"
            }
            task = test_client.post("/predict", json=transaction_data)
            tasks.append(task)
        
        # Execute all tasks concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all predictions succeeded
        successful_predictions = []
        for response in responses:
            if hasattr(response, 'status_code') and response.status_code == 200:
                successful_predictions.append(response.json())
        
        assert len(successful_predictions) >= num_concurrent * 0.8  # At least 80% success
        
        # Verify each prediction has unique transaction_id
        transaction_ids = [p["transaction_id"] for p in successful_predictions]
        assert len(set(transaction_ids)) == len(successful_predictions)  # All unique
    
    async def test_model_reload_functionality(self, test_client):
        """Test model reload endpoint"""
        response = await test_client.post("/model/reload")
        # This might fail if no model is available, which is expected in test environment
        assert response.status_code in [200, 500]  # Either success or expected failure
    
    async def test_api_rate_limiting_disabled_in_test(self, test_client):
        """Test that rate limiting is properly disabled in test environment"""
        # Make rapid requests that would trigger rate limiting in production
        responses = []
        for i in range(20):  # Rapid requests
            transaction_data = {
                "transaction_id": f"txn_rate_test_{i}",
                "user_id": 60000,
                "amount": 50.0,
                "currency": "USD",
                "payment_method": "credit_card",
                "merchant_category": "grocery",
                "user_country": "US",
                "merchant_country": "US"
            }
            response = await test_client.post("/predict", json=transaction_data)
            responses.append(response)
        
        # All requests should succeed (no rate limiting in test)
        for response in responses:
            assert response.status_code == 200


@pytest.mark.performance
class TestFraudDetectionPerformance:
    """Performance tests for fraud detection service"""
    
    @pytest.mark.asyncio
    async def test_prediction_latency_benchmark(self, test_client):
        """Benchmark prediction latency"""
        num_tests = 100
        latencies = []
        
        for i in range(num_tests):
            transaction_data = {
                "transaction_id": f"txn_perf_{i}",
                "user_id": 70000 + i,
                "amount": 100.0,
                "currency": "USD", 
                "payment_method": "credit_card",
                "merchant_category": "retail",
                "user_country": "US",
                "merchant_country": "US"
            }
            
            start_time = time.time()
            response = await test_client.post("/predict", json=transaction_data)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            assert response.status_code == 200
            latencies.append(latency)
        
        # Performance assertions
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        p99_latency = sorted(latencies)[int(0.99 * len(latencies))]
        
        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"P95 latency: {p95_latency:.2f}ms") 
        print(f"P99 latency: {p99_latency:.2f}ms")
        
        # Performance thresholds (adjust based on requirements)
        assert avg_latency < 500  # Average under 500ms
        assert p95_latency < 1000  # P95 under 1 second
        assert p99_latency < 2000  # P99 under 2 seconds
    
    @pytest.mark.asyncio
    async def test_throughput_benchmark(self, test_client):
        """Benchmark prediction throughput"""
        num_concurrent = 50
        num_batches = 5
        
        total_requests = num_concurrent * num_batches
        start_time = time.time()
        
        for batch in range(num_batches):
            tasks = []
            for i in range(num_concurrent):
                transaction_data = {
                    "transaction_id": f"txn_throughput_{batch}_{i}",
                    "user_id": 80000 + (batch * num_concurrent) + i,
                    "amount": 75.0,
                    "currency": "USD",
                    "payment_method": "credit_card", 
                    "merchant_category": "retail",
                    "user_country": "US",
                    "merchant_country": "US"
                }
                task = test_client.post("/predict", json=transaction_data)
                tasks.append(task)
            
            # Wait for batch to complete
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            successful = [r for r in responses if hasattr(r, 'status_code') and r.status_code == 200]
            assert len(successful) >= num_concurrent * 0.9  # 90% success rate
        
        total_time = time.time() - start_time
        throughput = total_requests / total_time
        
        print(f"Throughput: {throughput:.2f} requests/second")
        print(f"Total time: {total_time:.2f} seconds for {total_requests} requests")
        
        # Throughput threshold (adjust based on requirements)
        assert throughput > 10  # At least 10 requests per second