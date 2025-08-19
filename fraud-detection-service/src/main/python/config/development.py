"""
Development environment configuration for IntelliFlow Fraud Detection Service
"""

import os
from typing import Dict, Any, List

# Environment
ENVIRONMENT = "development"
DEBUG = True
LOG_LEVEL = "DEBUG"

# Server Configuration
HOST = "0.0.0.0"
PORT = 8083
WORKERS = 1
RELOAD = True

# Database Configuration
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "fraud_db"),
    "username": os.getenv("DB_USERNAME", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "pool_size": 5,
    "max_overflow": 10,
    "pool_timeout": 30,
    "pool_recycle": 3600,
    "echo": True  # SQL logging enabled for development
}

# Redis Configuration
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "password": os.getenv("REDIS_PASSWORD", "redis123"),
    "db": 0,
    "decode_responses": True,
    "socket_timeout": 10,
    "socket_connect_timeout": 10,
    "retry_on_timeout": True,
    "max_connections": 20
}

# Kafka Configuration
KAFKA_CONFIG = {
    "bootstrap_servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(","),
    "client_id": "fraud-detection-dev",
    "group_id": "fraud-detection-consumer-dev",
    "auto_offset_reset": "earliest",
    "enable_auto_commit": False,
    "max_poll_records": 100,
    "session_timeout_ms": 30000
}

# ML Model Configuration
MODEL_CONFIG = {
    "model_path": os.getenv("MODEL_PATH", "/models"),
    "model_name": "fraud_detection_model",
    "model_version": "1.0.0",
    "model_type": "sklearn",
    "auto_reload": True,
    "reload_interval_seconds": 300,  # 5 minutes
    "warm_up_samples": 10,
    "batch_size": 100,
    "prediction_timeout_seconds": 30,
    "feature_cache_ttl": 300,  # 5 minutes
    "model_cache_ttl": 3600   # 1 hour
}

# Feature Store Configuration
FEATURE_STORE_CONFIG = {
    "enabled": True,
    "type": "feast",
    "feast_repo_path": os.getenv("FEAST_REPO_PATH", "/feast"),
    "online_store": {
        "type": "redis",
        "connection_string": f"redis://:{REDIS_CONFIG['password']}@{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}/{REDIS_CONFIG['db']}"
    },
    "offline_store": {
        "type": "file",
        "path": "/tmp/feast/offline"
    },
    "cache_ttl_seconds": 300,
    "batch_size": 1000
}

# API Configuration
API_CONFIG = {
    "title": "IntelliFlow Fraud Detection API",
    "description": "Advanced fraud detection service using machine learning",
    "version": "1.0.0",
    "docs_url": "/docs",
    "redoc_url": "/redoc",
    "openapi_url": "/openapi.json",
    "cors_origins": [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:4200",
        "http://127.0.0.1:3000"
    ],
    "cors_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "cors_headers": ["*"],
    "max_request_size": 10 * 1024 * 1024,  # 10MB
    "request_timeout": 30
}

# Authentication Configuration
AUTH_CONFIG = {
    "enabled": False,  # Disabled for development
    "jwt_secret": os.getenv("JWT_SECRET", "dev-secret-key"),
    "jwt_algorithm": "HS256",
    "jwt_expiration_seconds": 3600,
    "require_auth_endpoints": [],  # No endpoints require auth in dev
    "api_key_header": "X-API-Key",
    "api_keys": {
        "dev-key": "development-access"
    }
}

# Rate Limiting Configuration
RATE_LIMIT_CONFIG = {
    "enabled": False,  # Disabled for development
    "requests_per_minute": 1000,
    "burst_capacity": 200,
    "key_func": "ip"  # Rate limit by IP address
}

# Caching Configuration
CACHE_CONFIG = {
    "enabled": True,
    "backend": "redis",
    "default_ttl": 300,  # 5 minutes
    "key_prefix": "dev:fraud-detection:",
    "features": {
        "user_features": {"ttl": 300, "enabled": True},
        "transaction_features": {"ttl": 60, "enabled": True},
        "merchant_features": {"ttl": 600, "enabled": True},
        "model_predictions": {"ttl": 30, "enabled": False}  # Disabled in dev
    }
}

# Monitoring Configuration
MONITORING_CONFIG = {
    "metrics_enabled": True,
    "prometheus_port": 8084,
    "health_check_interval": 30,
    "log_requests": True,
    "log_responses": True,
    "trace_requests": True,
    "custom_metrics": True
}

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
        },
        "simple": {
            "format": "%(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "logs/fraud-detection-dev.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "loggers": {
        "": {
            "level": "DEBUG",
            "handlers": ["console", "file"]
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False
        },
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False
        }
    }
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "async_workers": 4,
    "thread_pool_workers": 8,
    "max_concurrent_requests": 100,
    "request_queue_size": 200,
    "enable_request_batching": True,
    "batch_timeout_ms": 50,
    "connection_pool_size": 20
}

# Development-specific Configuration
DEV_CONFIG = {
    "auto_reload_models": True,
    "mock_external_services": True,
    "generate_test_data": True,
    "enable_debug_endpoints": True,
    "skip_feature_validation": True,
    "log_feature_values": True,
    "save_predictions": True,
    "prediction_storage_path": "/tmp/predictions"
}

# Security Configuration
SECURITY_CONFIG = {
    "enable_cors": True,
    "enable_csrf": False,
    "enable_rate_limiting": False,
    "allowed_hosts": ["*"],
    "secure_headers": False,
    "ssl_required": False
}

# Testing Configuration
TEST_CONFIG = {
    "enable_test_endpoints": True,
    "test_data_path": "/tmp/test_data",
    "mock_model_responses": False,
    "performance_testing": True
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    "enabled_feature_groups": [
        "transaction_features",
        "user_behavioral_features", 
        "merchant_features",
        "device_features",
        "temporal_features",
        "velocity_features"
    ],
    "feature_validation": {
        "enabled": False,  # Relaxed for development
        "strict_mode": False,
        "log_validation_errors": True
    },
    "feature_computation": {
        "parallel": True,
        "workers": 4,
        "timeout_seconds": 30
    }
}

# External Services Configuration
EXTERNAL_SERVICES = {
    "mlflow": {
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        "experiment_name": "fraud_detection_dev",
        "enabled": True,
        "timeout": 10
    },
    "notification_service": {
        "base_url": "http://localhost:8085",
        "enabled": False,  # Mock in development
        "timeout": 5
    },
    "user_service": {
        "base_url": "http://localhost:8081",
        "enabled": True,
        "timeout": 5
    }
}

# Batch Processing Configuration
BATCH_CONFIG = {
    "enabled": True,
    "max_batch_size": 1000,
    "batch_timeout_seconds": 30,
    "parallel_batches": 2,
    "retry_failed_batches": True,
    "max_retries": 3
}

# Model Validation Configuration
MODEL_VALIDATION_CONFIG = {
    "enabled": True,
    "validation_interval_hours": 24,
    "accuracy_threshold": 0.85,
    "drift_threshold": 0.1,
    "performance_threshold_ms": 100,
    "auto_retrain": False  # Manual retraining in development
}

# A/B Testing Configuration
AB_TESTING_CONFIG = {
    "enabled": True,
    "default_experiment": "baseline_vs_new_model",
    "traffic_split": {"control": 0.8, "treatment": 0.2},
    "random_seed": 42
}

# All configuration combined
CONFIG = {
    "environment": ENVIRONMENT,
    "debug": DEBUG,
    "log_level": LOG_LEVEL,
    "host": HOST,
    "port": PORT,
    "workers": WORKERS,
    "reload": RELOAD,
    "database": DATABASE_CONFIG,
    "redis": REDIS_CONFIG,
    "kafka": KAFKA_CONFIG,
    "model": MODEL_CONFIG,
    "feature_store": FEATURE_STORE_CONFIG,
    "api": API_CONFIG,
    "auth": AUTH_CONFIG,
    "rate_limit": RATE_LIMIT_CONFIG,
    "cache": CACHE_CONFIG,
    "monitoring": MONITORING_CONFIG,
    "logging": LOGGING_CONFIG,
    "performance": PERFORMANCE_CONFIG,
    "dev": DEV_CONFIG,
    "security": SECURITY_CONFIG,
    "test": TEST_CONFIG,
    "features": FEATURE_CONFIG,
    "external_services": EXTERNAL_SERVICES,
    "batch": BATCH_CONFIG,
    "model_validation": MODEL_VALIDATION_CONFIG,
    "ab_testing": AB_TESTING_CONFIG
}