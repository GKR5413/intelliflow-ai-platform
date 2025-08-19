"""
Production environment configuration for IntelliFlow Fraud Detection Service
"""

import os
from typing import Dict, Any, List

# Environment
ENVIRONMENT = "production"
DEBUG = False
LOG_LEVEL = "INFO"

# Server Configuration
HOST = "0.0.0.0"
PORT = 8083
WORKERS = int(os.getenv("UVICORN_WORKERS", "8"))
RELOAD = False

# Database Configuration
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "fraud_db"),
    "username": os.getenv("DB_USERNAME"),
    "password": os.getenv("DB_PASSWORD"),
    "pool_size": 50,
    "max_overflow": 100,
    "pool_timeout": 60,
    "pool_recycle": 3600,
    "echo": False,  # No SQL logging in production
    "ssl_mode": "require",
    "application_name": "fraud-detection-service"
}

# Redis Configuration
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "password": os.getenv("REDIS_PASSWORD"),
    "db": 0,
    "decode_responses": True,
    "socket_timeout": 30,
    "socket_connect_timeout": 30,
    "retry_on_timeout": True,
    "max_connections": 100,
    "ssl": True,
    "ssl_cert_reqs": "required",
    "health_check_interval": 30
}

# Kafka Configuration
KAFKA_CONFIG = {
    "bootstrap_servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS").split(","),
    "client_id": "fraud-detection-prod",
    "group_id": "fraud-detection-consumer-prod",
    "auto_offset_reset": "earliest",
    "enable_auto_commit": False,
    "max_poll_records": 500,
    "session_timeout_ms": 30000,
    "heartbeat_interval_ms": 10000,
    "fetch_max_wait_ms": 5000,
    "security_protocol": "SASL_SSL",
    "sasl_mechanism": "PLAIN",
    "sasl_username": os.getenv("KAFKA_USERNAME"),
    "sasl_password": os.getenv("KAFKA_PASSWORD")
}

# ML Model Configuration
MODEL_CONFIG = {
    "model_path": os.getenv("MODEL_PATH", "/models"),
    "model_name": "fraud_detection_model",
    "model_version": os.getenv("MODEL_VERSION", "1.0.0"),
    "model_type": "sklearn",
    "auto_reload": True,
    "reload_interval_seconds": 3600,  # 1 hour
    "warm_up_samples": 100,
    "batch_size": 1000,
    "prediction_timeout_seconds": 5,
    "feature_cache_ttl": 1800,  # 30 minutes
    "model_cache_ttl": 7200,    # 2 hours
    "enable_model_metrics": True,
    "model_performance_tracking": True
}

# Feature Store Configuration
FEATURE_STORE_CONFIG = {
    "enabled": True,
    "type": "feast",
    "feast_repo_path": os.getenv("FEAST_REPO_PATH", "/feast"),
    "online_store": {
        "type": "redis",
        "connection_string": f"rediss://:{REDIS_CONFIG['password']}@{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}/{REDIS_CONFIG['db']}"
    },
    "offline_store": {
        "type": "postgresql",
        "connection_string": f"postgresql://{DATABASE_CONFIG['username']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/feature_store"
    },
    "cache_ttl_seconds": 1800,
    "batch_size": 5000,
    "enable_feature_monitoring": True
}

# API Configuration
API_CONFIG = {
    "title": "IntelliFlow Fraud Detection API",
    "description": "Advanced fraud detection service using machine learning",
    "version": "1.0.0",
    "docs_url": None,  # Disabled in production
    "redoc_url": None,  # Disabled in production
    "openapi_url": None,  # Disabled in production
    "cors_origins": [
        "https://app.intelliflow.com",
        "https://admin.intelliflow.com"
    ],
    "cors_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "cors_headers": ["Authorization", "Content-Type", "X-Requested-With"],
    "max_request_size": 5 * 1024 * 1024,  # 5MB
    "request_timeout": 10
}

# Authentication Configuration
AUTH_CONFIG = {
    "enabled": True,
    "jwt_secret": os.getenv("JWT_SECRET"),
    "jwt_algorithm": "HS256",
    "jwt_expiration_seconds": 3600,
    "require_auth_endpoints": [
        "/predict",
        "/predict/batch",
        "/model/reload",
        "/admin/*"
    ],
    "api_key_header": "X-API-Key",
    "api_keys": {
        os.getenv("API_KEY_INTERNAL", ""): "internal-service-access",
        os.getenv("API_KEY_ADMIN", ""): "admin-access"
    }
}

# Rate Limiting Configuration
RATE_LIMIT_CONFIG = {
    "enabled": True,
    "requests_per_minute": 300,
    "burst_capacity": 50,
    "key_func": "ip_and_user",
    "whitelist_ips": os.getenv("RATE_LIMIT_WHITELIST_IPS", "").split(",") if os.getenv("RATE_LIMIT_WHITELIST_IPS") else []
}

# Caching Configuration
CACHE_CONFIG = {
    "enabled": True,
    "backend": "redis",
    "default_ttl": 1800,  # 30 minutes
    "key_prefix": "prod:fraud-detection:",
    "features": {
        "user_features": {"ttl": 1800, "enabled": True},
        "transaction_features": {"ttl": 300, "enabled": True},
        "merchant_features": {"ttl": 3600, "enabled": True},
        "model_predictions": {"ttl": 900, "enabled": True}
    },
    "cache_invalidation": {
        "enabled": True,
        "strategy": "time_based"
    }
}

# Monitoring Configuration
MONITORING_CONFIG = {
    "metrics_enabled": True,
    "prometheus_port": 8084,
    "health_check_interval": 30,
    "log_requests": False,  # Disabled for performance
    "log_responses": False,  # Disabled for performance
    "trace_requests": True,
    "custom_metrics": True,
    "alert_thresholds": {
        "response_time_p99_ms": 100,
        "error_rate_percent": 1.0,
        "memory_usage_percent": 85,
        "cpu_usage_percent": 80
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
        },
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "json",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "json",
            "filename": "/app/logs/fraud-detection.log",
            "maxBytes": 104857600,  # 100MB
            "backupCount": 10
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "json",
            "filename": "/app/logs/fraud-detection-error.log",
            "maxBytes": 104857600,  # 100MB
            "backupCount": 10
        }
    },
    "loggers": {
        "": {
            "level": "INFO",
            "handlers": ["console", "file"]
        },
        "error": {
            "level": "ERROR",
            "handlers": ["error_file"],
            "propagate": False
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False
        },
        "uvicorn.error": {
            "level": "ERROR",
            "handlers": ["console", "error_file"],
            "propagate": False
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["file"],
            "propagate": False
        }
    }
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "async_workers": 16,
    "thread_pool_workers": 32,
    "max_concurrent_requests": 1000,
    "request_queue_size": 2000,
    "enable_request_batching": True,
    "batch_timeout_ms": 10,
    "connection_pool_size": 100,
    "keepalive_timeout": 65,
    "max_requests_per_connection": 1000
}

# Production-specific Configuration
PROD_CONFIG = {
    "auto_reload_models": True,
    "mock_external_services": False,
    "generate_test_data": False,
    "enable_debug_endpoints": False,
    "skip_feature_validation": False,
    "log_feature_values": False,
    "save_predictions": True,
    "prediction_storage_path": "/app/data/predictions",
    "enable_graceful_shutdown": True,
    "shutdown_timeout": 30
}

# Security Configuration
SECURITY_CONFIG = {
    "enable_cors": True,
    "enable_csrf": True,
    "enable_rate_limiting": True,
    "allowed_hosts": [
        "fraud-detection.intelliflow.com",
        "api.intelliflow.com"
    ],
    "secure_headers": True,
    "ssl_required": True,
    "hsts_max_age": 31536000,
    "content_security_policy": "default-src 'self'",
    "x_frame_options": "DENY",
    "x_content_type_options": "nosniff"
}

# Testing Configuration
TEST_CONFIG = {
    "enable_test_endpoints": False,
    "test_data_path": None,
    "mock_model_responses": False,
    "performance_testing": False
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    "enabled_feature_groups": [
        "transaction_features",
        "user_behavioral_features", 
        "merchant_features",
        "device_features",
        "temporal_features",
        "velocity_features",
        "network_features",
        "risk_features"
    ],
    "feature_validation": {
        "enabled": True,
        "strict_mode": True,
        "log_validation_errors": True,
        "fail_on_validation_error": True
    },
    "feature_computation": {
        "parallel": True,
        "workers": 8,
        "timeout_seconds": 10
    },
    "feature_monitoring": {
        "enabled": True,
        "drift_detection": True,
        "distribution_tracking": True
    }
}

# External Services Configuration
EXTERNAL_SERVICES = {
    "mlflow": {
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
        "experiment_name": "fraud_detection_prod",
        "enabled": True,
        "timeout": 30,
        "retry_attempts": 3
    },
    "notification_service": {
        "base_url": os.getenv("NOTIFICATION_SERVICE_URL"),
        "enabled": True,
        "timeout": 10,
        "retry_attempts": 3
    },
    "user_service": {
        "base_url": os.getenv("USER_SERVICE_URL"),
        "enabled": True,
        "timeout": 5,
        "retry_attempts": 3
    },
    "feature_store_api": {
        "base_url": os.getenv("FEATURE_STORE_API_URL"),
        "enabled": True,
        "timeout": 10,
        "retry_attempts": 2
    }
}

# Batch Processing Configuration
BATCH_CONFIG = {
    "enabled": True,
    "max_batch_size": 5000,
    "batch_timeout_seconds": 10,
    "parallel_batches": 8,
    "retry_failed_batches": True,
    "max_retries": 3,
    "batch_queue_size": 1000,
    "enable_batch_metrics": True
}

# Model Validation Configuration
MODEL_VALIDATION_CONFIG = {
    "enabled": True,
    "validation_interval_hours": 6,
    "accuracy_threshold": 0.90,
    "drift_threshold": 0.05,
    "performance_threshold_ms": 50,
    "auto_retrain": True,
    "retrain_threshold": 0.02,
    "validation_data_percentage": 0.1
}

# A/B Testing Configuration
AB_TESTING_CONFIG = {
    "enabled": True,
    "default_experiment": "production_baseline",
    "traffic_split": {"control": 0.9, "treatment": 0.1},
    "random_seed": None,  # Use random seed in production
    "experiment_tracking": True,
    "auto_rollback": {
        "enabled": True,
        "error_rate_threshold": 0.05,
        "latency_threshold_ms": 200
    }
}

# Circuit Breaker Configuration
CIRCUIT_BREAKER_CONFIG = {
    "enabled": True,
    "failure_threshold": 50,
    "recovery_timeout": 60,
    "expected_exception": Exception,
    "name": "fraud_detection_circuit_breaker"
}

# Retry Configuration
RETRY_CONFIG = {
    "enabled": True,
    "max_attempts": 3,
    "backoff_factor": 2,
    "max_delay": 60,
    "exceptions": [
        "ConnectionError",
        "TimeoutError",
        "HTTPError"
    ]
}

# Data Quality Configuration
DATA_QUALITY_CONFIG = {
    "enabled": True,
    "validation_rules": {
        "required_fields": [
            "transaction_id",
            "user_id", 
            "amount",
            "currency",
            "payment_method"
        ],
        "data_types": {
            "amount": "float",
            "user_id": "int",
            "transaction_id": "string"
        },
        "value_ranges": {
            "amount": {"min": 0, "max": 1000000}
        }
    },
    "quality_thresholds": {
        "completeness": 0.95,
        "validity": 0.98,
        "consistency": 0.90
    }
}

# Backup and Recovery Configuration
BACKUP_CONFIG = {
    "enabled": True,
    "model_backup_interval_hours": 24,
    "prediction_backup_interval_hours": 6,
    "backup_retention_days": 30,
    "backup_location": os.getenv("BACKUP_S3_BUCKET", ""),
    "compression": True
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
    "prod": PROD_CONFIG,
    "security": SECURITY_CONFIG,
    "test": TEST_CONFIG,
    "features": FEATURE_CONFIG,
    "external_services": EXTERNAL_SERVICES,
    "batch": BATCH_CONFIG,
    "model_validation": MODEL_VALIDATION_CONFIG,
    "ab_testing": AB_TESTING_CONFIG,
    "circuit_breaker": CIRCUIT_BREAKER_CONFIG,
    "retry": RETRY_CONFIG,
    "data_quality": DATA_QUALITY_CONFIG,
    "backup": BACKUP_CONFIG
}