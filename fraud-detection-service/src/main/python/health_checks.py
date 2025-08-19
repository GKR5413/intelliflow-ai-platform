"""
Comprehensive health check system for Fraud Detection Service
"""

import asyncio
import time
import logging
import os
import psutil
import asyncpg
import aioredis
import joblib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    """Health check result structure"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    response_time_ms: float

@dataclass
class OverallHealth:
    """Overall system health"""
    status: HealthStatus
    message: str
    timestamp: datetime
    checks: List[HealthCheck]
    uptime_seconds: float
    version: str

class HealthChecker:
    """Comprehensive health checking system"""
    
    def __init__(self):
        self.start_time = time.time()
        self.version = os.getenv('SERVICE_VERSION', '1.0.0')
        self.db_pool = None
        self.redis_client = None
        self.model = None
        self.model_loaded_at = None
        
    async def initialize(self):
        """Initialize health checker dependencies"""
        try:
            # Initialize database connection
            if os.getenv('DATABASE_URL'):
                self.db_pool = await asyncpg.create_pool(
                    os.getenv('DATABASE_URL'),
                    min_size=1,
                    max_size=2,
                    command_timeout=5
                )
            
            # Initialize Redis connection
            if os.getenv('REDIS_URL'):
                self.redis_client = aioredis.from_url(
                    os.getenv('REDIS_URL'),
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
            
            # Load ML model for testing
            await self._load_test_model()
            
            logger.info("Health checker initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize health checker: {e}")
            raise
    
    async def _load_test_model(self):
        """Load a test model for health checks"""
        try:
            model_path = os.getenv('MODEL_PATH', '/models')
            if os.path.exists(f"{model_path}/fraud_model.joblib"):
                self.model = joblib.load(f"{model_path}/fraud_model.joblib")
                self.model_loaded_at = datetime.now()
                logger.info("Test model loaded for health checks")
        except Exception as e:
            logger.warning(f"Could not load test model: {e}")
    
    async def check_database(self) -> HealthCheck:
        """Check database connectivity and performance"""
        start_time = time.time()
        
        try:
            if not self.db_pool:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.UNKNOWN,
                    message="Database not configured",
                    details={},
                    timestamp=datetime.now(),
                    response_time_ms=0
                )
            
            async with self.db_pool.acquire() as conn:
                # Test basic connectivity
                result = await conn.fetchval("SELECT 1")
                
                # Test transaction performance
                async with conn.transaction():
                    await conn.execute("SELECT pg_sleep(0.01)")  # 10ms test
                
                # Get database stats
                db_size = await conn.fetchval("""
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                """)
                
                active_connections = await conn.fetchval("""
                    SELECT count(*) FROM pg_stat_activity 
                    WHERE state = 'active'
                """)
                
                response_time = (time.time() - start_time) * 1000
                
                # Determine status based on response time
                if response_time > 1000:  # 1 second
                    status = HealthStatus.UNHEALTHY
                    message = f"Database response time too high: {response_time:.2f}ms"
                elif response_time > 500:  # 500ms
                    status = HealthStatus.WARNING
                    message = f"Database response time elevated: {response_time:.2f}ms"
                else:
                    status = HealthStatus.HEALTHY
                    message = "Database is healthy"
                
                return HealthCheck(
                    name="database",
                    status=status,
                    message=message,
                    details={
                        "response_time_ms": round(response_time, 2),
                        "database_size": db_size,
                        "active_connections": active_connections,
                        "test_query_result": result
                    },
                    timestamp=datetime.now(),
                    response_time_ms=round(response_time, 2)
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database check failed: {str(e)}",
                details={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                timestamp=datetime.now(),
                response_time_ms=round(response_time, 2)
            )
    
    async def check_redis(self) -> HealthCheck:
        """Check Redis connectivity and performance"""
        start_time = time.time()
        
        try:
            if not self.redis_client:
                return HealthCheck(
                    name="redis",
                    status=HealthStatus.UNKNOWN,
                    message="Redis not configured",
                    details={},
                    timestamp=datetime.now(),
                    response_time_ms=0
                )
            
            # Test basic connectivity
            pong = await self.redis_client.ping()
            
            # Test read/write operations
            test_key = f"health_check_{int(time.time())}"
            test_value = "health_test_value"
            
            await self.redis_client.set(test_key, test_value, ex=60)
            retrieved_value = await self.redis_client.get(test_key)
            await self.redis_client.delete(test_key)
            
            # Get Redis info
            info = await self.redis_client.info()
            
            response_time = (time.time() - start_time) * 1000
            
            # Validate test operations
            if not pong or retrieved_value.decode() != test_value:
                status = HealthStatus.UNHEALTHY
                message = "Redis operations failed"
            elif response_time > 100:  # 100ms
                status = HealthStatus.WARNING
                message = f"Redis response time elevated: {response_time:.2f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = "Redis is healthy"
            
            # Extract relevant Redis info
            redis_details = {
                "response_time_ms": round(response_time, 2),
                "ping_result": pong,
                "read_write_test": "passed" if retrieved_value.decode() == test_value else "failed",
                "redis_version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0)
            }
            
            return HealthCheck(
                name="redis",
                status=status,
                message=message,
                details=redis_details,
                timestamp=datetime.now(),
                response_time_ms=round(response_time, 2)
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis check failed: {str(e)}",
                details={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                timestamp=datetime.now(),
                response_time_ms=round(response_time, 2)
            )
    
    async def check_model(self) -> HealthCheck:
        """Check ML model health and performance"""
        start_time = time.time()
        
        try:
            if not self.model:
                return HealthCheck(
                    name="model",
                    status=HealthStatus.WARNING,
                    message="ML model not loaded",
                    details={
                        "model_path": os.getenv('MODEL_PATH', '/models'),
                        "model_loaded_at": None
                    },
                    timestamp=datetime.now(),
                    response_time_ms=0
                )
            
            # Create test data for model inference
            # This should match your model's expected input format
            test_features = np.random.rand(1, 10)  # Adjust dimensions as needed
            
            # Test model prediction
            prediction_start = time.time()
            if hasattr(self.model, 'predict_proba'):
                prediction = self.model.predict_proba(test_features)[0]
            elif hasattr(self.model, 'predict'):
                prediction = self.model.predict(test_features)[0]
            else:
                raise Exception("Model doesn't have predict or predict_proba method")
            
            prediction_time = (time.time() - prediction_start) * 1000
            response_time = (time.time() - start_time) * 1000
            
            # Validate prediction
            if prediction is None:
                status = HealthStatus.UNHEALTHY
                message = "Model prediction failed"
            elif prediction_time > 100:  # 100ms
                status = HealthStatus.WARNING
                message = f"Model inference time elevated: {prediction_time:.2f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = "ML model is healthy"
            
            model_details = {
                "response_time_ms": round(response_time, 2),
                "prediction_time_ms": round(prediction_time, 2),
                "model_loaded_at": self.model_loaded_at.isoformat() if self.model_loaded_at else None,
                "model_type": type(self.model).__name__,
                "test_prediction": float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
                "model_age_hours": round((datetime.now() - self.model_loaded_at).total_seconds() / 3600, 2) if self.model_loaded_at else None
            }
            
            return HealthCheck(
                name="model",
                status=status,
                message=message,
                details=model_details,
                timestamp=datetime.now(),
                response_time_ms=round(response_time, 2)
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="model",
                status=HealthStatus.UNHEALTHY,
                message=f"Model check failed: {str(e)}",
                details={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "model_path": os.getenv('MODEL_PATH', '/models')
                },
                timestamp=datetime.now(),
                response_time_ms=round(response_time, 2)
            )
    
    async def check_system_resources(self) -> HealthCheck:
        """Check system resource utilization"""
        start_time = time.time()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine overall status
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = "Critical resource usage detected"
            elif cpu_percent > 80 or memory_percent > 80 or disk_percent > 85:
                status = HealthStatus.WARNING
                message = "High resource usage detected"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources are healthy"
            
            resource_details = {
                "response_time_ms": round(response_time, 2),
                "cpu_percent": round(cpu_percent, 2),
                "memory_percent": round(memory_percent, 2),
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "disk_percent": round(disk_percent, 2),
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv,
                "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
            }
            
            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                details=resource_details,
                timestamp=datetime.now(),
                response_time_ms=round(response_time, 2)
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"System resource check failed: {str(e)}",
                details={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                timestamp=datetime.now(),
                response_time_ms=round(response_time, 2)
            )
    
    async def check_external_dependencies(self) -> HealthCheck:
        """Check external service dependencies"""
        start_time = time.time()
        
        try:
            external_checks = []
            
            # Check MLflow if configured
            mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
            if mlflow_uri:
                try:
                    import aiohttp
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                        async with session.get(f"{mlflow_uri}/health") as response:
                            if response.status == 200:
                                external_checks.append({"mlflow": "healthy"})
                            else:
                                external_checks.append({"mlflow": f"unhealthy (status: {response.status})"})
                except Exception as e:
                    external_checks.append({"mlflow": f"failed ({str(e)})"})
            
            # Check feature store or other external APIs
            feature_store_url = os.getenv('FEATURE_STORE_URL')
            if feature_store_url:
                try:
                    import aiohttp
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                        async with session.get(f"{feature_store_url}/health") as response:
                            if response.status == 200:
                                external_checks.append({"feature_store": "healthy"})
                            else:
                                external_checks.append({"feature_store": f"unhealthy (status: {response.status})"})
                except Exception as e:
                    external_checks.append({"feature_store": f"failed ({str(e)})"})
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine status based on external service health
            failed_services = [check for check in external_checks 
                             if any("failed" in str(v) or "unhealthy" in str(v) for v in check.values())]
            
            if failed_services:
                if len(failed_services) == len(external_checks):
                    status = HealthStatus.UNHEALTHY
                    message = "All external dependencies are unhealthy"
                else:
                    status = HealthStatus.WARNING
                    message = f"{len(failed_services)} of {len(external_checks)} external dependencies are unhealthy"
            else:
                status = HealthStatus.HEALTHY
                message = "All external dependencies are healthy"
            
            return HealthCheck(
                name="external_dependencies",
                status=status,
                message=message,
                details={
                    "response_time_ms": round(response_time, 2),
                    "checks": external_checks,
                    "total_dependencies": len(external_checks),
                    "healthy_dependencies": len(external_checks) - len(failed_services)
                },
                timestamp=datetime.now(),
                response_time_ms=round(response_time, 2)
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="external_dependencies",
                status=HealthStatus.UNHEALTHY,
                message=f"External dependency check failed: {str(e)}",
                details={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                timestamp=datetime.now(),
                response_time_ms=round(response_time, 2)
            )
    
    async def get_overall_health(self) -> OverallHealth:
        """Get comprehensive health status"""
        # Run all health checks in parallel
        checks = await asyncio.gather(
            self.check_database(),
            self.check_redis(),
            self.check_model(),
            self.check_system_resources(),
            self.check_external_dependencies(),
            return_exceptions=True
        )
        
        # Filter out exceptions and convert to HealthCheck objects
        valid_checks = []
        for check in checks:
            if isinstance(check, HealthCheck):
                valid_checks.append(check)
            elif isinstance(check, Exception):
                # Create a health check for the failed check
                valid_checks.append(HealthCheck(
                    name="unknown",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(check)}",
                    details={"error": str(check)},
                    timestamp=datetime.now(),
                    response_time_ms=0
                ))
        
        # Determine overall status
        unhealthy_checks = [c for c in valid_checks if c.status == HealthStatus.UNHEALTHY]
        warning_checks = [c for c in valid_checks if c.status == HealthStatus.WARNING]
        
        if unhealthy_checks:
            overall_status = HealthStatus.UNHEALTHY
            message = f"{len(unhealthy_checks)} critical health check(s) failed"
        elif warning_checks:
            overall_status = HealthStatus.WARNING
            message = f"{len(warning_checks)} health check(s) have warnings"
        else:
            overall_status = HealthStatus.HEALTHY
            message = "All health checks passed"
        
        uptime = time.time() - self.start_time
        
        return OverallHealth(
            status=overall_status,
            message=message,
            timestamp=datetime.now(),
            checks=valid_checks,
            uptime_seconds=round(uptime, 2),
            version=self.version
        )

# FastAPI health endpoints
def create_health_app() -> FastAPI:
    """Create FastAPI app with health endpoints"""
    app = FastAPI(title="Fraud Detection Service Health", version="1.0.0")
    health_checker = HealthChecker()
    
    @app.on_event("startup")
    async def startup_event():
        await health_checker.initialize()
    
    @app.get("/health", response_model=dict)
    async def health():
        """Basic health check endpoint"""
        overall_health = await health_checker.get_overall_health()
        
        status_code = 200
        if overall_health.status == HealthStatus.UNHEALTHY:
            status_code = 503
        elif overall_health.status == HealthStatus.WARNING:
            status_code = 200  # Still considered healthy for load balancer
        
        return JSONResponse(
            status_code=status_code,
            content={
                "status": overall_health.status.value,
                "message": overall_health.message,
                "timestamp": overall_health.timestamp.isoformat(),
                "uptime_seconds": overall_health.uptime_seconds,
                "version": overall_health.version
            }
        )
    
    @app.get("/health/detailed", response_model=dict)
    async def health_detailed():
        """Detailed health check with all component status"""
        overall_health = await health_checker.get_overall_health()
        
        status_code = 200
        if overall_health.status == HealthStatus.UNHEALTHY:
            status_code = 503
        
        # Convert health checks to dict format
        checks_dict = {}
        for check in overall_health.checks:
            checks_dict[check.name] = {
                "status": check.status.value,
                "message": check.message,
                "details": check.details,
                "timestamp": check.timestamp.isoformat(),
                "response_time_ms": check.response_time_ms
            }
        
        return JSONResponse(
            status_code=status_code,
            content={
                "status": overall_health.status.value,
                "message": overall_health.message,
                "timestamp": overall_health.timestamp.isoformat(),
                "uptime_seconds": overall_health.uptime_seconds,
                "version": overall_health.version,
                "checks": checks_dict
            }
        )
    
    @app.get("/health/ready", response_model=dict)
    async def readiness():
        """Readiness probe endpoint for Kubernetes"""
        # Check critical dependencies for readiness
        db_check = await health_checker.check_database()
        model_check = await health_checker.check_model()
        
        critical_checks = [db_check, model_check]
        unhealthy_critical = [c for c in critical_checks if c.status == HealthStatus.UNHEALTHY]
        
        if unhealthy_critical:
            return JSONResponse(
                status_code=503,
                content={
                    "ready": False,
                    "message": f"{len(unhealthy_critical)} critical dependencies are unhealthy",
                    "critical_failures": [c.name for c in unhealthy_critical]
                }
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "ready": True,
                "message": "Service is ready to handle requests"
            }
        )
    
    @app.get("/health/live", response_model=dict)
    async def liveness():
        """Liveness probe endpoint for Kubernetes"""
        # Simple liveness check - just verify the service is responsive
        try:
            return {
                "alive": True,
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": round(time.time() - health_checker.start_time, 2)
            }
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={
                    "alive": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    return app

if __name__ == "__main__":
    # Run health check server standalone
    app = create_health_app()
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("HEALTH_PORT", "8001")),
        log_level="info"
    )
