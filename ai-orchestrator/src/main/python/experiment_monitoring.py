"""
Experiment tracking and performance monitoring for AI Orchestrator
"""

import os
import logging
import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid
import statistics
from pathlib import Path
import hashlib

# FastAPI and web framework
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# Pydantic models
from pydantic import BaseModel, Field, validator
from pydantic.types import UUID4

# Database and storage
import asyncpg
import aioredis
from motor.motor_asyncio import AsyncIOMotorClient
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

# ML tracking and monitoring
import mlflow
import mlflow.tracking
from mlflow.tracking import MlflowClient
import wandb

# Data processing and analysis
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# Monitoring and observability
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST
import structlog
import psutil

# Alerts and notifications
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import slack_sdk
from slack_sdk import WebClient

# Statistical analysis
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configure logging
logger = structlog.get_logger()

# Prometheus metrics
EXPERIMENT_RUNS_TOTAL = Counter('experiment_runs_total', 'Total experiment runs', ['experiment_name', 'status'])
EXPERIMENT_DURATION = Histogram('experiment_duration_seconds', 'Experiment duration', ['experiment_name'])
MODEL_PERFORMANCE_SCORE = Gauge('model_performance_score', 'Model performance score', ['model_name', 'metric'])
MONITORING_ALERTS_TOTAL = Counter('monitoring_alerts_total', 'Total monitoring alerts', ['alert_type', 'severity'])
DRIFT_DETECTION_SCORE = Gauge('drift_detection_score', 'Data/model drift score', ['model_name', 'drift_type'])
SYSTEM_RESOURCE_USAGE = Gauge('system_resource_usage', 'System resource usage', ['resource_type'])


class ExperimentStatus(Enum):
    """Experiment status enumeration"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class AlertSeverity(Enum):
    """Alert severity enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftType(Enum):
    """Drift type enumeration"""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    experiment_id: str
    name: str
    description: str
    model_type: str
    dataset_path: str
    hyperparameters: Dict[str, Any]
    metrics_to_track: List[str]
    early_stopping: bool = True
    max_epochs: int = 100
    checkpoint_frequency: int = 10
    notification_config: Dict[str, Any] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.notification_config is None:
            self.notification_config = {}
        if self.tags is None:
            self.tags = []


@dataclass
class PerformanceMetrics:
    """Performance metrics structure"""
    timestamp: datetime
    model_name: str
    model_version: str
    metrics: Dict[str, float]
    latency_ms: float
    throughput_qps: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float


@dataclass
class DriftDetectionResult:
    """Drift detection result"""
    timestamp: datetime
    model_name: str
    drift_type: DriftType
    drift_score: float
    threshold: float
    is_drift_detected: bool
    affected_features: List[str]
    statistical_test: str
    p_value: float
    confidence_interval: Tuple[float, float]


@dataclass
class Alert:
    """Alert structure"""
    alert_id: str
    timestamp: datetime
    alert_type: str
    severity: AlertSeverity
    message: str
    model_name: str
    experiment_id: Optional[str]
    metrics: Dict[str, Any]
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None


# Pydantic models for API
class ExperimentCreateRequest(BaseModel):
    """Experiment creation request"""
    name: str = Field(..., description="Experiment name")
    description: str = Field("", description="Experiment description")
    model_type: str = Field(..., description="Model type")
    dataset_path: str = Field(..., description="Dataset path")
    hyperparameters: Dict[str, Any] = Field(..., description="Hyperparameters")
    metrics_to_track: List[str] = Field(default_factory=list, description="Metrics to track")
    tags: List[str] = Field(default_factory=list, description="Experiment tags")


class MetricUpdate(BaseModel):
    """Metric update model"""
    experiment_id: str
    epoch: int
    metrics: Dict[str, float]
    timestamp: Optional[datetime] = None


class AlertCreateRequest(BaseModel):
    """Alert creation request"""
    alert_type: str
    severity: AlertSeverity
    message: str
    model_name: str
    experiment_id: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class ExperimentTracker:
    """Comprehensive experiment tracking system"""
    
    def __init__(self, 
                 db_pool: asyncpg.Pool,
                 redis_client: aioredis.Redis,
                 influx_client: influxdb_client.InfluxDBClient,
                 mlflow_client: MlflowClient):
        
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.influx_client = influx_client
        self.mlflow_client = mlflow_client
        
        # InfluxDB setup
        self.influx_write_api = influx_client.write_api(write_options=SYNCHRONOUS)
        self.influx_query_api = influx_client.query_api()
        
        # Active experiments tracking
        self.active_experiments = {}
        self.experiment_callbacks = {}
        
        # WebSocket connections for real-time updates
        self.websocket_connections = {}
        
        # Initialize W&B if configured
        self.wandb_enabled = os.getenv('WANDB_API_KEY') is not None
        if self.wandb_enabled:
            wandb.login(key=os.getenv('WANDB_API_KEY'))
    
    async def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment"""
        try:
            # Store experiment in database
            await self._store_experiment_config(config)
            
            # Initialize MLflow experiment
            mlflow_exp_id = await self._create_mlflow_experiment(config)
            
            # Initialize W&B if enabled
            if self.wandb_enabled:
                await self._create_wandb_experiment(config)
            
            # Store in active experiments
            self.active_experiments[config.experiment_id] = {
                'config': config,
                'status': ExperimentStatus.QUEUED,
                'start_time': None,
                'end_time': None,
                'metrics_history': [],
                'mlflow_exp_id': mlflow_exp_id
            }
            
            logger.info(f"Experiment created: {config.experiment_id}")
            EXPERIMENT_RUNS_TOTAL.labels(experiment_name=config.name, status='created').inc()
            
            return config.experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment"""
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.active_experiments[experiment_id]
            experiment['status'] = ExperimentStatus.RUNNING
            experiment['start_time'] = datetime.now()
            
            # Update database
            await self._update_experiment_status(experiment_id, ExperimentStatus.RUNNING)
            
            # Start MLflow run
            config = experiment['config']
            mlflow_run = mlflow.start_run(
                experiment_id=experiment['mlflow_exp_id'],
                run_name=f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            experiment['mlflow_run_id'] = mlflow_run.info.run_id
            
            # Log hyperparameters
            mlflow.log_params(config.hyperparameters)
            
            # Start W&B run if enabled
            if self.wandb_enabled:
                wandb.init(
                    project=config.name,
                    config=config.hyperparameters,
                    tags=config.tags
                )
            
            logger.info(f"Experiment started: {experiment_id}")
            EXPERIMENT_RUNS_TOTAL.labels(experiment_name=config.name, status='started').inc()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start experiment: {e}")
            return False
    
    async def log_metrics(self, experiment_id: str, epoch: int, metrics: Dict[str, float]):
        """Log metrics for an experiment"""
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.active_experiments[experiment_id]
            config = experiment['config']
            
            # Add timestamp
            timestamp = datetime.now()
            
            # Store metrics
            metric_record = {
                'experiment_id': experiment_id,
                'epoch': epoch,
                'timestamp': timestamp,
                'metrics': metrics
            }
            
            experiment['metrics_history'].append(metric_record)
            
            # Log to MLflow
            if 'mlflow_run_id' in experiment:
                with mlflow.start_run(run_id=experiment['mlflow_run_id']):
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value, step=epoch)
            
            # Log to W&B
            if self.wandb_enabled:
                wandb.log(metrics, step=epoch)
            
            # Store in InfluxDB for time-series analysis
            await self._store_metrics_influxdb(experiment_id, epoch, metrics, timestamp)
            
            # Store in Redis for real-time access
            await self._cache_latest_metrics(experiment_id, metrics)
            
            # Update Prometheus metrics
            for metric_name, metric_value in metrics.items():
                MODEL_PERFORMANCE_SCORE.labels(
                    model_name=config.name,
                    metric=metric_name
                ).set(metric_value)
            
            # Check for alerts
            await self._check_metric_alerts(experiment_id, metrics)
            
            # Notify WebSocket clients
            await self._notify_websocket_clients(experiment_id, metric_record)
            
            logger.debug(f"Metrics logged for experiment {experiment_id}, epoch {epoch}")
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            raise
    
    async def complete_experiment(self, experiment_id: str, final_metrics: Dict[str, float] = None):
        """Complete an experiment"""
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.active_experiments[experiment_id]
            config = experiment['config']
            
            # Update status
            experiment['status'] = ExperimentStatus.COMPLETED
            experiment['end_time'] = datetime.now()
            
            # Calculate duration
            if experiment['start_time']:
                duration = (experiment['end_time'] - experiment['start_time']).total_seconds()
                EXPERIMENT_DURATION.labels(experiment_name=config.name).observe(duration)
            
            # Log final metrics if provided
            if final_metrics:
                await self.log_metrics(experiment_id, -1, final_metrics)
            
            # End MLflow run
            if 'mlflow_run_id' in experiment:
                mlflow.end_run()
            
            # Finish W&B run
            if self.wandb_enabled:
                wandb.finish()
            
            # Update database
            await self._update_experiment_status(experiment_id, ExperimentStatus.COMPLETED)
            
            # Generate experiment report
            report = await self._generate_experiment_report(experiment_id)
            
            # Send completion notification
            await self._send_experiment_notification(experiment_id, "completed", report)
            
            logger.info(f"Experiment completed: {experiment_id}")
            EXPERIMENT_RUNS_TOTAL.labels(experiment_name=config.name, status='completed').inc()
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to complete experiment: {e}")
            raise
    
    async def get_experiment_metrics(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment metrics and history"""
        try:
            if experiment_id not in self.active_experiments:
                # Try to load from database
                experiment_data = await self._load_experiment_from_db(experiment_id)
                if not experiment_data:
                    raise ValueError(f"Experiment {experiment_id} not found")
            else:
                experiment_data = self.active_experiments[experiment_id]
            
            # Get metrics history from InfluxDB
            metrics_history = await self._get_metrics_from_influxdb(experiment_id)
            
            # Calculate statistics
            stats = await self._calculate_experiment_statistics(metrics_history)
            
            return {
                'experiment_id': experiment_id,
                'config': asdict(experiment_data['config']),
                'status': experiment_data['status'].value,
                'metrics_history': metrics_history,
                'statistics': stats,
                'duration_seconds': self._calculate_duration(experiment_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to get experiment metrics: {e}")
            raise
    
    # Private helper methods
    
    async def _store_experiment_config(self, config: ExperimentConfig):
        """Store experiment configuration in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO experiments (
                    experiment_id, name, description, model_type, dataset_path,
                    hyperparameters, metrics_to_track, created_at, status, tags
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
                config.experiment_id,
                config.name,
                config.description,
                config.model_type,
                config.dataset_path,
                json.dumps(config.hyperparameters),
                json.dumps(config.metrics_to_track),
                datetime.now(),
                ExperimentStatus.QUEUED.value,
                json.dumps(config.tags)
            )
    
    async def _create_mlflow_experiment(self, config: ExperimentConfig) -> str:
        """Create MLflow experiment"""
        try:
            experiment_name = f"{config.model_type}_{config.name}"
            
            try:
                experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
                return experiment.experiment_id
            except:
                return self.mlflow_client.create_experiment(experiment_name)
                
        except Exception as e:
            logger.warning(f"MLflow experiment creation failed: {e}")
            return None
    
    async def _create_wandb_experiment(self, config: ExperimentConfig):
        """Create W&B experiment"""
        try:
            wandb.init(
                project=config.name,
                config=config.hyperparameters,
                tags=config.tags,
                mode="offline"  # Start offline, will sync later
            )
            wandb.finish()
        except Exception as e:
            logger.warning(f"W&B experiment creation failed: {e}")
    
    async def _store_metrics_influxdb(self, experiment_id: str, epoch: int, 
                                    metrics: Dict[str, float], timestamp: datetime):
        """Store metrics in InfluxDB"""
        try:
            points = []
            for metric_name, metric_value in metrics.items():
                point = (
                    influxdb_client.Point("experiment_metrics")
                    .tag("experiment_id", experiment_id)
                    .tag("metric_name", metric_name)
                    .field("value", float(metric_value))
                    .field("epoch", epoch)
                    .time(timestamp)
                )
                points.append(point)
            
            self.influx_write_api.write(
                bucket=os.getenv('INFLUXDB_BUCKET', 'ml_metrics'),
                record=points
            )
            
        except Exception as e:
            logger.warning(f"InfluxDB write failed: {e}")
    
    async def _cache_latest_metrics(self, experiment_id: str, metrics: Dict[str, float]):
        """Cache latest metrics in Redis"""
        try:
            cache_key = f"experiment_metrics:{experiment_id}"
            await self.redis_client.setex(
                cache_key,
                300,  # 5 minutes TTL
                json.dumps(metrics)
            )
        except Exception as e:
            logger.warning(f"Redis cache failed: {e}")
    
    async def _check_metric_alerts(self, experiment_id: str, metrics: Dict[str, float]):
        """Check metrics against alert thresholds"""
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            return
        
        config = experiment['config']
        
        # Example alert conditions
        for metric_name, metric_value in metrics.items():
            if metric_name == 'loss' and metric_value > 1.0:
                await self._create_alert(
                    alert_type='high_loss',
                    severity=AlertSeverity.MEDIUM,
                    message=f"High loss detected: {metric_value:.4f}",
                    model_name=config.name,
                    experiment_id=experiment_id,
                    metrics={'loss': metric_value}
                )
            
            elif metric_name == 'accuracy' and metric_value < 0.5:
                await self._create_alert(
                    alert_type='low_accuracy',
                    severity=AlertSeverity.HIGH,
                    message=f"Low accuracy detected: {metric_value:.4f}",
                    model_name=config.name,
                    experiment_id=experiment_id,
                    metrics={'accuracy': metric_value}
                )
    
    async def _notify_websocket_clients(self, experiment_id: str, metric_record: Dict[str, Any]):
        """Notify WebSocket clients of metric updates"""
        if experiment_id in self.websocket_connections:
            message = {
                'type': 'metric_update',
                'experiment_id': experiment_id,
                'data': metric_record
            }
            
            # Send to all connected clients for this experiment
            for websocket in self.websocket_connections[experiment_id]:
                try:
                    await websocket.send_json(message)
                except:
                    # Remove disconnected client
                    self.websocket_connections[experiment_id].remove(websocket)


class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self, 
                 db_pool: asyncpg.Pool,
                 redis_client: aioredis.Redis,
                 influx_client: influxdb_client.InfluxDBClient):
        
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.influx_client = influx_client
        self.influx_write_api = influx_client.write_api(write_options=SYNCHRONOUS)
        
        # Monitoring configuration
        self.monitoring_interval = 30  # seconds
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Performance baselines
        self.performance_baselines = {}
        
        # Alert thresholds
        self.alert_thresholds = {
            'latency_ms': 1000,
            'error_rate': 0.05,
            'cpu_usage_percent': 80,
            'memory_usage_mb': 8192,
            'drift_score': 0.3
        }
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def collect_performance_metrics(self, model_name: str, model_version: str) -> PerformanceMetrics:
        """Collect performance metrics for a model"""
        try:
            # Get system metrics
            cpu_usage = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            memory_usage_mb = memory_info.used / 1024 / 1024
            
            # Get model-specific metrics from Redis cache
            metrics_key = f"model_metrics:{model_name}:{model_version}"
            cached_metrics = await self.redis_client.get(metrics_key)
            
            if cached_metrics:
                model_metrics = json.loads(cached_metrics)
            else:
                model_metrics = {}
            
            # Create performance metrics object
            performance_metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                model_name=model_name,
                model_version=model_version,
                metrics=model_metrics,
                latency_ms=model_metrics.get('latency_ms', 0),
                throughput_qps=model_metrics.get('throughput_qps', 0),
                error_rate=model_metrics.get('error_rate', 0),
                memory_usage_mb=memory_usage_mb,
                cpu_usage_percent=cpu_usage
            )
            
            # Store in InfluxDB
            await self._store_performance_metrics(performance_metrics)
            
            # Update Prometheus metrics
            SYSTEM_RESOURCE_USAGE.labels(resource_type='cpu').set(cpu_usage)
            SYSTEM_RESOURCE_USAGE.labels(resource_type='memory').set(memory_usage_mb)
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
            raise
    
    async def detect_drift(self, model_name: str, reference_data: pd.DataFrame, 
                          current_data: pd.DataFrame) -> DriftDetectionResult:
        """Detect data/concept drift"""
        try:
            # Perform statistical tests for drift detection
            drift_scores = []
            affected_features = []
            p_values = []
            
            for column in reference_data.columns:
                if column in current_data.columns:
                    # Kolmogorov-Smirnov test for numerical features
                    if pd.api.types.is_numeric_dtype(reference_data[column]):
                        ks_stat, p_value = stats.ks_2samp(
                            reference_data[column].dropna(),
                            current_data[column].dropna()
                        )
                        drift_scores.append(ks_stat)
                        p_values.append(p_value)
                        
                        if p_value < 0.05:  # Significant drift
                            affected_features.append(column)
                    
                    # Chi-square test for categorical features
                    elif pd.api.types.is_categorical_dtype(reference_data[column]) or \
                         pd.api.types.is_object_dtype(reference_data[column]):
                        
                        # Create contingency table
                        ref_counts = reference_data[column].value_counts()
                        curr_counts = current_data[column].value_counts()
                        
                        # Align indices
                        all_categories = set(ref_counts.index) | set(curr_counts.index)
                        ref_aligned = ref_counts.reindex(all_categories, fill_value=0)
                        curr_aligned = curr_counts.reindex(all_categories, fill_value=0)
                        
                        if len(all_categories) > 1:
                            chi2_stat, p_value = stats.chisquare(curr_aligned, ref_aligned)
                            drift_scores.append(chi2_stat / len(all_categories))  # Normalized
                            p_values.append(p_value)
                            
                            if p_value < 0.05:
                                affected_features.append(column)
            
            # Calculate overall drift score
            overall_drift_score = np.mean(drift_scores) if drift_scores else 0
            overall_p_value = np.mean(p_values) if p_values else 1
            
            # Determine if drift is detected
            threshold = self.alert_thresholds['drift_score']
            is_drift_detected = overall_drift_score > threshold
            
            drift_result = DriftDetectionResult(
                timestamp=datetime.now(),
                model_name=model_name,
                drift_type=DriftType.DATA_DRIFT,
                drift_score=overall_drift_score,
                threshold=threshold,
                is_drift_detected=is_drift_detected,
                affected_features=affected_features,
                statistical_test="ks_test_chi2",
                p_value=overall_p_value,
                confidence_interval=(0.95, 0.99)  # Would calculate proper CI
            )
            
            # Store drift detection result
            await self._store_drift_result(drift_result)
            
            # Update Prometheus metrics
            DRIFT_DETECTION_SCORE.labels(
                model_name=model_name,
                drift_type=DriftType.DATA_DRIFT.value
            ).set(overall_drift_score)
            
            # Create alert if drift detected
            if is_drift_detected:
                await self._create_drift_alert(drift_result)
            
            return drift_result
            
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            raise
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Get list of deployed models
                deployed_models = await self._get_deployed_models()
                
                for model_info in deployed_models:
                    model_name = model_info['name']
                    model_version = model_info['version']
                    
                    # Collect performance metrics
                    metrics = await self.collect_performance_metrics(model_name, model_version)
                    
                    # Check for performance alerts
                    await self._check_performance_alerts(metrics)
                
                # Sleep until next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _store_performance_metrics(self, metrics: PerformanceMetrics):
        """Store performance metrics in InfluxDB"""
        try:
            point = (
                influxdb_client.Point("performance_metrics")
                .tag("model_name", metrics.model_name)
                .tag("model_version", metrics.model_version)
                .field("latency_ms", metrics.latency_ms)
                .field("throughput_qps", metrics.throughput_qps)
                .field("error_rate", metrics.error_rate)
                .field("memory_usage_mb", metrics.memory_usage_mb)
                .field("cpu_usage_percent", metrics.cpu_usage_percent)
                .time(metrics.timestamp)
            )
            
            self.influx_write_api.write(
                bucket=os.getenv('INFLUXDB_BUCKET', 'ml_metrics'),
                record=point
            )
            
        except Exception as e:
            logger.warning(f"Performance metrics storage failed: {e}")
    
    async def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check performance metrics against thresholds"""
        alerts = []
        
        if metrics.latency_ms > self.alert_thresholds['latency_ms']:
            alerts.append({
                'type': 'high_latency',
                'severity': AlertSeverity.HIGH,
                'message': f"High latency: {metrics.latency_ms:.1f}ms",
                'value': metrics.latency_ms
            })
        
        if metrics.error_rate > self.alert_thresholds['error_rate']:
            alerts.append({
                'type': 'high_error_rate',
                'severity': AlertSeverity.CRITICAL,
                'message': f"High error rate: {metrics.error_rate:.2%}",
                'value': metrics.error_rate
            })
        
        if metrics.cpu_usage_percent > self.alert_thresholds['cpu_usage_percent']:
            alerts.append({
                'type': 'high_cpu_usage',
                'severity': AlertSeverity.MEDIUM,
                'message': f"High CPU usage: {metrics.cpu_usage_percent:.1f}%",
                'value': metrics.cpu_usage_percent
            })
        
        if metrics.memory_usage_mb > self.alert_thresholds['memory_usage_mb']:
            alerts.append({
                'type': 'high_memory_usage',
                'severity': AlertSeverity.MEDIUM,
                'message': f"High memory usage: {metrics.memory_usage_mb:.1f}MB",
                'value': metrics.memory_usage_mb
            })
        
        # Create alerts
        for alert_info in alerts:
            await self._create_performance_alert(metrics, alert_info)


class AlertManager:
    """Alert management system"""
    
    def __init__(self, 
                 db_pool: asyncpg.Pool,
                 redis_client: aioredis.Redis):
        
        self.db_pool = db_pool
        self.redis_client = redis_client
        
        # Notification channels
        self.email_config = {
            'smtp_server': os.getenv('SMTP_SERVER', 'localhost'),
            'smtp_port': int(os.getenv('SMTP_PORT', 587)),
            'username': os.getenv('SMTP_USERNAME'),
            'password': os.getenv('SMTP_PASSWORD'),
            'from_email': os.getenv('SMTP_FROM_EMAIL', 'alerts@mlops.com')
        }
        
        self.slack_client = None
        if os.getenv('SLACK_BOT_TOKEN'):
            self.slack_client = WebClient(token=os.getenv('SLACK_BOT_TOKEN'))
    
    async def create_alert(self, alert: Alert):
        """Create and process an alert"""
        try:
            # Store alert in database
            await self._store_alert(alert)
            
            # Cache in Redis for quick access
            await self._cache_alert(alert)
            
            # Send notifications based on severity
            await self._send_notifications(alert)
            
            # Update metrics
            MONITORING_ALERTS_TOTAL.labels(
                alert_type=alert.alert_type,
                severity=alert.severity.value
            ).inc()
            
            logger.info(f"Alert created: {alert.alert_id} - {alert.message}")
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            raise
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        try:
            # Update database
            async with self.db_pool.acquire() as conn:
                result = await conn.execute("""
                    UPDATE alerts 
                    SET is_resolved = TRUE, resolved_at = $1 
                    WHERE alert_id = $2
                """, datetime.now(), alert_id)
            
            if result == "UPDATE 1":
                # Update cache
                await self.redis_client.delete(f"alert:{alert_id}")
                logger.info(f"Alert resolved: {alert_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
            return False
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM alerts 
                    WHERE is_resolved = FALSE 
                    ORDER BY timestamp DESC
                """)
            
            alerts = []
            for row in rows:
                alert = Alert(
                    alert_id=row['alert_id'],
                    timestamp=row['timestamp'],
                    alert_type=row['alert_type'],
                    severity=AlertSeverity(row['severity']),
                    message=row['message'],
                    model_name=row['model_name'],
                    experiment_id=row['experiment_id'],
                    metrics=json.loads(row['metrics']),
                    is_resolved=row['is_resolved'],
                    resolved_at=row['resolved_at']
                )
                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to get active alerts: {e}")
            return []
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        try:
            # Send email for high/critical alerts
            if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                await self._send_email_notification(alert)
            
            # Send Slack notification
            if self.slack_client:
                await self._send_slack_notification(alert)
                
        except Exception as e:
            logger.warning(f"Notification sending failed: {e}")
    
    async def _send_email_notification(self, alert: Alert):
        """Send email notification"""
        try:
            if not self.email_config['username']:
                return
            
            subject = f"[{alert.severity.value.upper()}] ML Alert: {alert.alert_type}"
            
            body = f"""
            Alert Details:
            - Type: {alert.alert_type}
            - Severity: {alert.severity.value}
            - Model: {alert.model_name}
            - Message: {alert.message}
            - Timestamp: {alert.timestamp}
            
            Metrics: {json.dumps(alert.metrics, indent=2)}
            """
            
            msg = MimeMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = os.getenv('ALERT_EMAIL_RECIPIENTS', 'admin@mlops.com')
            msg['Subject'] = subject
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.warning(f"Email notification failed: {e}")
    
    async def _send_slack_notification(self, alert: Alert):
        """Send Slack notification"""
        try:
            if not self.slack_client:
                return
            
            color = {
                AlertSeverity.LOW: "good",
                AlertSeverity.MEDIUM: "warning", 
                AlertSeverity.HIGH: "danger",
                AlertSeverity.CRITICAL: "danger"
            }.get(alert.severity, "warning")
            
            message = {
                "channel": os.getenv('SLACK_ALERTS_CHANNEL', '#ml-alerts'),
                "attachments": [{
                    "color": color,
                    "title": f"{alert.severity.value.upper()} Alert: {alert.alert_type}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Model", "value": alert.model_name, "short": True},
                        {"title": "Timestamp", "value": str(alert.timestamp), "short": True}
                    ]
                }]
            }
            
            self.slack_client.chat_postMessage(**message)
            
        except Exception as e:
            logger.warning(f"Slack notification failed: {e}")


# FastAPI application
app = FastAPI(
    title="Experiment Tracking and Performance Monitoring",
    description="Comprehensive experiment tracking and model performance monitoring",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances
experiment_tracker: ExperimentTracker = None
performance_monitor: PerformanceMonitor = None
alert_manager: AlertManager = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global experiment_tracker, performance_monitor, alert_manager
    
    # Initialize database connections
    db_pool = await asyncpg.create_pool(
        os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/mlops')
    )
    
    redis_client = aioredis.from_url(
        os.getenv('REDIS_URL', 'redis://localhost:6379')
    )
    
    # Initialize InfluxDB client
    influx_client = influxdb_client.InfluxDBClient(
        url=os.getenv('INFLUXDB_URL', 'http://localhost:8086'),
        token=os.getenv('INFLUXDB_TOKEN'),
        org=os.getenv('INFLUXDB_ORG', 'mlops')
    )
    
    mlflow_client = MlflowClient(
        tracking_uri=os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    )
    
    # Initialize services
    experiment_tracker = ExperimentTracker(db_pool, redis_client, influx_client, mlflow_client)
    performance_monitor = PerformanceMonitor(db_pool, redis_client, influx_client)
    alert_manager = AlertManager(db_pool, redis_client)
    
    # Start performance monitoring
    await performance_monitor.start_monitoring()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if performance_monitor:
        await performance_monitor.stop_monitoring()

# API endpoints
@app.post("/experiments")
async def create_experiment(request: ExperimentCreateRequest):
    """Create a new experiment"""
    config = ExperimentConfig(
        experiment_id=str(uuid.uuid4()),
        name=request.name,
        description=request.description,
        model_type=request.model_type,
        dataset_path=request.dataset_path,
        hyperparameters=request.hyperparameters,
        metrics_to_track=request.metrics_to_track,
        tags=request.tags
    )
    
    experiment_id = await experiment_tracker.create_experiment(config)
    return {"experiment_id": experiment_id}

@app.post("/experiments/{experiment_id}/start")
async def start_experiment(experiment_id: str):
    """Start an experiment"""
    success = await experiment_tracker.start_experiment(experiment_id)
    return {"success": success}

@app.post("/experiments/{experiment_id}/metrics")
async def log_metrics(experiment_id: str, update: MetricUpdate):
    """Log metrics for an experiment"""
    await experiment_tracker.log_metrics(experiment_id, update.epoch, update.metrics)
    return {"status": "logged"}

@app.post("/experiments/{experiment_id}/complete")
async def complete_experiment(experiment_id: str, final_metrics: Dict[str, float] = None):
    """Complete an experiment"""
    report = await experiment_tracker.complete_experiment(experiment_id, final_metrics)
    return {"report": report}

@app.get("/experiments/{experiment_id}")
async def get_experiment_metrics(experiment_id: str):
    """Get experiment metrics and history"""
    metrics = await experiment_tracker.get_experiment_metrics(experiment_id)
    return metrics

@app.get("/alerts")
async def get_active_alerts():
    """Get all active alerts"""
    alerts = await alert_manager.get_active_alerts()
    return {"alerts": [asdict(alert) for alert in alerts]}

@app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an alert"""
    success = await alert_manager.resolve_alert(alert_id)
    return {"success": success}

@app.get("/metrics")
async def get_prometheus_metrics():
    """Get Prometheus metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.websocket("/experiments/{experiment_id}/live")
async def experiment_live_updates(websocket: WebSocket, experiment_id: str):
    """WebSocket endpoint for live experiment updates"""
    await websocket.accept()
    
    # Add to connections
    if experiment_id not in experiment_tracker.websocket_connections:
        experiment_tracker.websocket_connections[experiment_id] = []
    experiment_tracker.websocket_connections[experiment_id].append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        # Remove from connections
        if experiment_id in experiment_tracker.websocket_connections:
            experiment_tracker.websocket_connections[experiment_id].remove(websocket)

if __name__ == "__main__":
    uvicorn.run(
        "experiment_monitoring:app",
        host="0.0.0.0",
        port=8081,
        reload=False
    )
