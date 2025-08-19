"""
Automated model training and evaluation pipelines
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
import yaml
from pathlib import Path
import shutil
import subprocess

# Data processing and ML
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

# Deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# MLflow and experiment tracking
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

# Workflow orchestration
try:
    import airflow
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

try:
    import prefect
    from prefect import task, flow, get_run_logger
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False

# Kubernetes and containerization
import kubernetes
from kubernetes import client, config
import docker

# Database and storage
import asyncpg
import aioredis
from motor.motor_asyncio import AsyncIOMotorClient
import boto3

# Monitoring and observability
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Configuration management
from omegaconf import DictConfig, OmegaConf
import hydra

# Custom imports
import sys
sys.path.append('../../../fraud-detection-service/src/main/python')
sys.path.append('../../../ml-platform/src/fraud_detection')

from data_processor import FraudDataProcessor
from models import ModelFactory, ModelEvaluator
from pipeline import FraudDetectionPipeline

# Configure logging
logger = structlog.get_logger()

# Prometheus metrics
TRAINING_JOBS_TOTAL = Counter('training_jobs_total', 'Total training jobs', ['job_type', 'status'])
TRAINING_DURATION = Histogram('training_duration_seconds', 'Training duration', ['model_type'])
MODEL_ACCURACY = Gauge('model_accuracy_score', 'Model accuracy score', ['model_name', 'dataset'])
PIPELINE_EXECUTION_COUNT = Counter('pipeline_executions_total', 'Pipeline executions', ['pipeline_name', 'status'])


class JobStatus(Enum):
    """Training job status enumeration"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class PipelineType(Enum):
    """Pipeline type enumeration"""
    TRAINING = "training"
    EVALUATION = "evaluation"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    DATA_VALIDATION = "data_validation"
    MODEL_VALIDATION = "model_validation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


class TriggerType(Enum):
    """Pipeline trigger type enumeration"""
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    DATA_CHANGE = "data_change"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MODEL_DRIFT = "model_drift"


@dataclass
class TrainingJobConfig:
    """Training job configuration"""
    job_id: str
    job_name: str
    model_type: str
    dataset_path: str
    target_column: str
    feature_columns: List[str]
    hyperparameters: Dict[str, Any]
    validation_split: float = 0.2
    test_split: float = 0.2
    cross_validation_folds: int = 5
    early_stopping: bool = True
    max_epochs: int = 100
    model_selection_metric: str = "accuracy"
    resource_requirements: Dict[str, str] = None
    notification_config: Dict[str, Any] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.resource_requirements is None:
            self.resource_requirements = {
                "cpu": "2",
                "memory": "4Gi",
                "gpu": "0"
            }
        if self.notification_config is None:
            self.notification_config = {}
        if self.tags is None:
            self.tags = []


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    pipeline_id: str
    pipeline_name: str
    pipeline_type: PipelineType
    trigger_type: TriggerType
    schedule: Optional[str] = None  # Cron expression
    tasks: List[Dict[str, Any]] = None
    dependencies: List[str] = None
    retry_count: int = 3
    timeout_minutes: int = 60
    notification_on_failure: bool = True
    notification_on_success: bool = False
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = []
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ModelEvaluationResult:
    """Model evaluation result"""
    model_name: str
    model_version: str
    dataset_name: str
    evaluation_timestamp: datetime
    metrics: Dict[str, float]
    confusion_matrix: List[List[int]]
    feature_importance: Dict[str, float]
    model_size_mb: float
    inference_latency_ms: float
    passed_validation: bool
    validation_errors: List[str]


class AutomatedTrainingPipeline:
    """Automated model training pipeline"""
    
    def __init__(self, 
                 db_pool: asyncpg.Pool,
                 redis_client: aioredis.Redis,
                 s3_client: boto3.client,
                 mlflow_client: MlflowClient):
        
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.s3_client = s3_client
        self.mlflow_client = mlflow_client
        
        # Configuration
        self.data_bucket = os.getenv('DATA_BUCKET', 'ml-data-bucket')
        self.model_bucket = os.getenv('MODEL_BUCKET', 'ml-models-bucket')
        self.artifact_path = os.getenv('ARTIFACT_PATH', '/tmp/ml_artifacts')
        
        # Active jobs tracking
        self.active_jobs = {}
        self.job_queue = asyncio.Queue()
        self.worker_tasks = []
        self.max_concurrent_jobs = int(os.getenv('MAX_CONCURRENT_JOBS', '3'))
        
        # Pipeline templates
        self.pipeline_templates = self._load_pipeline_templates()
        
        # Kubernetes client for job execution
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        self.k8s_batch_v1 = client.BatchV1Api()
        self.k8s_core_v1 = client.CoreV1Api()
        
        # Docker client for building images
        try:
            self.docker_client = docker.from_env()
        except:
            self.docker_client = None
            logger.warning("Docker client not available")
    
    async def start_workers(self):
        """Start background workers for job processing"""
        for i in range(self.max_concurrent_jobs):
            task = asyncio.create_task(self._job_worker(f"worker_{i}"))
            self.worker_tasks.append(task)
        
        logger.info(f"Started {self.max_concurrent_jobs} training workers")
    
    async def stop_workers(self):
        """Stop background workers"""
        for task in self.worker_tasks:
            task.cancel()
        
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        logger.info("Training workers stopped")
    
    async def submit_training_job(self, config: TrainingJobConfig) -> str:
        """Submit a training job to the queue"""
        try:
            # Validate configuration
            await self._validate_training_config(config)
            
            # Store job configuration
            await self._store_job_config(config)
            
            # Add to queue
            await self.job_queue.put(config)
            
            # Update active jobs
            self.active_jobs[config.job_id] = {
                'config': config,
                'status': JobStatus.QUEUED,
                'submitted_at': datetime.now(),
                'started_at': None,
                'completed_at': None,
                'worker_id': None,
                'error_message': None
            }
            
            logger.info(f"Training job submitted: {config.job_id}")
            TRAINING_JOBS_TOTAL.labels(job_type='training', status='submitted').inc()
            
            return config.job_id
            
        except Exception as e:
            logger.error(f"Failed to submit training job: {e}")
            raise
    
    async def execute_training_job(self, config: TrainingJobConfig) -> ModelEvaluationResult:
        """Execute a single training job"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting training job: {config.job_id}")
            
            # Update job status
            await self._update_job_status(config.job_id, JobStatus.RUNNING)
            
            # Load and preprocess data
            X, y = await self._load_and_preprocess_data(config)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=config.test_split,
                random_state=42,
                stratify=y
            )
            
            # Further split training data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=config.validation_split,
                random_state=42,
                stratify=y_train
            )
            
            # Start MLflow run
            with mlflow.start_run(run_name=f"{config.job_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
                
                # Log parameters
                mlflow.log_params(config.hyperparameters)
                mlflow.log_param("model_type", config.model_type)
                mlflow.log_param("dataset_path", config.dataset_path)
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("val_samples", len(X_val))
                mlflow.log_param("test_samples", len(X_test))
                
                # Train model
                model, training_history = await self._train_model(
                    config, X_train, y_train, X_val, y_val
                )
                
                # Evaluate model
                evaluation_result = await self._evaluate_model(
                    model, X_test, y_test, config
                )
                
                # Log metrics to MLflow
                for metric_name, metric_value in evaluation_result.metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model
                model_path = await self._save_model(model, config, run.info.run_id)
                
                if config.model_type in ['sklearn', 'xgboost', 'lightgbm']:
                    mlflow.sklearn.log_model(model, "model")
                elif config.model_type == 'tensorflow':
                    mlflow.tensorflow.log_model(model, "model")
                
                # Save artifacts
                await self._save_training_artifacts(config, evaluation_result, training_history)
                
                # Update job status
                await self._update_job_status(config.job_id, JobStatus.COMPLETED)
                
                duration = time.time() - start_time
                TRAINING_DURATION.labels(model_type=config.model_type).observe(duration)
                TRAINING_JOBS_TOTAL.labels(job_type='training', status='completed').inc()
                
                # Update model accuracy metric
                MODEL_ACCURACY.labels(
                    model_name=config.job_name,
                    dataset=os.path.basename(config.dataset_path)
                ).set(evaluation_result.metrics.get('accuracy', 0))
                
                logger.info(f"Training job completed: {config.job_id} in {duration:.2f}s")
                
                return evaluation_result
                
        except Exception as e:
            logger.error(f"Training job failed: {config.job_id} - {e}")
            
            await self._update_job_status(config.job_id, JobStatus.FAILED, str(e))
            TRAINING_JOBS_TOTAL.labels(job_type='training', status='failed').inc()
            
            raise
    
    async def create_hyperparameter_tuning_job(self, 
                                             base_config: TrainingJobConfig,
                                             param_grid: Dict[str, List[Any]],
                                             search_strategy: str = "random",
                                             n_trials: int = 20) -> str:
        """Create hyperparameter tuning job"""
        try:
            tuning_job_id = f"hp_tuning_{uuid.uuid4().hex[:8]}"
            
            # Generate parameter combinations
            if search_strategy == "grid":
                param_combinations = self._generate_grid_search_params(param_grid)
            else:  # random
                param_combinations = self._generate_random_search_params(param_grid, n_trials)
            
            # Create individual training jobs for each combination
            job_ids = []
            for i, params in enumerate(param_combinations):
                job_config = TrainingJobConfig(
                    job_id=f"{tuning_job_id}_trial_{i:03d}",
                    job_name=f"{base_config.job_name}_hp_trial_{i:03d}",
                    model_type=base_config.model_type,
                    dataset_path=base_config.dataset_path,
                    target_column=base_config.target_column,
                    feature_columns=base_config.feature_columns,
                    hyperparameters=params,
                    validation_split=base_config.validation_split,
                    test_split=base_config.test_split,
                    model_selection_metric=base_config.model_selection_metric,
                    tags=base_config.tags + ["hyperparameter_tuning", tuning_job_id]
                )
                
                job_id = await self.submit_training_job(job_config)
                job_ids.append(job_id)
            
            # Store tuning job metadata
            tuning_metadata = {
                'tuning_job_id': tuning_job_id,
                'base_config': asdict(base_config),
                'param_grid': param_grid,
                'search_strategy': search_strategy,
                'n_trials': n_trials,
                'job_ids': job_ids,
                'created_at': datetime.now().isoformat()
            }
            
            await self.redis_client.setex(
                f"hp_tuning:{tuning_job_id}",
                86400,  # 24 hours
                json.dumps(tuning_metadata, default=str)
            )
            
            logger.info(f"Hyperparameter tuning job created: {tuning_job_id} with {len(job_ids)} trials")
            
            return tuning_job_id
            
        except Exception as e:
            logger.error(f"Failed to create hyperparameter tuning job: {e}")
            raise
    
    async def get_best_model_from_tuning(self, tuning_job_id: str) -> Dict[str, Any]:
        """Get best model from hyperparameter tuning job"""
        try:
            # Get tuning metadata
            metadata_json = await self.redis_client.get(f"hp_tuning:{tuning_job_id}")
            if not metadata_json:
                raise ValueError(f"Tuning job {tuning_job_id} not found")
            
            metadata = json.loads(metadata_json)
            job_ids = metadata['job_ids']
            
            # Get results from all jobs
            best_score = float('-inf')
            best_job_id = None
            best_metrics = None
            
            for job_id in job_ids:
                job_info = self.active_jobs.get(job_id)
                if job_info and job_info['status'] == JobStatus.COMPLETED:
                    # Get job results from database
                    job_results = await self._get_job_results(job_id)
                    if job_results:
                        metric_name = metadata['base_config']['model_selection_metric']
                        score = job_results['metrics'].get(metric_name, 0)
                        
                        if score > best_score:
                            best_score = score
                            best_job_id = job_id
                            best_metrics = job_results
            
            if best_job_id:
                return {
                    'tuning_job_id': tuning_job_id,
                    'best_job_id': best_job_id,
                    'best_score': best_score,
                    'best_metrics': best_metrics,
                    'total_trials': len(job_ids),
                    'completed_trials': sum(1 for job_id in job_ids 
                                          if self.active_jobs.get(job_id, {}).get('status') == JobStatus.COMPLETED)
                }
            else:
                raise ValueError("No completed trials found")
                
        except Exception as e:
            logger.error(f"Failed to get best model from tuning: {e}")
            raise
    
    async def create_automated_retraining_pipeline(self, 
                                                 base_config: TrainingJobConfig,
                                                 trigger_conditions: Dict[str, Any],
                                                 schedule: str = None) -> str:
        """Create automated retraining pipeline"""
        try:
            pipeline_id = f"retrain_pipeline_{uuid.uuid4().hex[:8]}"
            
            # Create pipeline configuration
            pipeline_config = PipelineConfig(
                pipeline_id=pipeline_id,
                pipeline_name=f"Automated Retraining - {base_config.job_name}",
                pipeline_type=PipelineType.TRAINING,
                trigger_type=TriggerType.SCHEDULED if schedule else TriggerType.PERFORMANCE_DEGRADATION,
                schedule=schedule,
                tasks=[
                    {
                        'name': 'data_validation',
                        'type': 'validation',
                        'config': {
                            'dataset_path': base_config.dataset_path,
                            'validation_rules': trigger_conditions.get('data_validation', {})
                        }
                    },
                    {
                        'name': 'model_training',
                        'type': 'training',
                        'config': asdict(base_config),
                        'depends_on': ['data_validation']
                    },
                    {
                        'name': 'model_evaluation',
                        'type': 'evaluation',
                        'config': {
                            'baseline_metrics': trigger_conditions.get('baseline_metrics', {}),
                            'performance_thresholds': trigger_conditions.get('performance_thresholds', {})
                        },
                        'depends_on': ['model_training']
                    },
                    {
                        'name': 'model_deployment',
                        'type': 'deployment',
                        'config': {
                            'deployment_strategy': 'canary',
                            'traffic_percentage': 10
                        },
                        'depends_on': ['model_evaluation']
                    }
                ]
            )
            
            # Store pipeline configuration
            await self._store_pipeline_config(pipeline_config)
            
            # Schedule pipeline if needed
            if schedule:
                await self._schedule_pipeline(pipeline_config)
            
            logger.info(f"Automated retraining pipeline created: {pipeline_id}")
            
            return pipeline_id
            
        except Exception as e:
            logger.error(f"Failed to create retraining pipeline: {e}")
            raise
    
    # Private helper methods
    
    async def _job_worker(self, worker_id: str):
        """Background worker for processing training jobs"""
        logger.info(f"Training worker {worker_id} started")
        
        while True:
            try:
                # Get job from queue
                config = await self.job_queue.get()
                
                # Update job info
                if config.job_id in self.active_jobs:
                    self.active_jobs[config.job_id]['worker_id'] = worker_id
                    self.active_jobs[config.job_id]['started_at'] = datetime.now()
                
                # Execute job
                try:
                    await self.execute_training_job(config)
                except Exception as e:
                    logger.error(f"Worker {worker_id} job execution failed: {e}")
                
                # Mark task as done
                self.job_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"Training worker {worker_id} stopped")
    
    async def _validate_training_config(self, config: TrainingJobConfig):
        """Validate training configuration"""
        # Check if dataset exists
        if config.dataset_path.startswith('s3://'):
            # Validate S3 path
            bucket, key = config.dataset_path[5:].split('/', 1)
            try:
                self.s3_client.head_object(Bucket=bucket, Key=key)
            except:
                raise ValueError(f"Dataset not found: {config.dataset_path}")
        else:
            # Validate local path
            if not os.path.exists(config.dataset_path):
                raise ValueError(f"Dataset not found: {config.dataset_path}")
        
        # Validate model type
        supported_models = ['sklearn', 'tensorflow', 'xgboost', 'lightgbm']
        if config.model_type not in supported_models:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        # Validate hyperparameters
        if not config.hyperparameters:
            raise ValueError("Hyperparameters cannot be empty")
    
    async def _load_and_preprocess_data(self, config: TrainingJobConfig) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and preprocess training data"""
        try:
            # Load data
            if config.dataset_path.startswith('s3://'):
                # Download from S3
                bucket, key = config.dataset_path[5:].split('/', 1)
                local_path = f"/tmp/{os.path.basename(key)}"
                self.s3_client.download_file(bucket, key, local_path)
                df = pd.read_csv(local_path)
                os.remove(local_path)
            else:
                df = pd.read_csv(config.dataset_path)
            
            # Extract features and target
            if config.feature_columns:
                X = df[config.feature_columns]
            else:
                X = df.drop(columns=[config.target_column])
            
            y = df[config.target_column]
            
            # Basic preprocessing
            processor = FraudDataProcessor()
            X_processed, y_processed = processor.fit_transform(
                pd.concat([X, y], axis=1),
                target_col=config.target_column
            )
            
            return X_processed, y_processed
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise
    
    async def _train_model(self, 
                          config: TrainingJobConfig,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_val: pd.DataFrame,
                          y_val: pd.Series) -> Tuple[Any, Dict[str, Any]]:
        """Train model based on configuration"""
        
        training_history = {}
        
        if config.model_type == 'sklearn':
            model = self._train_sklearn_model(config, X_train, y_train)
        elif config.model_type == 'tensorflow':
            model, training_history = self._train_tensorflow_model(
                config, X_train, y_train, X_val, y_val
            )
        elif config.model_type == 'xgboost':
            model = self._train_xgboost_model(config, X_train, y_train, X_val, y_val)
        elif config.model_type == 'lightgbm':
            model = self._train_lightgbm_model(config, X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        return model, training_history
    
    def _train_sklearn_model(self, config: TrainingJobConfig, X_train: pd.DataFrame, y_train: pd.Series):
        """Train scikit-learn model"""
        if 'algorithm' in config.hyperparameters:
            algorithm = config.hyperparameters.pop('algorithm')
        else:
            algorithm = 'random_forest'
        
        if algorithm == 'random_forest':
            model = RandomForestClassifier(**config.hyperparameters, random_state=42)
        elif algorithm == 'logistic_regression':
            model = LogisticRegression(**config.hyperparameters, random_state=42)
        else:
            raise ValueError(f"Unsupported sklearn algorithm: {algorithm}")
        
        model.fit(X_train, y_train)
        return model
    
    def _train_tensorflow_model(self, 
                               config: TrainingJobConfig,
                               X_train: pd.DataFrame,
                               y_train: pd.Series,
                               X_val: pd.DataFrame,
                               y_val: pd.Series) -> Tuple[tf.keras.Model, Dict[str, Any]]:
        """Train TensorFlow model"""
        
        # Build model
        model = keras.Sequential([
            keras.layers.Dense(
                config.hyperparameters.get('hidden_units_1', 128),
                activation='relu',
                input_shape=(X_train.shape[1],)
            ),
            keras.layers.Dropout(config.hyperparameters.get('dropout_1', 0.3)),
            keras.layers.Dense(
                config.hyperparameters.get('hidden_units_2', 64),
                activation='relu'
            ),
            keras.layers.Dropout(config.hyperparameters.get('dropout_2', 0.3)),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=config.hyperparameters.get('learning_rate', 0.001)
            ),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Callbacks
        callbacks = []
        
        if config.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor='val_loss',
                    patience=config.hyperparameters.get('patience', 10),
                    restore_best_weights=True
                )
            )
        
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        )
        
        # Train model
        history = model.fit(
            X_train.values, y_train.values,
            validation_data=(X_val.values, y_val.values),
            epochs=config.max_epochs,
            batch_size=config.hyperparameters.get('batch_size', 32),
            callbacks=callbacks,
            verbose=0
        )
        
        return model, history.history
    
    def _train_xgboost_model(self, 
                           config: TrainingJobConfig,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           X_val: pd.DataFrame,
                           y_val: pd.Series):
        """Train XGBoost model"""
        
        # Prepare data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            **config.hyperparameters
        }
        
        # Train model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=config.hyperparameters.get('n_estimators', 100),
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=config.hyperparameters.get('early_stopping_rounds', 10) if config.early_stopping else None,
            verbose_eval=False
        )
        
        return model
    
    def _train_lightgbm_model(self, 
                            config: TrainingJobConfig,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_val: pd.DataFrame,
                            y_val: pd.Series):
        """Train LightGBM model"""
        
        # Prepare data
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbose': -1,
            **config.hyperparameters
        }
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=config.hyperparameters.get('n_estimators', 100),
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            early_stopping_rounds=config.hyperparameters.get('early_stopping_rounds', 10) if config.early_stopping else None,
            verbose_eval=False
        )
        
        return model
    
    async def _evaluate_model(self, 
                            model: Any,
                            X_test: pd.DataFrame,
                            y_test: pd.Series,
                            config: TrainingJobConfig) -> ModelEvaluationResult:
        """Evaluate trained model"""
        
        start_time = time.time()
        
        # Make predictions
        if config.model_type == 'sklearn':
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        elif config.model_type == 'tensorflow':
            y_pred_proba = model.predict(X_test.values).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        elif config.model_type in ['xgboost', 'lightgbm']:
            if config.model_type == 'xgboost':
                dtest = xgb.DMatrix(X_test)
                y_pred_proba = model.predict(dtest)
            else:  # lightgbm
                y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
        
        inference_latency = (time.time() - start_time) * 1000 / len(X_test)  # ms per sample
        
        # Calculate metrics
        metrics = {
            'accuracy': float(np.mean(y_pred == y_test)),
            'precision': float(precision_score(y_test, y_pred, average='binary', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='binary', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='binary', zero_division=0))
        }
        
        if y_pred_proba is not None:
            metrics['auc'] = float(roc_auc_score(y_test, y_pred_proba))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_test.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(X_test.columns, np.abs(model.coef_[0])))
        
        # Model size (rough estimate)
        model_size_mb = 0
        try:
            import pickle
            model_size_mb = len(pickle.dumps(model)) / (1024 * 1024)
        except:
            pass
        
        # Validation checks
        passed_validation = True
        validation_errors = []
        
        # Basic validation rules
        if metrics['accuracy'] < 0.5:
            passed_validation = False
            validation_errors.append("Accuracy below 50%")
        
        if metrics.get('auc', 0) < 0.5:
            passed_validation = False
            validation_errors.append("AUC below 50%")
        
        return ModelEvaluationResult(
            model_name=config.job_name,
            model_version=config.job_id,
            dataset_name=os.path.basename(config.dataset_path),
            evaluation_timestamp=datetime.now(),
            metrics=metrics,
            confusion_matrix=cm.tolist(),
            feature_importance=feature_importance,
            model_size_mb=model_size_mb,
            inference_latency_ms=inference_latency,
            passed_validation=passed_validation,
            validation_errors=validation_errors
        )
    
    def _load_pipeline_templates(self) -> Dict[str, Any]:
        """Load pipeline templates"""
        templates_path = Path(__file__).parent / "pipeline_templates"
        templates = {}
        
        if templates_path.exists():
            for template_file in templates_path.glob("*.yaml"):
                with open(template_file, 'r') as f:
                    template_name = template_file.stem
                    templates[template_name] = yaml.safe_load(f)
        
        return templates
    
    def _generate_grid_search_params(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search"""
        from itertools import product
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combination in product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations
    
    def _generate_random_search_params(self, param_grid: Dict[str, List[Any]], n_trials: int) -> List[Dict[str, Any]]:
        """Generate random combinations for random search"""
        import random
        
        combinations = []
        for _ in range(n_trials):
            combination = {}
            for param_name, param_values in param_grid.items():
                combination[param_name] = random.choice(param_values)
            combinations.append(combination)
        
        return combinations
    
    async def _store_job_config(self, config: TrainingJobConfig):
        """Store job configuration in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO training_jobs (
                    job_id, job_name, model_type, dataset_path, target_column,
                    feature_columns, hyperparameters, created_at, status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
                config.job_id,
                config.job_name,
                config.model_type,
                config.dataset_path,
                config.target_column,
                json.dumps(config.feature_columns),
                json.dumps(config.hyperparameters),
                datetime.now(),
                JobStatus.QUEUED.value
            )
    
    async def _update_job_status(self, job_id: str, status: JobStatus, error_message: str = None):
        """Update job status in database and memory"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE training_jobs 
                SET status = $1, updated_at = $2, error_message = $3
                WHERE job_id = $4
            """, status.value, datetime.now(), error_message, job_id)
        
        if job_id in self.active_jobs:
            self.active_jobs[job_id]['status'] = status
            self.active_jobs[job_id]['error_message'] = error_message
            
            if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                self.active_jobs[job_id]['completed_at'] = datetime.now()
    
    async def _save_model(self, model: Any, config: TrainingJobConfig, run_id: str) -> str:
        """Save trained model"""
        # Create model directory
        model_dir = Path(self.artifact_path) / config.job_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.joblib"
        
        if config.model_type == 'tensorflow':
            model.save(str(model_dir / "model"))
            return str(model_dir / "model")
        else:
            import joblib
            joblib.dump(model, model_path)
            return str(model_path)
    
    async def _save_training_artifacts(self, 
                                     config: TrainingJobConfig,
                                     evaluation_result: ModelEvaluationResult,
                                     training_history: Dict[str, Any]):
        """Save training artifacts"""
        artifact_dir = Path(self.artifact_path) / config.job_id
        
        # Save evaluation results
        with open(artifact_dir / "evaluation_results.json", 'w') as f:
            json.dump(asdict(evaluation_result), f, indent=2, default=str)
        
        # Save training history
        if training_history:
            with open(artifact_dir / "training_history.json", 'w') as f:
                json.dump(training_history, f, indent=2, default=str)
        
        # Save configuration
        with open(artifact_dir / "config.json", 'w') as f:
            json.dump(asdict(config), f, indent=2, default=str)


# Factory function
async def create_training_pipeline(
    db_url: str = None,
    redis_url: str = None,
    mlflow_tracking_uri: str = None
) -> AutomatedTrainingPipeline:
    """Factory function to create training pipeline"""
    
    # Initialize database connections
    db_pool = await asyncpg.create_pool(
        db_url or os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/mlops')
    )
    
    redis_client = aioredis.from_url(
        redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379')
    )
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-west-2')
    )
    
    mlflow_client = MlflowClient(
        tracking_uri=mlflow_tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    )
    
    # Create pipeline
    pipeline = AutomatedTrainingPipeline(db_pool, redis_client, s3_client, mlflow_client)
    
    # Start workers
    await pipeline.start_workers()
    
    return pipeline


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create training pipeline
        pipeline = await create_training_pipeline()
        
        # Example training job
        config = TrainingJobConfig(
            job_id=f"fraud_training_{uuid.uuid4().hex[:8]}",
            job_name="fraud_detection_model",
            model_type="sklearn",
            dataset_path="s3://ml-data-bucket/fraud_data.csv",
            target_column="is_fraud",
            feature_columns=[],  # Use all columns except target
            hyperparameters={
                "algorithm": "random_forest",
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5
            }
        )
        
        # Submit job
        job_id = await pipeline.submit_training_job(config)
        print(f"Training job submitted: {job_id}")
        
        # Wait for completion
        await asyncio.sleep(60)  # In practice, you'd monitor job status
        
        # Stop workers
        await pipeline.stop_workers()
    
    asyncio.run(main())
