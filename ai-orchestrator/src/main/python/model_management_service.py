"""
AI Orchestrator - Model Management Service with versioning and deployment
"""

import os
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid
from pathlib import Path
import hashlib
import shutil

# FastAPI and web framework
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# Pydantic models
from pydantic import BaseModel, Field, validator
from pydantic.types import UUID4

# Database and storage
import asyncpg
import aioredis
from motor.motor_asyncio import AsyncIOMotorClient
import boto3
from botocore.exceptions import ClientError

# ML frameworks and model management
import mlflow
import mlflow.tracking
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import joblib
import pandas as pd
import numpy as np

# Kubernetes and deployment
import kubernetes
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Monitoring and observability
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Version control and Git
import git
from git import Repo

# Configuration and environment
from dotenv import load_dotenv

# Custom imports
import sys
sys.path.append('../../../fraud-detection-service/src/main/python')
from model_optimizer import ModelOptimizer, ModelConfig, OptimizationType

# Load environment variables
load_dotenv()

# Configure logging
logger = structlog.get_logger()

# Prometheus metrics
MODEL_DEPLOYMENTS_TOTAL = Counter('model_deployments_total', 'Total model deployments', ['model_name', 'version', 'status'])
MODEL_MANAGEMENT_REQUESTS = Counter('model_management_requests_total', 'Model management requests', ['operation'])
DEPLOYMENT_DURATION = Histogram('deployment_duration_seconds', 'Deployment duration', ['deployment_type'])
ACTIVE_MODELS = Gauge('active_models_total', 'Number of active models', ['environment'])
MODEL_HEALTH_STATUS = Gauge('model_health_status', 'Model health status', ['model_name', 'version'])


class ModelStatus(Enum):
    """Model status enumeration"""
    DRAFT = "draft"
    TRAINING = "training"
    VALIDATING = "validating"
    READY = "ready"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class DeploymentEnvironment(Enum):
    """Deployment environment enumeration"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"
    AB_TEST = "ab_test"


class ModelType(Enum):
    """Model type enumeration"""
    FRAUD_DETECTION = "fraud_detection"
    RISK_SCORING = "risk_scoring"
    ANOMALY_DETECTION = "anomaly_detection"
    CUSTOMER_SEGMENTATION = "customer_segmentation"
    CREDIT_SCORING = "credit_scoring"


@dataclass
class ModelMetadata:
    """Model metadata structure"""
    model_id: str
    name: str
    version: str
    model_type: ModelType
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    created_by: str
    description: str
    tags: List[str]
    performance_metrics: Dict[str, float]
    model_size_mb: float
    training_data_info: Dict[str, Any]
    dependencies: List[str]
    deployment_config: Dict[str, Any]


# Pydantic models for API
class ModelRegistrationRequest(BaseModel):
    """Model registration request"""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    model_type: ModelType = Field(..., description="Model type")
    description: str = Field("", description="Model description")
    tags: List[str] = Field(default_factory=list, description="Model tags")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    training_data_info: Dict[str, Any] = Field(default_factory=dict, description="Training data information")
    dependencies: List[str] = Field(default_factory=list, description="Model dependencies")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v) < 3:
            raise ValueError('Model name must be at least 3 characters')
        return v.lower().replace(' ', '_')
    
    @validator('version')
    def validate_version(cls, v):
        if not v:
            raise ValueError('Version is required')
        return v


class ModelDeploymentRequest(BaseModel):
    """Model deployment request"""
    model_id: str = Field(..., description="Model ID")
    environment: DeploymentEnvironment = Field(..., description="Deployment environment")
    replicas: int = Field(1, ge=1, le=10, description="Number of replicas")
    resources: Dict[str, str] = Field(default_factory=dict, description="Resource requirements")
    environment_variables: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    health_check_path: str = Field("/health", description="Health check path")
    auto_scaling: bool = Field(False, description="Enable auto-scaling")
    traffic_percentage: float = Field(100.0, ge=0, le=100, description="Traffic percentage for canary/AB testing")


class ModelVersionInfo(BaseModel):
    """Model version information"""
    model_id: str
    name: str
    version: str
    model_type: ModelType
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    created_by: str
    description: str
    tags: List[str]
    performance_metrics: Dict[str, float]
    model_size_mb: float
    deployment_info: Optional[Dict[str, Any]] = None


class DeploymentInfo(BaseModel):
    """Deployment information"""
    deployment_id: str
    model_id: str
    environment: DeploymentEnvironment
    status: str
    created_at: datetime
    updated_at: datetime
    replicas: int
    health_status: str
    endpoint_url: str
    resource_usage: Dict[str, Any]


class ModelManagerService:
    """Core model management service"""
    
    def __init__(self, 
                 db_pool: asyncpg.Pool,
                 redis_client: aioredis.Redis,
                 mongo_client: AsyncIOMotorClient,
                 s3_client: boto3.client,
                 mlflow_client: MlflowClient):
        
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.mongo_client = mongo_client
        self.s3_client = s3_client
        self.mlflow_client = mlflow_client
        
        # Configuration
        self.model_storage_bucket = os.getenv('MODEL_STORAGE_BUCKET', 'ml-models-bucket')
        self.artifact_storage_path = os.getenv('ARTIFACT_STORAGE_PATH', '/tmp/ml_artifacts')
        
        # Kubernetes client
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        self.k8s_apps_v1 = client.AppsV1Api()
        self.k8s_core_v1 = client.CoreV1Api()
        self.k8s_networking_v1 = client.NetworkingV1Api()
        
        # Git repository for model versioning
        self.git_repo_path = os.getenv('MODEL_GIT_REPO', '/tmp/model_repo')
        self._init_git_repo()
        
    async def register_model(self, request: ModelRegistrationRequest, model_file: UploadFile) -> str:
        """Register a new model"""
        try:
            # Generate model ID
            model_id = f"{request.name}_{request.version}_{uuid.uuid4().hex[:8]}"
            
            # Save model file to storage
            model_path = await self._save_model_file(model_id, model_file)
            
            # Extract model metadata
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            
            # Create model metadata
            metadata = ModelMetadata(
                model_id=model_id,
                name=request.name,
                version=request.version,
                model_type=request.model_type,
                status=ModelStatus.DRAFT,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                created_by="api_user",  # Would come from auth
                description=request.description,
                tags=request.tags,
                performance_metrics=request.performance_metrics,
                model_size_mb=model_size,
                training_data_info=request.training_data_info,
                dependencies=request.dependencies,
                deployment_config={}
            )
            
            # Store in database
            await self._store_model_metadata(metadata)
            
            # Register with MLflow
            await self._register_with_mlflow(metadata, model_path)
            
            # Version control with Git
            await self._version_control_model(metadata, model_path)
            
            # Upload to S3
            await self._upload_to_s3(model_id, model_path)
            
            logger.info(f"Model registered successfully: {model_id}")
            MODEL_MANAGEMENT_REQUESTS.labels(operation='register').inc()
            
            return model_id
            
        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            raise HTTPException(status_code=500, detail=f"Model registration failed: {str(e)}")
    
    async def get_model_versions(self, model_name: str) -> List[ModelVersionInfo]:
        """Get all versions of a model"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT * FROM model_metadata 
                    WHERE name = $1 
                    ORDER BY created_at DESC
                """
                rows = await conn.fetch(query, model_name)
                
                versions = []
                for row in rows:
                    # Get deployment info
                    deployment_info = await self._get_deployment_info(row['model_id'])
                    
                    version_info = ModelVersionInfo(
                        model_id=row['model_id'],
                        name=row['name'],
                        version=row['version'],
                        model_type=ModelType(row['model_type']),
                        status=ModelStatus(row['status']),
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        created_by=row['created_by'],
                        description=row['description'],
                        tags=json.loads(row['tags']),
                        performance_metrics=json.loads(row['performance_metrics']),
                        model_size_mb=row['model_size_mb'],
                        deployment_info=deployment_info
                    )
                    versions.append(version_info)
                
                return versions
                
        except Exception as e:
            logger.error(f"Failed to get model versions: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get model versions: {str(e)}")
    
    async def deploy_model(self, request: ModelDeploymentRequest) -> str:
        """Deploy model to specified environment"""
        try:
            start_time = time.time()
            
            # Generate deployment ID
            deployment_id = f"deploy_{request.model_id}_{request.environment.value}_{uuid.uuid4().hex[:8]}"
            
            # Get model metadata
            metadata = await self._get_model_metadata(request.model_id)
            if not metadata:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Update model status
            await self._update_model_status(request.model_id, ModelStatus.DEPLOYING)
            
            # Create Kubernetes deployment
            deployment_info = await self._create_k8s_deployment(request, metadata, deployment_id)
            
            # Store deployment information
            await self._store_deployment_info(deployment_id, request, deployment_info)
            
            # Update model status
            await self._update_model_status(request.model_id, ModelStatus.DEPLOYED)
            
            deployment_time = time.time() - start_time
            DEPLOYMENT_DURATION.labels(deployment_type='kubernetes').observe(deployment_time)
            MODEL_DEPLOYMENTS_TOTAL.labels(
                model_name=metadata['name'],
                version=metadata['version'],
                status='success'
            ).inc()
            
            logger.info(f"Model deployed successfully: {deployment_id}")
            
            return deployment_id
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            
            # Update model status to failed
            try:
                await self._update_model_status(request.model_id, ModelStatus.FAILED)
            except:
                pass
            
            MODEL_DEPLOYMENTS_TOTAL.labels(
                model_name=request.model_id,
                version='unknown',
                status='failed'
            ).inc()
            
            raise HTTPException(status_code=500, detail=f"Model deployment failed: {str(e)}")
    
    async def optimize_model(self, model_id: str, optimizations: List[str]) -> Dict[str, Any]:
        """Optimize model for inference"""
        try:
            # Get model metadata
            metadata = await self._get_model_metadata(model_id)
            if not metadata:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Download model from S3
            model_path = await self._download_from_s3(model_id)
            
            # Create optimization configuration
            model_config = ModelConfig(
                model_path=model_path,
                model_type=metadata['model_type'],
                optimizations=[OptimizationType(opt) for opt in optimizations],
                target_latency_ms=50.0,
                target_memory_mb=256.0
            )
            
            # Apply optimizations
            optimizer = ModelOptimizer(model_config)
            optimizer.load_model()
            optimized_model = optimizer.optimize_model()
            
            # Benchmark optimization
            test_data = pd.DataFrame(np.random.randn(100, 10))  # Dummy test data
            benchmark_results = optimizer.benchmark_model(test_data)
            
            # Save optimized model
            optimized_path = f"{model_path}_optimized"
            optimizer.save_optimized_model(optimized_path)
            
            # Upload optimized model
            optimized_model_id = f"{model_id}_optimized"
            await self._upload_to_s3(optimized_model_id, optimized_path)
            
            # Generate optimization report
            report = {
                'original_model_id': model_id,
                'optimized_model_id': optimized_model_id,
                'optimizations_applied': optimizations,
                'benchmark_results': benchmark_results,
                'optimization_timestamp': datetime.now().isoformat()
            }
            
            # Store optimization results
            await self._store_optimization_results(model_id, report)
            
            logger.info(f"Model optimization completed: {model_id}")
            
            return report
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            raise HTTPException(status_code=500, detail=f"Model optimization failed: {str(e)}")
    
    async def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback model deployment"""
        try:
            # Get deployment info
            deployment_info = await self._get_deployment_by_id(deployment_id)
            if not deployment_info:
                raise HTTPException(status_code=404, detail="Deployment not found")
            
            # Get previous deployment
            previous_deployment = await self._get_previous_deployment(
                deployment_info['model_id'],
                deployment_info['environment']
            )
            
            if not previous_deployment:
                raise HTTPException(status_code=400, detail="No previous deployment found")
            
            # Scale down current deployment
            await self._scale_k8s_deployment(deployment_id, 0)
            
            # Scale up previous deployment
            await self._scale_k8s_deployment(previous_deployment['deployment_id'], 
                                           previous_deployment['replicas'])
            
            # Update deployment status
            await self._update_deployment_status(deployment_id, 'rolled_back')
            await self._update_deployment_status(previous_deployment['deployment_id'], 'active')
            
            logger.info(f"Deployment rolled back successfully: {deployment_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment rollback failed: {e}")
            raise HTTPException(status_code=500, detail=f"Deployment rollback failed: {str(e)}")
    
    async def get_model_health(self, model_id: str) -> Dict[str, Any]:
        """Get model health status"""
        try:
            # Get model metadata
            metadata = await self._get_model_metadata(model_id)
            if not metadata:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Get deployment info
            deployment_info = await self._get_deployment_info(model_id)
            
            health_status = {
                'model_id': model_id,
                'name': metadata['name'],
                'version': metadata['version'],
                'status': metadata['status'],
                'deployments': []
            }
            
            if deployment_info:
                for deployment in deployment_info:
                    # Check Kubernetes deployment health
                    k8s_health = await self._check_k8s_deployment_health(deployment['deployment_id'])
                    
                    deployment_health = {
                        'deployment_id': deployment['deployment_id'],
                        'environment': deployment['environment'],
                        'status': deployment['status'],
                        'replicas': deployment['replicas'],
                        'kubernetes_health': k8s_health,
                        'endpoint_health': await self._check_endpoint_health(deployment.get('endpoint_url'))
                    }
                    
                    health_status['deployments'].append(deployment_health)
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
    
    # Private helper methods
    
    async def _save_model_file(self, model_id: str, model_file: UploadFile) -> str:
        """Save uploaded model file"""
        # Create artifact directory
        artifact_dir = Path(self.artifact_storage_path) / model_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = artifact_dir / model_file.filename
        with open(file_path, "wb") as f:
            content = await model_file.read()
            f.write(content)
        
        return str(file_path)
    
    async def _store_model_metadata(self, metadata: ModelMetadata):
        """Store model metadata in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO model_metadata (
                    model_id, name, version, model_type, status, created_at, updated_at,
                    created_by, description, tags, performance_metrics, model_size_mb,
                    training_data_info, dependencies, deployment_config
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            """, 
                metadata.model_id,
                metadata.name,
                metadata.version,
                metadata.model_type.value,
                metadata.status.value,
                metadata.created_at,
                metadata.updated_at,
                metadata.created_by,
                metadata.description,
                json.dumps(metadata.tags),
                json.dumps(metadata.performance_metrics),
                metadata.model_size_mb,
                json.dumps(metadata.training_data_info),
                json.dumps(metadata.dependencies),
                json.dumps(metadata.deployment_config)
            )
    
    async def _register_with_mlflow(self, metadata: ModelMetadata, model_path: str):
        """Register model with MLflow"""
        try:
            # Create or get experiment
            experiment_name = f"{metadata.model_type.value}_models"
            try:
                experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
                experiment_id = experiment.experiment_id
            except:
                experiment_id = self.mlflow_client.create_experiment(experiment_name)
            
            # Create run
            with mlflow.start_run(experiment_id=experiment_id) as run:
                # Log parameters
                mlflow.log_param("model_type", metadata.model_type.value)
                mlflow.log_param("version", metadata.version)
                
                # Log metrics
                for metric_name, metric_value in metadata.performance_metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model
                mlflow.log_artifact(model_path, "model")
                
                # Register model
                model_uri = f"runs:/{run.info.run_id}/model"
                self.mlflow_client.create_registered_model(metadata.name)
                
                model_version = self.mlflow_client.create_model_version(
                    name=metadata.name,
                    source=model_uri,
                    run_id=run.info.run_id
                )
                
                logger.info(f"Model registered with MLflow: {metadata.name} v{model_version.version}")
                
        except Exception as e:
            logger.warning(f"MLflow registration failed: {e}")
    
    def _init_git_repo(self):
        """Initialize Git repository for model versioning"""
        try:
            if not os.path.exists(self.git_repo_path):
                os.makedirs(self.git_repo_path)
                self.git_repo = Repo.init(self.git_repo_path)
            else:
                self.git_repo = Repo(self.git_repo_path)
        except Exception as e:
            logger.warning(f"Git repository initialization failed: {e}")
            self.git_repo = None
    
    async def _version_control_model(self, metadata: ModelMetadata, model_path: str):
        """Version control model with Git"""
        if not self.git_repo:
            return
        
        try:
            # Copy model to git repo
            git_model_path = os.path.join(self.git_repo_path, f"{metadata.model_id}.joblib")
            shutil.copy2(model_path, git_model_path)
            
            # Create metadata file
            metadata_path = os.path.join(self.git_repo_path, f"{metadata.model_id}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2, default=str)
            
            # Git operations
            self.git_repo.index.add([git_model_path, metadata_path])
            commit_message = f"Add model {metadata.name} v{metadata.version}"
            self.git_repo.index.commit(commit_message)
            
            # Create tag
            tag_name = f"{metadata.name}_v{metadata.version}"
            self.git_repo.create_tag(tag_name)
            
            logger.info(f"Model versioned with Git: {tag_name}")
            
        except Exception as e:
            logger.warning(f"Git versioning failed: {e}")
    
    async def _upload_to_s3(self, model_id: str, model_path: str):
        """Upload model to S3"""
        try:
            s3_key = f"models/{model_id}/{os.path.basename(model_path)}"
            
            # Upload file
            self.s3_client.upload_file(
                model_path,
                self.model_storage_bucket,
                s3_key
            )
            
            logger.info(f"Model uploaded to S3: {s3_key}")
            
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            raise
    
    async def _download_from_s3(self, model_id: str) -> str:
        """Download model from S3"""
        try:
            # List objects to find the model file
            response = self.s3_client.list_objects_v2(
                Bucket=self.model_storage_bucket,
                Prefix=f"models/{model_id}/"
            )
            
            if 'Contents' not in response:
                raise FileNotFoundError(f"Model {model_id} not found in S3")
            
            # Get the first model file
            s3_key = response['Contents'][0]['Key']
            
            # Download to local path
            local_path = os.path.join(self.artifact_storage_path, f"{model_id}_downloaded.joblib")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            self.s3_client.download_file(
                self.model_storage_bucket,
                s3_key,
                local_path
            )
            
            return local_path
            
        except ClientError as e:
            logger.error(f"S3 download failed: {e}")
            raise
    
    async def _create_k8s_deployment(self, request: ModelDeploymentRequest, 
                                   metadata: Dict[str, Any], deployment_id: str) -> Dict[str, Any]:
        """Create Kubernetes deployment"""
        
        # Deployment configuration
        deployment_name = f"model-{deployment_id}"
        namespace = "ml-serving"
        
        # Container image (would be your model serving image)
        image = "fraud-detection-service:latest"
        
        # Environment variables
        env_vars = [
            client.V1EnvVar(name="MODEL_ID", value=request.model_id),
            client.V1EnvVar(name="MODEL_NAME", value=metadata['name']),
            client.V1EnvVar(name="MODEL_VERSION", value=metadata['version']),
            client.V1EnvVar(name="DEPLOYMENT_ID", value=deployment_id),
        ]
        
        # Add custom environment variables
        for key, value in request.environment_variables.items():
            env_vars.append(client.V1EnvVar(name=key, value=value))
        
        # Resource requirements
        resources = client.V1ResourceRequirements(
            requests={
                "cpu": request.resources.get("cpu", "500m"),
                "memory": request.resources.get("memory", "1Gi")
            },
            limits={
                "cpu": request.resources.get("cpu_limit", "2000m"),
                "memory": request.resources.get("memory_limit", "4Gi")
            }
        )
        
        # Container definition
        container = client.V1Container(
            name=deployment_name,
            image=image,
            ports=[client.V1ContainerPort(container_port=8000)],
            env=env_vars,
            resources=resources,
            liveness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(
                    path=request.health_check_path,
                    port=8000
                ),
                initial_delay_seconds=30,
                period_seconds=10
            ),
            readiness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(
                    path=request.health_check_path,
                    port=8000
                ),
                initial_delay_seconds=5,
                period_seconds=5
            )
        )
        
        # Pod template
        pod_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={
                    "app": deployment_name,
                    "model_id": request.model_id,
                    "environment": request.environment.value
                }
            ),
            spec=client.V1PodSpec(containers=[container])
        )
        
        # Deployment spec
        deployment_spec = client.V1DeploymentSpec(
            replicas=request.replicas,
            selector=client.V1LabelSelector(
                match_labels={"app": deployment_name}
            ),
            template=pod_template
        )
        
        # Deployment object
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(
                name=deployment_name,
                namespace=namespace,
                labels={
                    "deployment_id": deployment_id,
                    "model_id": request.model_id,
                    "environment": request.environment.value
                }
            ),
            spec=deployment_spec
        )
        
        # Create deployment
        try:
            self.k8s_apps_v1.create_namespaced_deployment(
                namespace=namespace,
                body=deployment
            )
        except ApiException as e:
            if e.status == 409:  # Already exists
                self.k8s_apps_v1.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body=deployment
                )
            else:
                raise
        
        # Create service
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=f"{deployment_name}-service",
                namespace=namespace
            ),
            spec=client.V1ServiceSpec(
                selector={"app": deployment_name},
                ports=[client.V1ServicePort(
                    port=80,
                    target_port=8000,
                    protocol="TCP"
                )],
                type="ClusterIP"
            )
        )
        
        try:
            self.k8s_core_v1.create_namespaced_service(
                namespace=namespace,
                body=service
            )
        except ApiException as e:
            if e.status != 409:  # Ignore if already exists
                raise
        
        # Create HPA if auto-scaling is enabled
        if request.auto_scaling:
            await self._create_hpa(deployment_name, namespace, request.replicas)
        
        return {
            "deployment_name": deployment_name,
            "namespace": namespace,
            "service_name": f"{deployment_name}-service",
            "endpoint_url": f"http://{deployment_name}-service.{namespace}.svc.cluster.local"
        }
    
    async def _create_hpa(self, deployment_name: str, namespace: str, min_replicas: int):
        """Create Horizontal Pod Autoscaler"""
        hpa = client.V2HorizontalPodAutoscaler(
            api_version="autoscaling/v2",
            kind="HorizontalPodAutoscaler",
            metadata=client.V1ObjectMeta(
                name=f"{deployment_name}-hpa",
                namespace=namespace
            ),
            spec=client.V2HorizontalPodAutoscalerSpec(
                scale_target_ref=client.V2CrossVersionObjectReference(
                    api_version="apps/v1",
                    kind="Deployment",
                    name=deployment_name
                ),
                min_replicas=min_replicas,
                max_replicas=min_replicas * 3,
                metrics=[
                    client.V2MetricSpec(
                        type="Resource",
                        resource=client.V2ResourceMetricSource(
                            name="cpu",
                            target=client.V2MetricTarget(
                                type="Utilization",
                                average_utilization=70
                            )
                        )
                    )
                ]
            )
        )
        
        try:
            autoscaling_v2 = client.AutoscalingV2Api()
            autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                namespace=namespace,
                body=hpa
            )
        except ApiException as e:
            if e.status != 409:  # Ignore if already exists
                logger.warning(f"Failed to create HPA: {e}")
    
    async def _get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model metadata from database"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM model_metadata WHERE model_id = $1",
                model_id
            )
            return dict(row) if row else None
    
    async def _update_model_status(self, model_id: str, status: ModelStatus):
        """Update model status"""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE model_metadata SET status = $1, updated_at = $2 WHERE model_id = $3",
                status.value, datetime.now(), model_id
            )
    
    async def _store_deployment_info(self, deployment_id: str, request: ModelDeploymentRequest, 
                                   deployment_info: Dict[str, Any]):
        """Store deployment information"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO model_deployments (
                    deployment_id, model_id, environment, status, created_at, updated_at,
                    replicas, deployment_config, endpoint_url
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
                deployment_id,
                request.model_id,
                request.environment.value,
                "active",
                datetime.now(),
                datetime.now(),
                request.replicas,
                json.dumps(asdict(request)),
                deployment_info.get("endpoint_url")
            )
    
    async def _get_deployment_info(self, model_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get deployment information for model"""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM model_deployments WHERE model_id = $1 AND status = 'active'",
                model_id
            )
            return [dict(row) for row in rows] if rows else None


# FastAPI application
app = FastAPI(
    title="AI Orchestrator - Model Management Service",
    description="Model management service with versioning and deployment",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
model_service: ModelManagerService = None

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global model_service
    
    # Initialize database connections
    db_pool = await asyncpg.create_pool(
        os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/mlops')
    )
    
    redis_client = aioredis.from_url(
        os.getenv('REDIS_URL', 'redis://localhost:6379')
    )
    
    mongo_client = AsyncIOMotorClient(
        os.getenv('MONGODB_URL', 'mongodb://localhost:27017')
    )
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-west-2')
    )
    
    mlflow_client = MlflowClient(
        tracking_uri=os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    )
    
    model_service = ModelManagerService(
        db_pool=db_pool,
        redis_client=redis_client,
        mongo_client=mongo_client,
        s3_client=s3_client,
        mlflow_client=mlflow_client
    )

# API endpoints
@app.post("/models/register")
async def register_model(
    request: ModelRegistrationRequest,
    model_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Register a new model"""
    model_id = await model_service.register_model(request, model_file)
    return {"model_id": model_id, "status": "registered"}

@app.get("/models/{model_name}/versions")
async def get_model_versions(model_name: str):
    """Get all versions of a model"""
    versions = await model_service.get_model_versions(model_name)
    return {"versions": versions}

@app.post("/models/deploy")
async def deploy_model(request: ModelDeploymentRequest):
    """Deploy a model to specified environment"""
    deployment_id = await model_service.deploy_model(request)
    return {"deployment_id": deployment_id, "status": "deploying"}

@app.post("/models/{model_id}/optimize")
async def optimize_model(model_id: str, optimizations: List[str]):
    """Optimize model for inference"""
    report = await model_service.optimize_model(model_id, optimizations)
    return {"optimization_report": report}

@app.post("/deployments/{deployment_id}/rollback")
async def rollback_deployment(deployment_id: str):
    """Rollback a deployment"""
    success = await model_service.rollback_deployment(deployment_id)
    return {"success": success}

@app.get("/models/{model_id}/health")
async def get_model_health(model_id: str):
    """Get model health status"""
    health = await model_service.get_model_health(model_id)
    return health

if __name__ == "__main__":
    uvicorn.run(
        "model_management_service:app",
        host="0.0.0.0",
        port=8080,
        reload=False
    )
