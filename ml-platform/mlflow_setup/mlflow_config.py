"""
MLflow configuration and utilities for fraud detection ML platform
"""

import os
import logging
import yaml
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.deployments import get_deploy_client
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

logger = logging.getLogger(__name__)


class MLflowManager:
    """
    Comprehensive MLflow manager for experiment tracking, model versioning, and deployment
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.client = None
        self.experiment_id = None
        self._setup_mlflow()
    
    def _get_default_config(self) -> Dict:
        """Get default MLflow configuration"""
        return {
            'tracking_uri': 'http://localhost:5000',
            'artifact_location': 's3://mlflow-artifacts-bucket/fraud-detection',
            'experiment_name': 'fraud_detection',
            'registry_uri': 'postgresql://mlflow:mlflow@localhost:5432/mlflow',
            
            # Model registry settings
            'model_stage_aliases': {
                'development': 'dev',
                'staging': 'staging', 
                'production': 'prod',
                'archived': 'archived'
            },
            
            # Deployment settings
            'deployment_targets': {
                'kubernetes': {
                    'target_uri': 'kubernetes://https://kubernetes.default.svc:443',
                    'namespace': 'ml-serving',
                    'resources': {
                        'requests': {'cpu': '500m', 'memory': '1Gi'},
                        'limits': {'cpu': '2000m', 'memory': '4Gi'}
                    }
                },
                'sagemaker': {
                    'target_uri': 'sagemaker',
                    'region': 'us-west-2',
                    'instance_type': 'ml.m5.large',
                    'instance_count': 1
                },
                'azure_ml': {
                    'target_uri': 'azureml',
                    'workspace_name': 'fraud-detection-ws',
                    'resource_group': 'ml-resources'
                }
            },
            
            # Logging settings
            'log_system_metrics': True,
            'log_model_signature': True,
            'log_input_example': True,
            'auto_log': {
                'sklearn': True,
                'tensorflow': True,
                'pytorch': False
            }
        }
    
    def _setup_mlflow(self):
        """Setup MLflow tracking and model registry"""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.config['tracking_uri'])
            
            # Set registry URI if different
            if 'registry_uri' in self.config:
                mlflow.set_registry_uri(self.config['registry_uri'])
            
            # Initialize client
            self.client = MlflowClient()
            
            # Setup experiment
            self._setup_experiment()
            
            # Configure autologging
            self._configure_autologging()
            
            logger.info(f"MLflow setup completed. Tracking URI: {self.config['tracking_uri']}")
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            raise
    
    def _setup_experiment(self):
        """Setup or get existing experiment"""
        experiment_name = self.config['experiment_name']
        
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment:
                self.experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name} (ID: {self.experiment_id})")
            else:
                raise Exception("Experiment not found")
        except:
            # Create new experiment
            self.experiment_id = self.client.create_experiment(
                name=experiment_name,
                artifact_location=self.config.get('artifact_location')
            )
            logger.info(f"Created new experiment: {experiment_name} (ID: {self.experiment_id})")
    
    def _configure_autologging(self):
        """Configure automatic logging for ML frameworks"""
        auto_log_config = self.config.get('auto_log', {})
        
        if auto_log_config.get('sklearn', False):
            mlflow.sklearn.autolog()
            logger.info("Enabled sklearn autologging")
        
        if auto_log_config.get('tensorflow', False):
            mlflow.tensorflow.autolog()
            logger.info("Enabled TensorFlow autologging")
        
        if auto_log_config.get('pytorch', False):
            try:
                mlflow.pytorch.autolog()
                logger.info("Enabled PyTorch autologging")
            except:
                logger.warning("PyTorch autologging not available")
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None, nested: bool = False) -> mlflow.ActiveRun:
        """Start a new MLflow run"""
        tags = tags or {}
        tags['mlflow.runName'] = run_name or f"fraud_detection_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return mlflow.start_run(
            experiment_id=self.experiment_id,
            tags=tags,
            nested=nested
        )
    
    def log_model_training(self, 
                          model: Any,
                          model_type: str,
                          training_data: pd.DataFrame,
                          validation_data: pd.DataFrame = None,
                          metrics: Dict[str, float] = None,
                          params: Dict[str, Any] = None,
                          artifacts: Dict[str, str] = None,
                          model_signature: mlflow.models.ModelSignature = None,
                          input_example: pd.DataFrame = None) -> str:
        """
        Comprehensive model training logging
        """
        try:
            with mlflow.start_run(experiment_id=self.experiment_id) as run:
                run_id = run.info.run_id
                
                # Log parameters
                if params:
                    mlflow.log_params(params)
                
                # Log model type and training info
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("training_samples", len(training_data))
                mlflow.log_param("feature_count", len(training_data.columns))
                
                if validation_data is not None:
                    mlflow.log_param("validation_samples", len(validation_data))
                
                # Log metrics
                if metrics:
                    mlflow.log_metrics(metrics)
                
                # Log model with signature and input example
                if model_type == 'tensorflow':
                    mlflow.tensorflow.log_model(
                        model=model,
                        artifact_path="model",
                        signature=model_signature,
                        input_example=input_example.values if input_example is not None else None
                    )
                else:
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        signature=model_signature,
                        input_example=input_example
                    )
                
                # Log training data statistics
                self._log_data_statistics(training_data, "training")
                if validation_data is not None:
                    self._log_data_statistics(validation_data, "validation")
                
                # Log additional artifacts
                if artifacts:
                    for artifact_name, artifact_path in artifacts.items():
                        mlflow.log_artifact(artifact_path, artifact_name)
                
                # Log environment info
                mlflow.log_param("python_version", os.sys.version)
                mlflow.log_param("mlflow_version", mlflow.__version__)
                
                logger.info(f"Model training logged successfully. Run ID: {run_id}")
                return run_id
                
        except Exception as e:
            logger.error(f"Failed to log model training: {e}")
            raise
    
    def _log_data_statistics(self, data: pd.DataFrame, data_type: str):
        """Log dataset statistics"""
        stats = {
            f"{data_type}_mean": data.select_dtypes(include=[np.number]).mean().to_dict(),
            f"{data_type}_std": data.select_dtypes(include=[np.number]).std().to_dict(),
            f"{data_type}_null_count": data.isnull().sum().to_dict()
        }
        
        # Log basic statistics
        mlflow.log_metric(f"{data_type}_samples", len(data))
        mlflow.log_metric(f"{data_type}_features", len(data.columns))
        mlflow.log_metric(f"{data_type}_null_percentage", data.isnull().sum().sum() / (len(data) * len(data.columns)))
        
        # Save detailed statistics as artifacts
        stats_file = f"{data_type}_statistics.json"
        import json
        with open(stats_file, 'w') as f:
            json.dump(stats, f, default=str, indent=2)
        mlflow.log_artifact(stats_file, "statistics")
        os.remove(stats_file)
    
    def register_model(self, 
                      run_id: str, 
                      model_name: str,
                      model_version_description: str = None,
                      tags: Dict[str, str] = None) -> mlflow.entities.ModelVersion:
        """Register model in MLflow Model Registry"""
        try:
            model_uri = f"runs:/{run_id}/model"
            
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags
            )
            
            # Update model version description
            if model_version_description:
                self.client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=model_version_description
                )
            
            logger.info(f"Model registered: {model_name} version {model_version.version}")
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def transition_model_stage(self, 
                              model_name: str,
                              version: str,
                              stage: str,
                              archive_existing_versions: bool = False) -> mlflow.entities.ModelVersion:
        """Transition model to a specific stage"""
        try:
            model_version = self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing_versions
            )
            
            logger.info(f"Model {model_name} version {version} transitioned to {stage}")
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            raise
    
    def compare_models(self, run_ids: List[str], metrics: List[str]) -> pd.DataFrame:
        """Compare multiple model runs"""
        try:
            comparison_data = []
            
            for run_id in run_ids:
                run = self.client.get_run(run_id)
                run_data = {
                    'run_id': run_id,
                    'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
                    'model_type': run.data.params.get('model_type', 'Unknown'),
                    'start_time': run.info.start_time,
                    'status': run.info.status
                }
                
                # Add requested metrics
                for metric in metrics:
                    run_data[metric] = run.data.metrics.get(metric, None)
                
                comparison_data.append(run_data)
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df['start_time'] = pd.to_datetime(comparison_df['start_time'], unit='ms')
            
            logger.info(f"Compared {len(run_ids)} model runs")
            return comparison_df
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            raise
    
    def get_best_model(self, 
                       model_name: str,
                       metric_name: str,
                       stage: str = None,
                       max_results: int = 10) -> Tuple[mlflow.entities.ModelVersion, Dict]:
        """Get the best model version based on a metric"""
        try:
            # Get model versions
            if stage:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
            else:
                all_versions = self.client.search_model_versions(f"name='{model_name}'")
                versions = sorted(all_versions, key=lambda x: int(x.version), reverse=True)[:max_results]
            
            best_model = None
            best_metric = float('-inf')
            best_run_info = None
            
            for version in versions:
                run = self.client.get_run(version.run_id)
                metric_value = run.data.metrics.get(metric_name)
                
                if metric_value is not None and metric_value > best_metric:
                    best_metric = metric_value
                    best_model = version
                    best_run_info = {
                        'run_id': version.run_id,
                        'metrics': run.data.metrics,
                        'params': run.data.params,
                        'tags': run.data.tags
                    }
            
            logger.info(f"Best model found: {model_name} v{best_model.version} with {metric_name}={best_metric}")
            return best_model, best_run_info
            
        except Exception as e:
            logger.error(f"Failed to get best model: {e}")
            raise
    
    def deploy_model(self, 
                     model_name: str,
                     model_version: str,
                     deployment_name: str,
                     target: str = 'kubernetes',
                     config: Dict = None) -> Dict:
        """Deploy model to specified target"""
        try:
            target_config = self.config['deployment_targets'].get(target, {})
            target_uri = target_config.get('target_uri')
            
            if not target_uri:
                raise ValueError(f"No configuration found for deployment target: {target}")
            
            # Get deployment client
            deployment_client = get_deploy_client(target_uri)
            
            # Prepare model URI
            model_uri = f"models:/{model_name}/{model_version}"
            
            # Prepare deployment config
            deployment_config = config or {}
            deployment_config.update(target_config.get('config', {}))
            
            # Deploy model
            deployment_info = deployment_client.create_deployment(
                name=deployment_name,
                model_uri=model_uri,
                config=deployment_config
            )
            
            logger.info(f"Model deployed successfully. Deployment: {deployment_name}")
            return deployment_info
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            raise
    
    def monitor_deployment(self, deployment_name: str, target: str = 'kubernetes') -> Dict:
        """Monitor model deployment status"""
        try:
            target_config = self.config['deployment_targets'].get(target, {})
            target_uri = target_config.get('target_uri')
            
            deployment_client = get_deploy_client(target_uri)
            deployment_info = deployment_client.get_deployment(deployment_name)
            
            return deployment_info
            
        except Exception as e:
            logger.error(f"Failed to monitor deployment: {e}")
            raise
    
    def create_model_serving_pipeline(self, 
                                     model_name: str,
                                     model_version: str,
                                     pipeline_name: str) -> str:
        """Create a complete model serving pipeline"""
        try:
            # This would create a comprehensive serving pipeline including:
            # 1. Model loading
            # 2. Feature preprocessing 
            # 3. Prediction serving
            # 4. Response formatting
            # 5. Monitoring and logging
            
            pipeline_code = f"""
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

class FraudModelServingPipeline(mlflow.pyfunc.PythonModel):
    
    def load_context(self, context):
        '''Load model and preprocessing artifacts'''
        self.model = mlflow.pyfunc.load_model(context.artifacts["model"])
        self.preprocessor = mlflow.pyfunc.load_model(context.artifacts["preprocessor"])
        self.logger = logging.getLogger(__name__)
    
    def predict(self, context, model_input):
        '''Make fraud predictions'''
        try:
            # Preprocess input data
            processed_input = self.preprocessor.predict(model_input)
            
            # Make predictions
            predictions = self.model.predict(processed_input)
            probabilities = self.model.predict_proba(processed_input)[:, 1]
            
            # Format results
            results = {{
                'predictions': predictions.tolist(),
                'fraud_probability': probabilities.tolist(),
                'model_name': '{model_name}',
                'model_version': '{model_version}',
                'timestamp': pd.Timestamp.now().isoformat()
            }}
            
            self.logger.info(f"Processed {{len(model_input)}} predictions")
            return results
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {{e}}")
            raise

# Example usage:
# pipeline = FraudModelServingPipeline()
# pipeline.load_context(context)
# results = pipeline.predict(context, input_data)
"""
            
            # Save pipeline code
            pipeline_file = f"{pipeline_name}_serving_pipeline.py"
            with open(pipeline_file, 'w') as f:
                f.write(pipeline_code)
            
            logger.info(f"Model serving pipeline created: {pipeline_file}")
            return pipeline_file
            
        except Exception as e:
            logger.error(f"Failed to create serving pipeline: {e}")
            raise
    
    def setup_model_monitoring(self, deployment_name: str, config: Dict = None) -> Dict:
        """Setup model monitoring and alerting"""
        monitoring_config = {
            'metrics': [
                'prediction_latency',
                'throughput',
                'error_rate',
                'model_drift',
                'data_drift'
            ],
            'alerts': {
                'latency_threshold': 1000,  # ms
                'error_rate_threshold': 0.05,  # 5%
                'drift_threshold': 0.1
            },
            'logging': {
                'log_predictions': True,
                'log_features': True,
                'sample_rate': 0.1
            }
        }
        
        if config:
            monitoring_config.update(config)
        
        # This would typically integrate with monitoring systems like:
        # - Prometheus/Grafana for metrics
        # - ELK stack for logging
        # - Custom drift detection systems
        
        logger.info(f"Model monitoring setup completed for deployment: {deployment_name}")
        return monitoring_config
    
    def cleanup_old_models(self, 
                          model_name: str,
                          keep_latest: int = 5,
                          archive_older_than_days: int = 90):
        """Cleanup old model versions"""
        try:
            # Get all model versions
            all_versions = self.client.search_model_versions(f"name='{model_name}'")
            
            # Sort by creation time
            sorted_versions = sorted(all_versions, 
                                   key=lambda x: x.creation_timestamp, 
                                   reverse=True)
            
            archived_count = 0
            deleted_count = 0
            
            for i, version in enumerate(sorted_versions):
                # Keep latest versions
                if i < keep_latest:
                    continue
                
                # Archive old versions
                days_old = (datetime.now().timestamp() * 1000 - version.creation_timestamp) / (1000 * 60 * 60 * 24)
                
                if days_old > archive_older_than_days and version.current_stage != 'Archived':
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=version.version,
                        stage='Archived'
                    )
                    archived_count += 1
            
            logger.info(f"Model cleanup completed. Archived: {archived_count}, Deleted: {deleted_count}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old models: {e}")
            raise


def create_mlflow_docker_compose() -> str:
    """Create Docker Compose configuration for MLflow"""
    
    docker_compose = """
version: '3.8'

services:
  mlflow-db:
    image: postgres:13
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
    volumes:
      - mlflow_db_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - mlflow-network

  mlflow-minio:
    image: minio/minio:latest
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - mlflow_minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"
    command: server /data --console-address ":9001"
    networks:
      - mlflow-network

  mlflow-server:
    image: python:3.9-slim
    depends_on:
      - mlflow-db
      - mlflow-minio
    environment:
      - MLFLOW_S3_ENDPOINT_URL=http://mlflow-minio:9000
      - AWS_ACCESS_KEY_ID=minioadmin
      - AWS_SECRET_ACCESS_KEY=minioadmin
    ports:
      - "5000:5000"
    volumes:
      - ./mlflow_setup:/app
    working_dir: /app
    command: >
      bash -c "
        pip install mlflow psycopg2-binary boto3 &&
        mlflow server
        --backend-store-uri postgresql://mlflow:mlflow@mlflow-db:5432/mlflow
        --default-artifact-root s3://mlflow-artifacts/
        --host 0.0.0.0
        --port 5000
      "
    networks:
      - mlflow-network

  mlflow-ui:
    image: python:3.9-slim
    depends_on:
      - mlflow-server
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow-server:5000
    ports:
      - "8080:8080"
    volumes:
      - ./mlflow_setup:/app
    working_dir: /app
    command: >
      bash -c "
        pip install mlflow streamlit plotly &&
        streamlit run mlflow_dashboard.py --server.port 8080 --server.address 0.0.0.0
      "
    networks:
      - mlflow-network

volumes:
  mlflow_db_data:
  mlflow_minio_data:

networks:
  mlflow-network:
    driver: bridge
"""
    
    with open('docker-compose.mlflow.yml', 'w') as f:
        f.write(docker_compose)
    
    return 'docker-compose.mlflow.yml'


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize MLflow manager
    mlflow_manager = MLflowManager()
    
    # Create Docker Compose for MLflow
    docker_compose_file = create_mlflow_docker_compose()
    print(f"MLflow Docker Compose created: {docker_compose_file}")
    
    # Example: Start a run and log some metrics
    with mlflow_manager.start_run("example_run") as run:
        mlflow.log_param("example_param", "test")
        mlflow.log_metric("example_metric", 0.95)
        print(f"Example run logged: {run.info.run_id}")
    
    print("MLflow setup completed!")
