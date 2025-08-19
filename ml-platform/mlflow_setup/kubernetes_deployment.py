"""
Kubernetes deployment automation for MLflow models
"""

import os
import yaml
import logging
from typing import Dict, List, Optional, Any
import kubernetes
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import mlflow
from mlflow.deployments import get_deploy_client

logger = logging.getLogger(__name__)


class KubernetesMLflowDeployment:
    """
    Kubernetes deployment manager for MLflow models
    """
    
    def __init__(self, kubeconfig_path: str = None, namespace: str = "ml-serving"):
        self.namespace = namespace
        self.k8s_client = None
        self.apps_v1 = None
        self.core_v1 = None
        self.networking_v1 = None
        self._setup_kubernetes_client(kubeconfig_path)
    
    def _setup_kubernetes_client(self, kubeconfig_path: str = None):
        """Setup Kubernetes client"""
        try:
            if kubeconfig_path:
                config.load_kube_config(config_file=kubeconfig_path)
            else:
                # Try in-cluster config first, then local config
                try:
                    config.load_incluster_config()
                except:
                    config.load_kube_config()
            
            self.k8s_client = client.ApiClient()
            self.apps_v1 = client.AppsV1Api()
            self.core_v1 = client.CoreV1Api()
            self.networking_v1 = client.NetworkingV1Api()
            
            logger.info("Kubernetes client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Kubernetes client: {e}")
            raise
    
    def create_namespace(self) -> bool:
        """Create namespace if it doesn't exist"""
        try:
            # Check if namespace exists
            try:
                self.core_v1.read_namespace(name=self.namespace)
                logger.info(f"Namespace {self.namespace} already exists")
                return True
            except ApiException as e:
                if e.status != 404:
                    raise
            
            # Create namespace
            namespace_manifest = client.V1Namespace(
                metadata=client.V1ObjectMeta(
                    name=self.namespace,
                    labels={
                        'app': 'mlflow-serving',
                        'managed-by': 'mlflow-deployment-manager'
                    }
                )
            )
            
            self.core_v1.create_namespace(body=namespace_manifest)
            logger.info(f"Created namespace: {self.namespace}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create namespace: {e}")
            return False
    
    def create_model_deployment(self, 
                               deployment_name: str,
                               model_uri: str,
                               replicas: int = 2,
                               resources: Dict[str, Dict[str, str]] = None,
                               environment_vars: Dict[str, str] = None,
                               image: str = "python:3.9-slim") -> bool:
        """Create Kubernetes deployment for MLflow model"""
        try:
            # Default resources
            if resources is None:
                resources = {
                    'requests': {'cpu': '500m', 'memory': '1Gi'},
                    'limits': {'cpu': '2000m', 'memory': '4Gi'}
                }
            
            # Default environment variables
            env_vars = environment_vars or {}
            env_vars.update({
                'MODEL_URI': model_uri,
                'MLFLOW_TRACKING_URI': os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000'),
                'PYTHONPATH': '/app'
            })
            
            # Create environment variable list
            env_list = [
                client.V1EnvVar(name=k, value=v) for k, v in env_vars.items()
            ]
            
            # Create container
            container = client.V1Container(
                name=f"{deployment_name}-container",
                image=image,
                ports=[client.V1ContainerPort(container_port=8080)],
                env=env_list,
                resources=client.V1ResourceRequirements(
                    requests=resources['requests'],
                    limits=resources['limits']
                ),
                command=[
                    "/bin/bash", "-c",
                    f"""
                    pip install mlflow[extras] pandas numpy scikit-learn &&
                    mlflow models serve -m {model_uri} -h 0.0.0.0 -p 8080 --no-conda
                    """
                ],
                liveness_probe=client.V1Probe(
                    http_get=client.V1HTTPGetAction(
                        path="/health",
                        port=8080
                    ),
                    initial_delay_seconds=30,
                    period_seconds=10
                ),
                readiness_probe=client.V1Probe(
                    http_get=client.V1HTTPGetAction(
                        path="/health",
                        port=8080
                    ),
                    initial_delay_seconds=5,
                    period_seconds=5
                )
            )
            
            # Create pod template
            pod_template = client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={
                        'app': deployment_name,
                        'version': 'v1',
                        'component': 'model-server'
                    }
                ),
                spec=client.V1PodSpec(
                    containers=[container],
                    restart_policy='Always'
                )
            )
            
            # Create deployment spec
            deployment_spec = client.V1DeploymentSpec(
                replicas=replicas,
                selector=client.V1LabelSelector(
                    match_labels={'app': deployment_name}
                ),
                template=pod_template,
                strategy=client.V1DeploymentStrategy(
                    type='RollingUpdate',
                    rolling_update=client.V1RollingUpdateDeployment(
                        max_surge='25%',
                        max_unavailable='25%'
                    )
                )
            )
            
            # Create deployment
            deployment = client.V1Deployment(
                api_version='apps/v1',
                kind='Deployment',
                metadata=client.V1ObjectMeta(
                    name=deployment_name,
                    namespace=self.namespace,
                    labels={
                        'app': deployment_name,
                        'managed-by': 'mlflow-deployment-manager'
                    }
                ),
                spec=deployment_spec
            )
            
            # Apply deployment
            self.apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment
            )
            
            logger.info(f"Created deployment: {deployment_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create deployment: {e}")
            return False
    
    def create_service(self, 
                      deployment_name: str,
                      service_type: str = "ClusterIP",
                      port: int = 8080) -> bool:
        """Create Kubernetes service for model deployment"""
        try:
            service_spec = client.V1ServiceSpec(
                selector={'app': deployment_name},
                ports=[
                    client.V1ServicePort(
                        name='http',
                        port=port,
                        target_port=8080,
                        protocol='TCP'
                    )
                ],
                type=service_type
            )
            
            service = client.V1Service(
                api_version='v1',
                kind='Service',
                metadata=client.V1ObjectMeta(
                    name=f"{deployment_name}-service",
                    namespace=self.namespace,
                    labels={
                        'app': deployment_name,
                        'component': 'service'
                    }
                ),
                spec=service_spec
            )
            
            self.core_v1.create_namespaced_service(
                namespace=self.namespace,
                body=service
            )
            
            logger.info(f"Created service: {deployment_name}-service")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create service: {e}")
            return False
    
    def create_ingress(self, 
                      deployment_name: str,
                      host: str,
                      path: str = "/",
                      tls_enabled: bool = False,
                      tls_secret_name: str = None) -> bool:
        """Create Kubernetes ingress for model deployment"""
        try:
            # Create path
            backend = client.V1IngressBackend(
                service=client.V1IngressServiceBackend(
                    name=f"{deployment_name}-service",
                    port=client.V1ServiceBackendPort(number=8080)
                )
            )
            
            path_obj = client.V1HTTPIngressPath(
                path=path,
                path_type='Prefix',
                backend=backend
            )
            
            rule = client.V1IngressRule(
                host=host,
                http=client.V1HTTPIngressRuleValue(paths=[path_obj])
            )
            
            spec = client.V1IngressSpec(rules=[rule])
            
            # Add TLS if enabled
            if tls_enabled and tls_secret_name:
                tls = client.V1IngressTLS(
                    hosts=[host],
                    secret_name=tls_secret_name
                )
                spec.tls = [tls]
            
            ingress = client.V1Ingress(
                api_version='networking.k8s.io/v1',
                kind='Ingress',
                metadata=client.V1ObjectMeta(
                    name=f"{deployment_name}-ingress",
                    namespace=self.namespace,
                    annotations={
                        'nginx.ingress.kubernetes.io/rewrite-target': '/',
                        'nginx.ingress.kubernetes.io/ssl-redirect': 'true' if tls_enabled else 'false'
                    }
                ),
                spec=spec
            )
            
            self.networking_v1.create_namespaced_ingress(
                namespace=self.namespace,
                body=ingress
            )
            
            logger.info(f"Created ingress: {deployment_name}-ingress")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create ingress: {e}")
            return False
    
    def create_hpa(self, 
                   deployment_name: str,
                   min_replicas: int = 2,
                   max_replicas: int = 10,
                   target_cpu_utilization: int = 70) -> bool:
        """Create Horizontal Pod Autoscaler"""
        try:
            hpa_spec = client.V2HorizontalPodAutoscalerSpec(
                scale_target_ref=client.V2CrossVersionObjectReference(
                    api_version='apps/v1',
                    kind='Deployment',
                    name=deployment_name
                ),
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                metrics=[
                    client.V2MetricSpec(
                        type='Resource',
                        resource=client.V2ResourceMetricSource(
                            name='cpu',
                            target=client.V2MetricTarget(
                                type='Utilization',
                                average_utilization=target_cpu_utilization
                            )
                        )
                    )
                ]
            )
            
            hpa = client.V2HorizontalPodAutoscaler(
                api_version='autoscaling/v2',
                kind='HorizontalPodAutoscaler',
                metadata=client.V1ObjectMeta(
                    name=f"{deployment_name}-hpa",
                    namespace=self.namespace
                ),
                spec=hpa_spec
            )
            
            autoscaling_v2 = client.AutoscalingV2Api()
            autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                namespace=self.namespace,
                body=hpa
            )
            
            logger.info(f"Created HPA: {deployment_name}-hpa")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create HPA: {e}")
            return False
    
    def deploy_complete_model_stack(self, 
                                   deployment_name: str,
                                   model_uri: str,
                                   config: Dict[str, Any] = None) -> Dict[str, bool]:
        """Deploy complete model serving stack"""
        
        default_config = {
            'replicas': 2,
            'resources': {
                'requests': {'cpu': '500m', 'memory': '1Gi'},
                'limits': {'cpu': '2000m', 'memory': '4Gi'}
            },
            'service_type': 'ClusterIP',
            'create_ingress': False,
            'host': f"{deployment_name}.example.com",
            'tls_enabled': False,
            'hpa_enabled': True,
            'min_replicas': 2,
            'max_replicas': 10,
            'target_cpu_utilization': 70
        }
        
        if config:
            default_config.update(config)
        
        results = {}
        
        # Create namespace
        results['namespace'] = self.create_namespace()
        
        # Create deployment
        results['deployment'] = self.create_model_deployment(
            deployment_name=deployment_name,
            model_uri=model_uri,
            replicas=default_config['replicas'],
            resources=default_config['resources']
        )
        
        # Create service
        results['service'] = self.create_service(
            deployment_name=deployment_name,
            service_type=default_config['service_type']
        )
        
        # Create ingress if requested
        if default_config['create_ingress']:
            results['ingress'] = self.create_ingress(
                deployment_name=deployment_name,
                host=default_config['host'],
                tls_enabled=default_config['tls_enabled']
            )
        
        # Create HPA if requested
        if default_config['hpa_enabled']:
            results['hpa'] = self.create_hpa(
                deployment_name=deployment_name,
                min_replicas=default_config['min_replicas'],
                max_replicas=default_config['max_replicas'],
                target_cpu_utilization=default_config['target_cpu_utilization']
            )
        
        logger.info(f"Complete model stack deployment results: {results}")
        return results
    
    def get_deployment_status(self, deployment_name: str) -> Dict[str, Any]:
        """Get deployment status"""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            status = {
                'name': deployment_name,
                'namespace': self.namespace,
                'replicas': deployment.status.replicas,
                'ready_replicas': deployment.status.ready_replicas,
                'available_replicas': deployment.status.available_replicas,
                'updated_replicas': deployment.status.updated_replicas,
                'conditions': []
            }
            
            if deployment.status.conditions:
                for condition in deployment.status.conditions:
                    status['conditions'].append({
                        'type': condition.type,
                        'status': condition.status,
                        'reason': condition.reason,
                        'message': condition.message
                    })
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {}
    
    def scale_deployment(self, deployment_name: str, replicas: int) -> bool:
        """Scale deployment to specified number of replicas"""
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            # Update replica count
            deployment.spec.replicas = replicas
            
            # Apply update
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=deployment
            )
            
            logger.info(f"Scaled deployment {deployment_name} to {replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale deployment: {e}")
            return False
    
    def delete_deployment_stack(self, deployment_name: str) -> Dict[str, bool]:
        """Delete complete deployment stack"""
        results = {}
        
        try:
            # Delete HPA
            try:
                autoscaling_v2 = client.AutoscalingV2Api()
                autoscaling_v2.delete_namespaced_horizontal_pod_autoscaler(
                    name=f"{deployment_name}-hpa",
                    namespace=self.namespace
                )
                results['hpa'] = True
            except:
                results['hpa'] = False
            
            # Delete ingress
            try:
                self.networking_v1.delete_namespaced_ingress(
                    name=f"{deployment_name}-ingress",
                    namespace=self.namespace
                )
                results['ingress'] = True
            except:
                results['ingress'] = False
            
            # Delete service
            try:
                self.core_v1.delete_namespaced_service(
                    name=f"{deployment_name}-service",
                    namespace=self.namespace
                )
                results['service'] = True
            except:
                results['service'] = False
            
            # Delete deployment
            try:
                self.apps_v1.delete_namespaced_deployment(
                    name=deployment_name,
                    namespace=self.namespace
                )
                results['deployment'] = True
            except:
                results['deployment'] = False
            
            logger.info(f"Deployment stack deletion results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to delete deployment stack: {e}")
            return results


def create_monitoring_manifests(deployment_name: str, namespace: str = "ml-serving") -> List[str]:
    """Create Kubernetes manifests for monitoring"""
    
    # ServiceMonitor for Prometheus
    service_monitor = f"""
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: {deployment_name}-monitor
  namespace: {namespace}
  labels:
    app: {deployment_name}
spec:
  selector:
    matchLabels:
      app: {deployment_name}
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
"""
    
    # Grafana Dashboard ConfigMap
    grafana_dashboard = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: {deployment_name}-dashboard
  namespace: {namespace}
  labels:
    grafana_dashboard: "1"
data:
  dashboard.json: |
    {{
      "dashboard": {{
        "title": "MLflow Model - {deployment_name}",
        "panels": [
          {{
            "title": "Request Rate",
            "type": "graph",
            "targets": [
              {{
                "expr": "rate(http_requests_total{{job=\\"{deployment_name}\\"}}[5m])"
              }}
            ]
          }},
          {{
            "title": "Response Time",
            "type": "graph", 
            "targets": [
              {{
                "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{job=\\"{deployment_name}\\"}}[5m]))"
              }}
            ]
          }},
          {{
            "title": "Error Rate",
            "type": "graph",
            "targets": [
              {{
                "expr": "rate(http_requests_total{{job=\\"{deployment_name}\\", status=~\\"5..\\|4..\\"}}[5m])"
              }}
            ]
          }}
        ]
      }}
    }}
"""
    
    # Save manifests
    manifests = []
    
    service_monitor_file = f"{deployment_name}-service-monitor.yaml"
    with open(service_monitor_file, 'w') as f:
        f.write(service_monitor)
    manifests.append(service_monitor_file)
    
    dashboard_file = f"{deployment_name}-grafana-dashboard.yaml"
    with open(dashboard_file, 'w') as f:
        f.write(grafana_dashboard)
    manifests.append(dashboard_file)
    
    return manifests


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize deployment manager
    k8s_deployment = KubernetesMLflowDeployment(namespace="ml-serving")
    
    # Example deployment
    model_uri = "models:/fraud_detection_model/1"
    deployment_name = "fraud-model-v1"
    
    # Deploy complete stack
    results = k8s_deployment.deploy_complete_model_stack(
        deployment_name=deployment_name,
        model_uri=model_uri,
        config={
            'replicas': 3,
            'create_ingress': True,
            'host': 'fraud-model.ml.example.com',
            'hpa_enabled': True
        }
    )
    
    print(f"Deployment results: {results}")
    
    # Create monitoring manifests
    monitoring_manifests = create_monitoring_manifests(deployment_name)
    print(f"Created monitoring manifests: {monitoring_manifests}")
    
    print("Kubernetes deployment setup completed!")
