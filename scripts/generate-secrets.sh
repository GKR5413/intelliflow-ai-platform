#!/bin/bash

# IntelliFlow AI Platform - Secret Generation Script
# This script generates secure secrets and creates Kubernetes secret manifests

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="${NAMESPACE:-intelliflow-prod}"
OUTPUT_DIR="${OUTPUT_DIR:-./k8s/secrets}"
VAULT_ENABLED="${VAULT_ENABLED:-false}"

echo -e "${BLUE}IntelliFlow AI Platform - Secret Generation${NC}"
echo "==========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to generate random password
generate_password() {
    local length=${1:-32}
    openssl rand -base64 "$length" | tr -d "=+/" | cut -c1-$length
}

# Function to base64 encode
b64encode() {
    echo -n "$1" | base64 | tr -d '\n'
}

# Function to create secret file
create_secret() {
    local secret_name="$1"
    local namespace="$2"
    local output_file="$OUTPUT_DIR/${secret_name}.yaml"
    
    cat > "$output_file" <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: $secret_name
  namespace: $namespace
  labels:
    app.kubernetes.io/name: intelliflow
    app.kubernetes.io/component: secrets
type: Opaque
data:
EOF
    echo "$output_file"
}

echo -e "${YELLOW}Generating secure passwords...${NC}"

# Generate passwords
DB_PASSWORD=$(generate_password 24)
REDIS_PASSWORD=$(generate_password 16)
MONGO_PASSWORD=$(generate_password 24)
JWT_SECRET=$(generate_password 32)
SMTP_PASSWORD=$(generate_password 16)
MLFLOW_PASSWORD=$(generate_password 16)
GRAFANA_API_KEY=$(generate_password 40)

echo -e "${GREEN}✓ Passwords generated${NC}"

# 1. Database secrets
echo -e "${YELLOW}Creating database secrets...${NC}"
DB_SECRET_FILE=$(create_secret "database-secrets" "$NAMESPACE")
cat >> "$DB_SECRET_FILE" <<EOF
  DB_USERNAME: $(b64encode "intelliflow_user")
  DB_PASSWORD: $(b64encode "$DB_PASSWORD")
  DB_HOST: $(b64encode "postgres")
  DB_PORT: $(b64encode "5432")
  DB_SSL_MODE: $(b64encode "require")
EOF

# 2. Redis secrets
echo -e "${YELLOW}Creating Redis secrets...${NC}"
REDIS_SECRET_FILE=$(create_secret "redis-secrets" "$NAMESPACE")
cat >> "$REDIS_SECRET_FILE" <<EOF
  REDIS_PASSWORD: $(b64encode "$REDIS_PASSWORD")
  REDIS_HOST: $(b64encode "redis")
  REDIS_PORT: $(b64encode "6379")
  REDIS_DB: $(b64encode "0")
EOF

# 3. MongoDB secrets
echo -e "${YELLOW}Creating MongoDB secrets...${NC}"
MONGO_SECRET_FILE=$(create_secret "mongodb-secrets" "$NAMESPACE")
cat >> "$MONGO_SECRET_FILE" <<EOF
  MONGO_USERNAME: $(b64encode "intelliflow_analytics")
  MONGO_PASSWORD: $(b64encode "$MONGO_PASSWORD")
  MONGO_HOST: $(b64encode "mongodb")
  MONGO_PORT: $(b64encode "27017")
  MONGO_DATABASE: $(b64encode "analytics")
EOF

# 4. JWT secrets
echo -e "${YELLOW}Creating JWT secrets...${NC}"
JWT_SECRET_FILE=$(create_secret "jwt-secrets" "$NAMESPACE")
cat >> "$JWT_SECRET_FILE" <<EOF
  JWT_SECRET: $(b64encode "$JWT_SECRET")
  JWT_EXPIRATION: $(b64encode "86400000")
  JWT_REFRESH_EXPIRATION: $(b64encode "604800000")
  JWT_ISSUER: $(b64encode "intelliflow-platform")
EOF

# 5. Kafka secrets
echo -e "${YELLOW}Creating Kafka secrets...${NC}"
KAFKA_SECRET_FILE=$(create_secret "kafka-secrets" "$NAMESPACE")
cat >> "$KAFKA_SECRET_FILE" <<EOF
  KAFKA_BOOTSTRAP_SERVERS: $(b64encode "kafka:9092")
  KAFKA_CLIENT_ID: $(b64encode "intelliflow-platform")
  KAFKA_CONSUMER_GROUP: $(b64encode "intelliflow-consumers")
EOF

# 6. Notification secrets
echo -e "${YELLOW}Creating notification secrets...${NC}"
NOTIFICATION_SECRET_FILE=$(create_secret "notification-secrets" "$NAMESPACE")
cat >> "$NOTIFICATION_SECRET_FILE" <<EOF
  SMTP_HOST: $(b64encode "smtp.gmail.com")
  SMTP_PORT: $(b64encode "587")
  SMTP_USERNAME: $(b64encode "notifications@intelliflow.com")
  SMTP_PASSWORD: $(b64encode "$SMTP_PASSWORD")
  SMTP_TLS_ENABLED: $(b64encode "true")
  SLACK_WEBHOOK_URL: $(b64encode "https://hooks.slack.com/services/CHANGE_ME")
  FROM_EMAIL: $(b64encode "IntelliFlow Platform <noreply@intelliflow.com>")
EOF

# 7. AWS secrets (placeholder)
echo -e "${YELLOW}Creating AWS secrets...${NC}"
AWS_SECRET_FILE=$(create_secret "aws-secrets" "$NAMESPACE")
cat >> "$AWS_SECRET_FILE" <<EOF
  AWS_ACCESS_KEY_ID: $(b64encode "AKIAIOSFODNN7EXAMPLE")
  AWS_SECRET_ACCESS_KEY: $(b64encode "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
  AWS_REGION: $(b64encode "us-west-2")
  S3_BUCKET_NAME: $(b64encode "intelliflow-models")
EOF

# 8. MLflow secrets
echo -e "${YELLOW}Creating MLflow secrets...${NC}"
MLFLOW_SECRET_FILE=$(create_secret "mlflow-secrets" "$NAMESPACE")
cat >> "$MLFLOW_SECRET_FILE" <<EOF
  MLFLOW_TRACKING_URI: $(b64encode "http://mlflow.intelliflow.com")
  MLFLOW_S3_ENDPOINT_URL: $(b64encode "https://s3.us-west-2.amazonaws.com")
  MLFLOW_TRACKING_USERNAME: $(b64encode "mlflow")
  MLFLOW_TRACKING_PASSWORD: $(b64encode "$MLFLOW_PASSWORD")
EOF

# 9. Monitoring secrets
echo -e "${YELLOW}Creating monitoring secrets...${NC}"
MONITORING_SECRET_FILE=$(create_secret "monitoring-secrets" "$NAMESPACE")
cat >> "$MONITORING_SECRET_FILE" <<EOF
  GRAFANA_API_KEY: $(b64encode "$GRAFANA_API_KEY")
  GRAFANA_USERNAME: $(b64encode "admin")
  GRAFANA_PASSWORD: $(b64encode "$(generate_password 16)")
  PROMETHEUS_API_TOKEN: $(b64encode "$(generate_password 32)")
EOF

# Create environment file for local development
echo -e "${YELLOW}Creating .env file for local development...${NC}"
cat > ".env.local" <<EOF
# IntelliFlow AI Platform - Local Development Environment
# Generated on $(date)

# Database Configuration
DB_USERNAME=intelliflow_user
DB_PASSWORD=$DB_PASSWORD
DB_HOST=localhost
DB_PORT=5432

# Redis Configuration
REDIS_PASSWORD=$REDIS_PASSWORD
REDIS_HOST=localhost
REDIS_PORT=6379

# MongoDB Configuration
MONGO_USERNAME=intelliflow_analytics
MONGO_PASSWORD=$MONGO_PASSWORD
MONGO_HOST=localhost
MONGO_PORT=27017

# JWT Configuration
JWT_SECRET=$JWT_SECRET
JWT_EXPIRATION=86400000
JWT_REFRESH_EXPIRATION=604800000

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Mail Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=notifications@intelliflow.com
SMTP_PASSWORD=$SMTP_PASSWORD

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_TRACKING_PASSWORD=$MLFLOW_PASSWORD

# Monitoring Configuration
GRAFANA_API_KEY=$GRAFANA_API_KEY

# Application Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
EUREKA_URL=http://localhost:8761/eureka
EOF

# Create summary file
echo -e "${YELLOW}Creating secret summary...${NC}"
cat > "$OUTPUT_DIR/SECRET_SUMMARY.md" <<EOF
# IntelliFlow AI Platform - Secrets Summary

**Generated on:** $(date)
**Namespace:** $NAMESPACE

## Secrets Created

1. **database-secrets** - PostgreSQL connection details
2. **redis-secrets** - Redis cache configuration
3. **mongodb-secrets** - MongoDB analytics database
4. **jwt-secrets** - JWT token signing and validation
5. **kafka-secrets** - Kafka messaging configuration
6. **notification-secrets** - Email and Slack notifications
7. **aws-secrets** - AWS service credentials (PLACEHOLDER)
8. **mlflow-secrets** - MLflow experiment tracking
9. **monitoring-secrets** - Grafana and Prometheus

## Security Notes

⚠️  **IMPORTANT SECURITY REMINDERS:**

1. **Change AWS credentials** in aws-secrets.yaml before production use
2. **Update Slack webhook URL** in notification-secrets.yaml
3. **Rotate secrets regularly** (every 90 days recommended)
4. **Use HashiCorp Vault** or AWS Secrets Manager in production
5. **Never commit these files to git** - add to .gitignore
6. **Restrict access** to these secret files (chmod 600)

## Applying Secrets

\`\`\`bash
# Apply all secrets
kubectl apply -f $OUTPUT_DIR/

# Verify secrets
kubectl get secrets -n $NAMESPACE

# View secret (without values)
kubectl describe secret database-secrets -n $NAMESPACE
\`\`\`

## Production Recommendations

1. Use external secret management (Vault, AWS Secrets Manager)
2. Enable secret encryption at rest in etcd
3. Use RBAC to restrict secret access
4. Implement secret rotation policies
5. Monitor secret access with audit logs
EOF

echo -e "${GREEN}✓ Secret generation completed!${NC}"
echo
echo -e "${BLUE}Files created:${NC}"
find "$OUTPUT_DIR" -name "*.yaml" -o -name "*.md" | sort
echo
echo -e "${YELLOW}⚠️  Security Reminders:${NC}"
echo "1. Review and update placeholder values before production use"
echo "2. Add secret files to .gitignore"
echo "3. Set appropriate file permissions (chmod 600)"
echo "4. Consider using external secret management for production"
echo
echo -e "${GREEN}Next steps:${NC}"
echo "1. Review generated secrets in $OUTPUT_DIR/"
echo "2. Update placeholder values (AWS credentials, Slack webhook, etc.)"
echo "3. Apply secrets: kubectl apply -f $OUTPUT_DIR/"
echo "4. For local development, use: cp .env.local .env"