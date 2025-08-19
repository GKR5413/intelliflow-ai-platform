# IntelliFlow AI Platform

A comprehensive microservices-based financial platform with advanced fraud detection, analytics, and notification capabilities.

## Repository

- GitHub: https://github.com/GKR5413/intelliflow-ai-platform

## Architecture Overview

The platform consists of 5 core microservices:

- **User Service** (Port 8081) - User management, authentication, and authorization
- **Transaction Service** (Port 8082) - Payment processing and transaction management
- **Fraud Detection Service** (Port 8083) - ML-based fraud detection and risk scoring
- **Analytics Service** (Port 8084) - Data analytics, reporting, and business intelligence
- **Notification Service** (Port 8085) - Multi-channel notifications (Email, SMS, Push, WebSocket)

## Technology Stack

### Backend
- **Java 17** with **Spring Boot 3.2.0**
- **Spring Cloud 2023.0.0** for microservices
- **PostgreSQL 15** for relational data
- **Redis 7** for caching and session management
- **MongoDB 7** for analytics data
- **Apache Kafka** for event streaming
- **Docker & Docker Compose** for containerization

### Infrastructure & DevOps
- **Terraform** for AWS infrastructure as code
- **Amazon EKS** for Kubernetes orchestration
- **Amazon RDS** for managed PostgreSQL
- **Amazon ElastiCache** for managed Redis
- **Amazon DocumentDB** for MongoDB compatibility
- **Amazon MSK** for managed Kafka
- **Amazon S3** for object storage

### Monitoring & Observability
- **Prometheus** for metrics collection
- **Grafana** for visualization
- **Jaeger** for distributed tracing
- **Spring Boot Actuator** for application metrics

## 🎨 **NEW: Beautiful Web UI Available!**

The IntelliFlow AI Platform now includes a **comprehensive React-based web interface**!

### **🌐 Access the Web UI:**
- **URL**: http://localhost:3001
- **Demo Login**: admin / admin
- **Features**: Complete transaction management, real-time fraud detection, interactive analytics, and more!

See [WEB_UI_GUIDE.md](./WEB_UI_GUIDE.md) for detailed information.

## 🌐 **NEW: Production Server Deployment!**

Deploy the complete platform to your production server with one command:

### **🚀 Quick Server Deployment:**
```bash
# Deploy to your server
./deployment/deploy-to-server.sh --host YOUR_SERVER_IP --user root --ssl --domain your-domain.com
```

### **📊 Real-time Server Monitoring:**
```bash
# Monitor your production server
./deployment/monitor-server.sh --host YOUR_SERVER_IP
```

See [SERVER_DEPLOYMENT_GUIDE.md](./deployment/SERVER_DEPLOYMENT_GUIDE.md) for complete instructions.

---

## Quick Start

### Prerequisites
- Docker Desktop
- Java 17 JDK
- Maven 3.9+
- AWS CLI (for production deployment)
- Terraform (for infrastructure)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/GKR5413/intelliflow-ai-platform.git
   cd intelliflow-ai-platform
   ```

2. **Start infrastructure services**
   ```bash
   docker-compose up -d postgres redis mongodb kafka zookeeper eureka-server
   ```

3. **Build all services**
   ```bash
   # Build each service
   cd user-service && mvn clean install && cd ..
   cd transaction-service && mvn clean install && cd ..
   cd fraud-detection-service && mvn clean install && cd ..
   cd analytics-service && mvn clean install && cd ..
   cd notification-service && mvn clean install && cd ..
   ```

4. **Start all services**
   ```bash
   docker-compose up -d
   ```

5. **Verify services are running**
   ```bash
   # Check service health
   curl http://localhost:8081/actuator/health                 # User Service
   curl http://localhost:8082/actuator/health                 # Transaction Service
   curl http://localhost:8083/api/v1/actuator/health          # Fraud Detection Service
   curl http://localhost:8084/actuator/health                 # Analytics Service
   curl http://localhost:8085/actuator/health                 # Notification Service
   ```

### Development Tools Access

- **Eureka Dashboard**: http://localhost:8761
- **Kafka UI**: http://localhost:8080
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Jaeger**: http://localhost:16686
- **MailHog**: http://localhost:8025

## Database Schema

### User Service Database (user_db)

**Tables:**
- `users` - Main user accounts with authentication details
- `roles` - User roles (ADMIN, USER, MANAGER, SUPPORT)
- `permissions` - Granular permissions for resources
- `user_roles` - User-role assignments
- `role_permissions` - Role-permission mappings
- `user_sessions` - Active user sessions with device tracking
- `user_activities` - Audit log of user activities

### Transaction Service Database (transaction_db)

**Tables:**
- `transactions` - Core transaction data with comprehensive metadata
  - Payment processing details
  - Fraud detection scores
  - Retry mechanisms
  - Settlement tracking
  - Device fingerprinting

### Fraud Detection Service Database (fraud_db)

**Tables:**
- `fraud_rules` - Configurable fraud detection rules
- `fraud_alerts` - Generated fraud alerts
- `risk_profiles` - User risk scoring profiles
- `ml_models` - Machine learning model metadata

### Analytics Service Database (analytics_db + MongoDB)

**PostgreSQL Tables:**
- `analytics_reports` - Generated reports metadata
- `data_aggregations` - Pre-computed aggregations

**MongoDB Collections:**
- `transaction_events` - Real-time transaction event stream
- `user_behavior` - User behavior analytics
- `business_metrics` - KPIs and business metrics

### Notification Service Database (notification_db)

**Tables:**
- `notifications` - Notification queue and history
- `notification_templates` - Message templates
- `notification_preferences` - User notification preferences
- `delivery_status` - Delivery tracking and status

## API Documentation

### User Service API Endpoints

```
POST /api/v1/auth/register - User registration
POST /api/v1/auth/login - User authentication
POST /api/v1/auth/logout - User logout
GET /api/v1/users/profile - Get user profile
PUT /api/v1/users/profile - Update user profile
POST /api/v1/auth/forgot-password - Password reset request
POST /api/v1/auth/reset-password - Password reset confirmation
```

### Transaction Service API Endpoints

```
POST /api/v1/transactions - Create new transaction
GET /api/v1/transactions/{id} - Get transaction details
GET /api/v1/transactions/user/{userId} - Get user transactions
PUT /api/v1/transactions/{id}/status - Update transaction status
POST /api/v1/transactions/{id}/refund - Process refund
GET /api/v1/transactions/search - Search transactions
```

### Fraud Detection Service API Endpoints

```
POST /api/v1/fraud/check - Perform fraud check
GET /api/v1/fraud/score/{transactionId} - Get risk score
PUT /api/v1/fraud/rules - Update fraud rules
GET /api/v1/fraud/alerts - Get fraud alerts
POST /api/v1/fraud/whitelist - Add to whitelist
```

## Infrastructure Deployment

### AWS Infrastructure Setup

1. **Configure AWS credentials**
   ```bash
   aws configure
   ```

2. **Initialize Terraform**
   ```bash
   cd terraform
   terraform init
   ```

3. **Plan infrastructure**
   ```bash
   terraform plan -var="environment=dev"
   ```

4. **Deploy infrastructure**
   ```bash
   terraform apply -var="environment=dev"
   ```

### Kubernetes Deployment

After infrastructure is deployed:

1. **Configure kubectl**
   ```bash
   aws eks update-kubeconfig --name intelliflow-dev --region us-west-2
   ```

2. **Deploy services**
   ```bash
   kubectl apply -f k8s/
   ```

## Configuration

### Environment Variables

Each service supports the following environment variables:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=service_db
DB_USERNAME=postgres
DB_PASSWORD=postgres

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Security
JWT_SECRET=your-secret-key
JWT_EXPIRATION=86400000

# Service Discovery
EUREKA_URL=http://localhost:8761/eureka
```

### Application Profiles

- `dev` - Development environment with debug logging
- `staging` - Staging environment for testing
- `prod` - Production environment with optimized settings

## Security Features

- JWT-based authentication with refresh tokens
- Role-based access control (RBAC)
- Session management with device tracking
- Password reset with secure tokens
- Email verification
- Two-factor authentication support
- API rate limiting
- Request signing and validation
- Database encryption at rest
- TLS encryption in transit

## Monitoring & Logging

### Metrics
- Application metrics via Spring Boot Actuator
- Business metrics (transaction volumes, success rates)
- System metrics (CPU, memory, database connections)
- Custom fraud detection metrics

### Logging
- Structured JSON logging
- Correlation IDs for request tracing
- Security audit logs
- Performance monitoring

### Alerting
- Failed transaction alerts
- Fraud detection alerts
- System health alerts
- Performance degradation alerts

## Development Guidelines

### Code Standards
- Java 17 features and best practices
- Spring Boot conventions
- Comprehensive unit and integration tests
- API documentation with OpenAPI/Swagger
- Database migration scripts with Flyway

### Testing
- Unit tests with JUnit 5 and Mockito
- Integration tests with TestContainers
- Contract testing with Spring Cloud Contract
- Performance testing with JMeter

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request
5. Ensure all CI checks pass

---
