# 🚀 IntelliFlow AI Platform - Quick Start Guide

## 🎯 **How to Run the Application**

### **Option 1: Local Development (Recommended for Testing)**

#### **1. Prerequisites**
```bash
# Install required tools
brew install docker docker-compose maven openjdk@17 python3

# Verify installations
docker --version
docker-compose --version
mvn --version
java --version
python3 --version
```

#### **2. Environment Setup**
```bash
# Clone or navigate to project
cd "/Users/spider_myan/Documents/IntelliFlow AI Platform"

# Create environment file from template
cp .env.template .env

# Edit the .env file with your configurations
nano .env
```

#### **3. Start Infrastructure Services**
```bash
# Start all infrastructure (databases, messaging, monitoring)
docker-compose up -d postgres redis mongodb kafka eureka-server prometheus grafana

# Wait for services to be ready (check health)
docker-compose ps
```

#### **4. Build and Start Microservices**
```bash
# Build all Java services
mvn clean package -DskipTests

# Start all microservices
docker-compose up -d user-service transaction-service fraud-detection-service analytics-service notification-service

# Check if all services are running
docker-compose ps
```

#### **5. Access the Platform**
- **API Gateway**: http://localhost:8080
- **User Service**: http://localhost:8081
- **Transaction Service**: http://localhost:8082
- **Fraud Detection**: http://localhost:8083
- **Analytics Service**: http://localhost:8084
- **Notification Service**: http://localhost:8085

---

### **Option 2: Kubernetes Deployment (Production-like)**

#### **1. Prerequisites**
```bash
# Install Kubernetes tools
brew install kubernetes-cli helm

# Start local Kubernetes (Docker Desktop or Minikube)
kubectl cluster-info
```

#### **2. Deploy with Helm**
```bash
# Create namespace
kubectl create namespace intelliflow-prod

# Deploy the platform
helm upgrade --install intelliflow ./helm/intelliflow-platform \
  --namespace intelliflow-prod \
  --set global.environment=development \
  --set global.imageTag=latest
```

#### **3. Access Services**
```bash
# Port forward to access locally
kubectl port-forward service/api-gateway 8080:80 -n intelliflow-prod

# Or get external IP (if using cloud)
kubectl get ingress -n intelliflow-prod
```

---

## 📝 **How to Provide Inputs & Test the Platform**

### **1. User Registration and Authentication**

#### **Register a New User**
```bash
curl -X POST http://localhost:8081/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "email": "john@example.com",
    "password": "SecurePassword123!",
    "firstName": "John",
    "lastName": "Doe",
    "phoneNumber": "+1234567890"
  }'
```

#### **Login to Get JWT Token**
```bash
# Login and save the token
JWT_TOKEN=$(curl -X POST http://localhost:8081/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "usernameOrEmail": "john_doe",
    "password": "SecurePassword123!"
  }' | jq -r '.accessToken')

echo "JWT Token: $JWT_TOKEN"
```

### **2. Transaction Processing**

#### **Create a Transaction**
```bash
curl -X POST http://localhost:8082/api/v1/transactions \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "userId": 1,
    "amount": 250.75,
    "currency": "USD",
    "transactionType": "PAYMENT",
    "merchantId": "merchant_123",
    "description": "Online purchase",
    "paymentMethod": "CREDIT_CARD",
    "metadata": {
      "cardLast4": "1234",
      "merchantCategory": "ECOMMERCE"
    }
  }'
```

#### **Check Transaction Status**
```bash
# Get transaction by ID
curl -X GET http://localhost:8082/api/v1/transactions/1 \
  -H "Authorization: Bearer $JWT_TOKEN"

# Get user transactions
curl -X GET http://localhost:8082/api/v1/transactions/user/1?page=0&size=10 \
  -H "Authorization: Bearer $JWT_TOKEN"
```

### **3. Fraud Detection Testing**

#### **Trigger Fraud Detection**
```bash
# Send a suspicious transaction for fraud scoring
curl -X POST http://localhost:8083/api/v1/fraud/score \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "transactionId": 1,
    "userId": 1,
    "amount": 10000.00,
    "currency": "USD",
    "merchantId": "suspicious_merchant",
    "location": {
      "country": "CN",
      "city": "Unknown",
      "ipAddress": "192.168.1.1"
    },
    "deviceInfo": {
      "deviceId": "unknown_device",
      "userAgent": "suspicious_agent"
    },
    "transactionFeatures": {
      "isHighRisk": true,
      "velocityScore": 0.95,
      "geolocationRisk": 0.8
    }
  }'
```

#### **Get Fraud Detection Results**
```bash
curl -X GET http://localhost:8083/api/v1/fraud/transaction/1/result \
  -H "Authorization: Bearer $JWT_TOKEN"
```

### **4. Analytics and Reporting**

#### **Get User Analytics**
```bash
curl -X GET "http://localhost:8084/api/v1/analytics/user/1?startDate=2024-01-01&endDate=2024-12-31" \
  -H "Authorization: Bearer $JWT_TOKEN"
```

#### **Get Transaction Analytics**
```bash
curl -X GET "http://localhost:8084/api/v1/analytics/transactions/summary?period=DAILY&days=30" \
  -H "Authorization: Bearer $JWT_TOKEN"
```

### **5. Notification Testing**

#### **Send Test Notification**
```bash
curl -X POST http://localhost:8085/api/v1/notifications/send \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "userId": 1,
    "type": "EMAIL",
    "template": "transaction_confirmation",
    "recipient": "john@example.com",
    "data": {
      "transactionId": 1,
      "amount": 250.75,
      "merchantName": "Test Merchant"
    }
  }'
```

---

## 🧪 **Pre-built Test Scenarios**

### **Scenario 1: Complete User Journey**
```bash
#!/bin/bash
# Run this script to test the complete user journey

# 1. Register user
echo "🔐 Registering new user..."
curl -X POST http://localhost:8081/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "test_user",
    "email": "test@example.com",
    "password": "TestPassword123!",
    "firstName": "Test",
    "lastName": "User"
  }'

# 2. Login
echo "🔑 Logging in..."
JWT_TOKEN=$(curl -s -X POST http://localhost:8081/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "usernameOrEmail": "test_user",
    "password": "TestPassword123!"
  }' | jq -r '.accessToken')

# 3. Create transaction
echo "💳 Creating transaction..."
TRANSACTION_ID=$(curl -s -X POST http://localhost:8082/api/v1/transactions \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "userId": 1,
    "amount": 100.00,
    "currency": "USD",
    "transactionType": "PAYMENT",
    "merchantId": "test_merchant"
  }' | jq -r '.id')

# 4. Check fraud score
echo "🔍 Checking fraud score..."
curl -X GET http://localhost:8083/api/v1/fraud/transaction/$TRANSACTION_ID/result \
  -H "Authorization: Bearer $JWT_TOKEN"

echo "✅ Complete user journey test finished!"
```

### **Scenario 2: Load Testing**
```bash
# Run performance tests
cd performance-testing
./scripts/run-performance-tests.sh --users 50 --duration 60
```

---

## 🖥️ **Web Interfaces**

### **Monitoring Dashboards**
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Jaeger Tracing**: http://localhost:16686
- **Kafka UI**: http://localhost:8080

### **Development Tools**
- **MailHog** (Email Testing): http://localhost:8025
- **Eureka Dashboard**: http://localhost:8761

---

## 📊 **API Documentation**

### **OpenAPI/Swagger Endpoints**
- **User Service**: http://localhost:8081/api/v1/swagger-ui.html
- **Transaction Service**: http://localhost:8082/api/v1/swagger-ui.html
- **Fraud Detection**: http://localhost:8083/docs
- **Analytics Service**: http://localhost:8084/api/v1/swagger-ui.html
- **Notification Service**: http://localhost:8085/api/v1/swagger-ui.html

### **Health Check Endpoints**
```bash
# Check all services health
curl http://localhost:8081/actuator/health  # User Service
curl http://localhost:8082/actuator/health  # Transaction Service
curl http://localhost:8083/health          # Fraud Detection (Python)
curl http://localhost:8084/actuator/health  # Analytics Service
curl http://localhost:8085/actuator/health  # Notification Service
```

---

## 🔧 **Troubleshooting**

### **Common Issues**

#### **Services Not Starting**
```bash
# Check Docker logs
docker-compose logs user-service
docker-compose logs postgres

# Restart specific service
docker-compose restart user-service
```

#### **Port Conflicts**
```bash
# Check what's using ports
lsof -i :8081
lsof -i :5432

# Kill conflicting processes
sudo kill -9 <PID>
```

#### **Database Connection Issues**
```bash
# Connect to PostgreSQL directly
docker exec -it postgres-db psql -U postgres -d user_db

# Check database tables
\dt
```

#### **Reset Everything**
```bash
# Stop all services and remove data
docker-compose down -v

# Remove all containers and images
docker system prune -a

# Start fresh
docker-compose up -d
```

---

## 🎯 **Sample Input Data**

### **Test Users**
```json
{
  "username": "admin",
  "email": "admin@intelliflow.com",
  "password": "admin",
  "firstName": "Admin",
  "lastName": "User",
  "roles": ["ADMIN"]
}

{
  "username": "johndoe",
  "email": "john.doe@example.com", 
  "password": "SecurePass123!",
  "firstName": "John",
  "lastName": "Doe"
}
```

### **Test Transactions**
```json
{
  "amount": 99.99,
  "currency": "USD",
  "transactionType": "PAYMENT",
  "merchantId": "amazon_store",
  "description": "Book purchase"
}

{
  "amount": 1500.00,
  "currency": "USD", 
  "transactionType": "TRANSFER",
  "merchantId": "suspicious_merchant",
  "description": "High-risk transaction"
}
```

---

## 🚀 **Quick Commands Cheat Sheet**

```bash
# Start everything
docker-compose up -d

# Stop everything  
docker-compose down

# View logs
docker-compose logs -f user-service

# Scale services
docker-compose up -d --scale user-service=3

# Check status
docker-compose ps

# Run tests
mvn test

# Deploy to Kubernetes
helm upgrade --install intelliflow ./helm/intelliflow-platform

# Performance test
./performance-testing/scripts/run-performance-tests.sh --users 100
```

---

**🎉 Your IntelliFlow AI Platform is now ready to use!**

Start with the local development option for testing, then move to Kubernetes for production-like deployment. Use the provided API examples to interact with all services and test the complete fraud detection workflow.
