#!/bin/bash

# 🚀 IntelliFlow AI Platform - Quick Start Script
# This script starts the entire platform in the correct order

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "  ___       _       _ _ _ _____ _                "
echo " |_ _|_ __ | |_ ___| | (_)  ___| | _____      __"
echo "  | || '_ \| __/ _ \ | | | |_  | |/ _ \ \ /\ / /"
echo "  | || | | | ||  __/ | | |  _| | | (_) \ V  V / "
echo " |___|_| |_|\__\___|_|_|_|_|   |_|\___/ \_/\_/  "
echo ""
echo "        🚀 Starting AI Platform..."
echo -e "${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Docker is not running. Please start Docker Desktop first.${NC}"
    exit 1
fi

# Check if docker-compose exists
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}⚠️  docker-compose not found. Please install Docker Compose.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker is running${NC}"

# Step 1: Start infrastructure services first
echo -e "${BLUE}📦 Starting infrastructure services...${NC}"
docker-compose up -d postgres redis mongodb kafka zookeeper eureka-server

# Wait for infrastructure to be ready
echo -e "${YELLOW}⏳ Waiting for infrastructure services to be ready...${NC}"
sleep 30

# Step 2: Start monitoring services
echo -e "${BLUE}📊 Starting monitoring services...${NC}"
docker-compose up -d prometheus grafana jaeger mailhog kafka-ui

# Step 3: Start microservices
echo -e "${BLUE}🚀 Starting microservices...${NC}"
docker-compose up -d user-service transaction-service fraud-detection-service analytics-service notification-service

# Wait for services to start
echo -e "${YELLOW}⏳ Waiting for all services to start...${NC}"
sleep 45

# Check service health
echo -e "${BLUE}🔍 Checking service health...${NC}"

services=(
    "postgres:5432"
    "redis:6379" 
    "mongodb:27017"
    "kafka:9092"
    "eureka-server:8761"
    "user-service:8081"
    "transaction-service:8082"
    "fraud-detection-service:8083"
    "analytics-service:8084"
    "notification-service:8085"
    "prometheus:9090"
    "grafana:3000"
)

all_healthy=true

for service in "${services[@]}"; do
    service_name=${service%:*}
    port=${service#*:}
    
    if docker-compose ps | grep -q "$service_name.*Up"; then
        echo -e "${GREEN}✅ $service_name is running${NC}"
    else
        echo -e "${YELLOW}⚠️  $service_name may not be ready yet${NC}"
        all_healthy=false
    fi
done

echo ""
echo -e "${GREEN}🎉 IntelliFlow AI Platform is starting up!${NC}"
echo ""
echo -e "${BLUE}📊 Access URLs:${NC}"
echo "• User Service:        http://localhost:8081/api/v1"
echo "• Transaction Service: http://localhost:8082/api/v1"
echo "• Fraud Detection:     http://localhost:8083"
echo "• Analytics Service:   http://localhost:8084/api/v1"
echo "• Notification Service: http://localhost:8085/api/v1"
echo ""
echo "• Grafana Dashboard:   http://localhost:3000 (admin/admin123)"
echo "• Prometheus:          http://localhost:9090"
echo "• Jaeger Tracing:      http://localhost:16686"
echo "• Email Testing:       http://localhost:8025"
echo "• Kafka UI:            http://localhost:8080"
echo ""
echo -e "${BLUE}🧪 To run the interactive demo:${NC}"
echo "./demo.sh"
echo ""
echo -e "${BLUE}📚 For detailed instructions, see:${NC}"
echo "• QUICK_START_GUIDE.md"
echo "• README.md"
echo ""

if [ "$all_healthy" = true ]; then
    echo -e "${GREEN}✅ All services are healthy and ready to use!${NC}"
else
    echo -e "${YELLOW}⚠️  Some services may still be starting. Wait a few more minutes and check service logs if needed.${NC}"
    echo "• Check logs: docker-compose logs [service-name]"
    echo "• Check status: docker-compose ps"
fi

echo ""
echo -e "${BLUE}🛑 To stop the platform: docker-compose down${NC}"
