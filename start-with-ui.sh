#!/bin/bash

# 🎨 IntelliFlow AI Platform - Start with Web UI
# Complete platform startup including the beautiful React web interface

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}"
echo "  ___       _       _ _ _ _____ _                "
echo " |_ _|_ __ | |_ ___| | (_)  ___| | _____      __"
echo "  | || '_ \| __/ _ \ | | | |_  | |/ _ \ \ /\ / /"
echo "  | || | | | ||  __/ | | |  _| | | (_) \ V  V / "
echo " |___|_| |_|\__\___|_|_|_|_|   |_|\___/ \_/\_/  "
echo ""
echo "        🎨 Starting with Beautiful Web UI..."
echo -e "${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Docker is not running. Please start Docker Desktop first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker is running${NC}"

# Start the complete platform
echo -e "${BLUE}🚀 Starting IntelliFlow AI Platform with Web UI...${NC}"
docker-compose up -d

# Wait for services to be ready
echo -e "${YELLOW}⏳ Waiting for all services to start (this may take 2-3 minutes)...${NC}"
sleep 120

# Check service health
echo -e "${BLUE}🔍 Checking service health...${NC}"

services=(
    "web-ui:3000"
    "user-service:8081"
    "transaction-service:8082"
    "fraud-detection-service:8083"
    "analytics-service:8084"
    "notification-service:8085"
)

all_healthy=true

for service in "${services[@]}"; do
    service_name=${service%:*}
    port=${service#*:}
    
    if docker-compose ps | grep -q "$service_name.*Up"; then
        echo -e "${GREEN}✅ $service_name is running on port $port${NC}"
    else
        echo -e "${YELLOW}⚠️  $service_name may still be starting...${NC}"
        all_healthy=false
    fi
done

echo ""
echo -e "${PURPLE}🎉 IntelliFlow AI Platform with Web UI is ready!${NC}"
echo ""

# Display access information
echo -e "${BLUE}🌐 Access URLs:${NC}"
echo ""
echo -e "${GREEN}🎨 WEB UI (Primary Interface):${NC}"
echo "   http://localhost:3000"
echo "   📧 Demo Login: admin / admin"
echo ""
echo -e "${BLUE}📊 Monitoring Dashboards:${NC}"
echo "   • Grafana:     http://localhost:3000 (admin/admin123)"
echo "   • Prometheus:  http://localhost:9090"
echo "   • Jaeger:      http://localhost:16686"
echo "   • MailHog:     http://localhost:8025"
echo "   • Kafka UI:    http://localhost:8080"
echo ""
echo -e "${BLUE}🔧 Direct API Access:${NC}"
echo "   • User Service:        http://localhost:8081/api/v1"
echo "   • Transaction Service: http://localhost:8082/api/v1"
echo "   • Fraud Detection:     http://localhost:8083"
echo "   • Analytics Service:   http://localhost:8084/api/v1"
echo "   • Notification Service: http://localhost:8085/api/v1"
echo ""

# Show Web UI features
echo -e "${PURPLE}🎯 Web UI Features:${NC}"
echo "   ✨ Beautiful Material Design interface"
echo "   🔐 Secure authentication with demo login"
echo "   💳 Step-by-step transaction creation wizard"
echo "   🛡️  Real-time fraud detection with ML scoring"
echo "   📊 Interactive analytics and charts"
echo "   🔔 Smart notification system"
echo "   📱 Fully responsive design"
echo ""

# Instructions
echo -e "${BLUE}🚀 Getting Started:${NC}"
echo "   1. Open your browser to: ${GREEN}http://localhost:3000${NC}"
echo "   2. Click 'Demo Login' or use: admin / admin"
echo "   3. Explore the dashboard and create transactions"
echo "   4. Watch real-time fraud detection in action"
echo ""

# Demo workflow
echo -e "${PURPLE}🎬 Try This Demo Workflow:${NC}"
echo "   1. 🔐 Login with demo credentials"
echo "   2. 📊 View the dashboard overview"
echo "   3. 💳 Create a new transaction (\$250 payment)"
echo "   4. 🛡️  Watch ML fraud detection analyze it"
echo "   5. 📈 Check analytics and service status"
echo "   6. 🔔 View notifications in the notification center"
echo ""

if [ "$all_healthy" = true ]; then
    echo -e "${GREEN}🎉 All services are healthy and ready to use!${NC}"
    echo ""
    echo -e "${BLUE}🌟 The beautiful Web UI is your main interface - enjoy exploring!${NC}"
else
    echo -e "${YELLOW}⚠️  Some services may still be starting.${NC}"
    echo "   💡 Wait a few more minutes and refresh the web UI"
    echo "   🔧 Check logs: docker-compose logs [service-name]"
fi

echo ""
echo -e "${BLUE}🛑 To stop the platform: docker-compose down${NC}"
echo ""
echo -e "${PURPLE}📚 For more information:${NC}"
echo "   • WEB_UI_GUIDE.md - Complete web interface guide"
echo "   • QUICK_START_GUIDE.md - Detailed platform instructions"
echo "   • demo.sh - Interactive command-line demo"
echo ""
echo -e "${GREEN}Happy exploring! 🚀${NC}"
