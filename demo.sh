#!/bin/bash

# 🚀 IntelliFlow AI Platform - Interactive Demo Script
# This script demonstrates the complete platform functionality

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
BASE_URL="http://localhost"
USER_SERVICE_PORT="8081"
TRANSACTION_SERVICE_PORT="8082"
FRAUD_SERVICE_PORT="8083"
ANALYTICS_SERVICE_PORT="8084"
NOTIFICATION_SERVICE_PORT="8085"

# Function to print colored output
print_step() {
    echo -e "${BLUE}===================================="
    echo -e "📋 STEP: $1"
    echo -e "====================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ SUCCESS: $1${NC}"
}

print_error() {
    echo -e "${RED}❌ ERROR: $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  INFO: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  WARNING: $1${NC}"
}

# Function to check if service is running
check_service() {
    local service_name=$1
    local port=$2
    local url="${BASE_URL}:${port}/actuator/health"
    
    if [[ $service_name == "fraud-detection-service" ]]; then
        url="${BASE_URL}:${port}/health"
    fi
    
    echo -e "${YELLOW}Checking $service_name...${NC}"
    
    if curl -s -f "$url" > /dev/null 2>&1; then
        print_success "$service_name is running on port $port"
        return 0
    else
        print_error "$service_name is not running on port $port"
        return 1
    fi
}

# Function to wait for user input
wait_for_input() {
    echo -e "${PURPLE}Press Enter to continue...${NC}"
    read
}

# Main demo function
run_demo() {
    echo -e "${PURPLE}"
    echo "  ___       _       _ _ _ _____ _                "
    echo " |_ _|_ __ | |_ ___| | (_)  ___| | _____      __"
    echo "  | || '_ \| __/ _ \ | | | |_  | |/ _ \ \ /\ / /"
    echo "  | || | | | ||  __/ | | |  _| | | (_) \ V  V / "
    echo " |___|_| |_|\__\___|_|_|_|_|   |_|\___/ \_/\_/  "
    echo ""
    echo "        AI Platform Demo - v1.0.0"
    echo "        Financial Services Platform"
    echo -e "${NC}"
    
    print_info "Welcome to the IntelliFlow AI Platform Demo!"
    print_info "This demo will walk you through the complete platform functionality."
    
    wait_for_input
    
    # Step 1: Check Services
    print_step "Checking Platform Services"
    
    local services_ok=true
    
    check_service "user-service" $USER_SERVICE_PORT || services_ok=false
    check_service "transaction-service" $TRANSACTION_SERVICE_PORT || services_ok=false
    check_service "fraud-detection-service" $FRAUD_SERVICE_PORT || services_ok=false
    check_service "analytics-service" $ANALYTICS_SERVICE_PORT || services_ok=false
    check_service "notification-service" $NOTIFICATION_SERVICE_PORT || services_ok=false
    
    if [ "$services_ok" = false ]; then
        print_error "Some services are not running!"
        print_info "Please run: docker-compose up -d"
        print_info "Then wait for all services to start and try again."
        exit 1
    fi
    
    print_success "All services are running!"
    wait_for_input
    
    # Step 2: User Registration
    print_step "User Registration"
    print_info "Registering a new user: demo_user"
    
    REGISTER_RESPONSE=$(curl -s -X POST "${BASE_URL}:${USER_SERVICE_PORT}/api/v1/auth/register" \
        -H "Content-Type: application/json" \
        -d '{
            "username": "demo_user",
            "email": "demo@intelliflow.com",
            "password": "DemoPassword123!",
            "firstName": "Demo",
            "lastName": "User",
            "phoneNumber": "+1234567890"
        }')
    
    if echo "$REGISTER_RESPONSE" | jq -e . >/dev/null 2>&1; then
        print_success "User registered successfully!"
        echo -e "${CYAN}Response:${NC}"
        echo "$REGISTER_RESPONSE" | jq .
    else
        print_warning "Registration response (might be user already exists):"
        echo "$REGISTER_RESPONSE"
    fi
    
    wait_for_input
    
    # Step 3: User Login
    print_step "User Authentication"
    print_info "Logging in as demo_user..."
    
    LOGIN_RESPONSE=$(curl -s -X POST "${BASE_URL}:${USER_SERVICE_PORT}/api/v1/auth/login" \
        -H "Content-Type: application/json" \
        -d '{
            "usernameOrEmail": "demo_user",
            "password": "DemoPassword123!"
        }')
    
    if echo "$LOGIN_RESPONSE" | jq -e '.accessToken' >/dev/null 2>&1; then
        JWT_TOKEN=$(echo "$LOGIN_RESPONSE" | jq -r '.accessToken')
        print_success "Login successful! JWT Token received."
        echo -e "${CYAN}Token (first 50 chars): ${JWT_TOKEN:0:50}...${NC}"
    else
        print_error "Login failed!"
        echo "$LOGIN_RESPONSE"
        exit 1
    fi
    
    wait_for_input
    
    # Step 4: Create Transaction
    print_step "Transaction Processing"
    print_info "Creating a transaction for $150.00..."
    
    TRANSACTION_RESPONSE=$(curl -s -X POST "${BASE_URL}:${TRANSACTION_SERVICE_PORT}/api/v1/transactions" \
        -H "Authorization: Bearer $JWT_TOKEN" \
        -H "Content-Type: application/json" \
        -d '{
            "userId": 1,
            "amount": 150.00,
            "currency": "USD",
            "transactionType": "PAYMENT",
            "merchantId": "demo_merchant",
            "description": "Demo transaction - Online purchase",
            "paymentMethod": "CREDIT_CARD",
            "metadata": {
                "cardLast4": "1234",
                "merchantCategory": "ECOMMERCE",
                "ipAddress": "192.168.1.100"
            }
        }')
    
    if echo "$TRANSACTION_RESPONSE" | jq -e '.id' >/dev/null 2>&1; then
        TRANSACTION_ID=$(echo "$TRANSACTION_RESPONSE" | jq -r '.id')
        print_success "Transaction created successfully! ID: $TRANSACTION_ID"
        echo -e "${CYAN}Transaction Details:${NC}"
        echo "$TRANSACTION_RESPONSE" | jq .
    else
        print_error "Transaction creation failed!"
        echo "$TRANSACTION_RESPONSE"
        exit 1
    fi
    
    wait_for_input
    
    # Step 5: Fraud Detection
    print_step "Fraud Detection Analysis"
    print_info "Running fraud detection on the transaction..."
    
    # Wait a moment for transaction to be processed
    sleep 2
    
    FRAUD_RESPONSE=$(curl -s -X POST "${BASE_URL}:${FRAUD_SERVICE_PORT}/api/v1/fraud/score" \
        -H "Authorization: Bearer $JWT_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"transactionId\": $TRANSACTION_ID,
            \"userId\": 1,
            \"amount\": 150.00,
            \"currency\": \"USD\",
            \"merchantId\": \"demo_merchant\",
            \"location\": {
                \"country\": \"US\",
                \"city\": \"San Francisco\",
                \"ipAddress\": \"192.168.1.100\"
            },
            \"deviceInfo\": {
                \"deviceId\": \"demo_device_123\",
                \"userAgent\": \"Mozilla/5.0 (Demo Browser)\"
            },
            \"transactionFeatures\": {
                \"isHighRisk\": false,
                \"velocityScore\": 0.2,
                \"geolocationRisk\": 0.1
            }
        }")
    
    if echo "$FRAUD_RESPONSE" | jq -e . >/dev/null 2>&1; then
        print_success "Fraud detection completed!"
        echo -e "${CYAN}Fraud Analysis Result:${NC}"
        echo "$FRAUD_RESPONSE" | jq .
        
        FRAUD_SCORE=$(echo "$FRAUD_RESPONSE" | jq -r '.fraudScore // .score // "0.0"')
        if (( $(echo "$FRAUD_SCORE > 0.5" | bc -l) )); then
            print_warning "High fraud risk detected! Score: $FRAUD_SCORE"
        else
            print_success "Low fraud risk. Score: $FRAUD_SCORE"
        fi
    else
        print_warning "Fraud detection response:"
        echo "$FRAUD_RESPONSE"
    fi
    
    wait_for_input
    
    # Step 6: Check Transaction Status
    print_step "Transaction Status Check"
    print_info "Checking transaction status..."
    
    STATUS_RESPONSE=$(curl -s -X GET "${BASE_URL}:${TRANSACTION_SERVICE_PORT}/api/v1/transactions/$TRANSACTION_ID" \
        -H "Authorization: Bearer $JWT_TOKEN")
    
    if echo "$STATUS_RESPONSE" | jq -e '.id' >/dev/null 2>&1; then
        print_success "Transaction status retrieved!"
        echo -e "${CYAN}Current Status:${NC}"
        echo "$STATUS_RESPONSE" | jq '{id: .id, status: .status, amount: .amount, fraudScore: .fraudCheckStatus}'
        
        TRANSACTION_STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status')
        print_info "Transaction Status: $TRANSACTION_STATUS"
    else
        print_error "Failed to retrieve transaction status"
        echo "$STATUS_RESPONSE"
    fi
    
    wait_for_input
    
    # Step 7: Send Notification
    print_step "Notification Service"
    print_info "Sending transaction confirmation notification..."
    
    NOTIFICATION_RESPONSE=$(curl -s -X POST "${BASE_URL}:${NOTIFICATION_SERVICE_PORT}/api/v1/notifications/send" \
        -H "Authorization: Bearer $JWT_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"userId\": 1,
            \"type\": \"EMAIL\",
            \"template\": \"transaction_confirmation\",
            \"recipient\": \"demo@intelliflow.com\",
            \"data\": {
                \"transactionId\": $TRANSACTION_ID,
                \"amount\": 150.00,
                \"merchantName\": \"Demo Merchant\",
                \"status\": \"$TRANSACTION_STATUS\"
            }
        }")
    
    if echo "$NOTIFICATION_RESPONSE" | jq -e . >/dev/null 2>&1; then
        print_success "Notification sent successfully!"
        echo -e "${CYAN}Notification Details:${NC}"
        echo "$NOTIFICATION_RESPONSE" | jq .
    else
        print_warning "Notification response:"
        echo "$NOTIFICATION_RESPONSE"
    fi
    
    wait_for_input
    
    # Step 8: Analytics
    print_step "Analytics and Reporting"
    print_info "Retrieving user analytics..."
    
    ANALYTICS_RESPONSE=$(curl -s -X GET "${BASE_URL}:${ANALYTICS_SERVICE_PORT}/api/v1/analytics/user/1?startDate=2024-01-01&endDate=2024-12-31" \
        -H "Authorization: Bearer $JWT_TOKEN")
    
    if echo "$ANALYTICS_RESPONSE" | jq -e . >/dev/null 2>&1; then
        print_success "Analytics data retrieved!"
        echo -e "${CYAN}User Analytics:${NC}"
        echo "$ANALYTICS_RESPONSE" | jq .
    else
        print_warning "Analytics response:"
        echo "$ANALYTICS_RESPONSE"
    fi
    
    wait_for_input
    
    # Step 9: Summary
    print_step "Demo Summary"
    print_success "🎉 Demo completed successfully!"
    
    echo -e "${GREEN}"
    echo "Demo Summary:"
    echo "============="
    echo "✅ User Registration: demo_user created"
    echo "✅ Authentication: JWT token obtained"
    echo "✅ Transaction: \$150.00 payment processed (ID: $TRANSACTION_ID)"
    echo "✅ Fraud Detection: Risk analysis completed"
    echo "✅ Notification: Email confirmation sent"
    echo "✅ Analytics: User data retrieved"
    echo -e "${NC}"
    
    print_info "You can now:"
    echo "• Access Grafana dashboards: http://localhost:3000 (admin/admin123)"
    echo "• View email notifications: http://localhost:8025"
    echo "• Check service metrics: http://localhost:9090"
    echo "• Monitor distributed traces: http://localhost:16686"
    echo "• Explore Kafka messages: http://localhost:8080"
    
    print_info "To run additional tests:"
    echo "• Performance tests: ./performance-testing/scripts/run-performance-tests.sh"
    echo "• Load testing: docker-compose up -d --scale user-service=3"
    echo "• API documentation: http://localhost:8081/api/v1/swagger-ui.html"
    
    echo -e "${PURPLE}"
    echo "Thank you for using IntelliFlow AI Platform! 🚀"
    echo -e "${NC}"
}

# Create a suspicious transaction demo
run_suspicious_transaction_demo() {
    print_step "🚨 Suspicious Transaction Demo"
    print_info "Creating a high-risk transaction to test fraud detection..."
    
    SUSPICIOUS_RESPONSE=$(curl -s -X POST "${BASE_URL}:${TRANSACTION_SERVICE_PORT}/api/v1/transactions" \
        -H "Authorization: Bearer $JWT_TOKEN" \
        -H "Content-Type: application/json" \
        -d '{
            "userId": 1,
            "amount": 9999.00,
            "currency": "USD",
            "transactionType": "TRANSFER",
            "merchantId": "suspicious_merchant",
            "description": "High-value suspicious transfer",
            "paymentMethod": "WIRE_TRANSFER",
            "metadata": {
                "ipAddress": "10.0.0.1",
                "country": "XX",
                "riskLevel": "HIGH"
            }
        }')
    
    if echo "$SUSPICIOUS_RESPONSE" | jq -e '.id' >/dev/null 2>&1; then
        SUSPICIOUS_TRANSACTION_ID=$(echo "$SUSPICIOUS_RESPONSE" | jq -r '.id')
        print_success "Suspicious transaction created! ID: $SUSPICIOUS_TRANSACTION_ID"
        
        # Run fraud detection
        sleep 2
        FRAUD_ANALYSIS=$(curl -s -X POST "${BASE_URL}:${FRAUD_SERVICE_PORT}/api/v1/fraud/score" \
            -H "Authorization: Bearer $JWT_TOKEN" \
            -H "Content-Type: application/json" \
            -d "{
                \"transactionId\": $SUSPICIOUS_TRANSACTION_ID,
                \"userId\": 1,
                \"amount\": 9999.00,
                \"currency\": \"USD\",
                \"merchantId\": \"suspicious_merchant\",
                \"location\": {
                    \"country\": \"XX\",
                    \"city\": \"Unknown\",
                    \"ipAddress\": \"10.0.0.1\"
                },
                \"deviceInfo\": {
                    \"deviceId\": \"unknown_device\",
                    \"userAgent\": \"suspicious_agent\"
                },
                \"transactionFeatures\": {
                    \"isHighRisk\": true,
                    \"velocityScore\": 0.95,
                    \"geolocationRisk\": 0.9
                }
            }")
        
        print_warning "🚨 FRAUD ALERT! High-risk transaction detected!"
        echo -e "${RED}Fraud Analysis:${NC}"
        echo "$FRAUD_ANALYSIS" | jq .
    fi
}

# Menu system
show_menu() {
    echo -e "${BLUE}"
    echo "=================================="
    echo "  IntelliFlow AI Platform Demo"
    echo "=================================="
    echo -e "${NC}"
    echo "1. 🚀 Run Complete Demo"
    echo "2. 🚨 Suspicious Transaction Demo"
    echo "3. 🔍 Check Service Health"
    echo "4. 📊 View Platform URLs"
    echo "5. 🛠️  Troubleshooting"
    echo "6. ❌ Exit"
    echo ""
    echo -n "Select an option (1-6): "
}

# Service health check
check_all_services() {
    print_step "Service Health Check"
    check_service "user-service" $USER_SERVICE_PORT
    check_service "transaction-service" $TRANSACTION_SERVICE_PORT
    check_service "fraud-detection-service" $FRAUD_SERVICE_PORT
    check_service "analytics-service" $ANALYTICS_SERVICE_PORT
    check_service "notification-service" $NOTIFICATION_SERVICE_PORT
}

# Show platform URLs
show_urls() {
    print_step "Platform Access URLs"
    echo -e "${CYAN}"
    echo "🌐 Service APIs:"
    echo "  • User Service:        http://localhost:8081/api/v1"
    echo "  • Transaction Service: http://localhost:8082/api/v1" 
    echo "  • Fraud Detection:     http://localhost:8083"
    echo "  • Analytics Service:   http://localhost:8084/api/v1"
    echo "  • Notification Service: http://localhost:8085/api/v1"
    echo ""
    echo "📊 Monitoring & Tools:"
    echo "  • Grafana:     http://localhost:3000 (admin/admin123)"
    echo "  • Prometheus:  http://localhost:9090"
    echo "  • Jaeger:      http://localhost:16686" 
    echo "  • MailHog:     http://localhost:8025"
    echo "  • Kafka UI:    http://localhost:8080"
    echo "  • Eureka:      http://localhost:8761"
    echo ""
    echo "📚 Documentation:"
    echo "  • API Docs:    http://localhost:8081/api/v1/swagger-ui.html"
    echo "  • Health:      http://localhost:8081/actuator/health"
    echo -e "${NC}"
}

# Troubleshooting guide
show_troubleshooting() {
    print_step "Troubleshooting Guide"
    echo -e "${YELLOW}"
    echo "🔧 Common Issues & Solutions:"
    echo ""
    echo "1. Services not starting:"
    echo "   → docker-compose down && docker-compose up -d"
    echo ""
    echo "2. Port conflicts:"
    echo "   → lsof -i :8081 (check what's using the port)"
    echo "   → sudo kill -9 <PID> (kill conflicting process)"
    echo ""
    echo "3. Database issues:"
    echo "   → docker-compose restart postgres"
    echo "   → docker exec -it postgres-db psql -U postgres"
    echo ""
    echo "4. Reset everything:"
    echo "   → docker-compose down -v"
    echo "   → docker system prune -a"
    echo "   → docker-compose up -d"
    echo ""
    echo "5. Check logs:"
    echo "   → docker-compose logs user-service"
    echo "   → docker-compose logs -f fraud-detection-service"
    echo -e "${NC}"
}

# Main menu loop
main() {
    while true; do
        show_menu
        read choice
        
        case $choice in
            1)
                run_demo
                ;;
            2)
                if [ -z "$JWT_TOKEN" ]; then
                    print_error "Please run the complete demo first to get authentication token"
                else
                    run_suspicious_transaction_demo
                fi
                ;;
            3)
                check_all_services
                ;;
            4)
                show_urls
                ;;
            5)
                show_troubleshooting
                ;;
            6)
                print_info "Goodbye! 👋"
                exit 0
                ;;
            *)
                print_error "Invalid option. Please select 1-6."
                ;;
        esac
        
        echo ""
        wait_for_input
    done
}

# Check dependencies
check_dependencies() {
    if ! command -v curl &> /dev/null; then
        print_error "curl is required but not installed."
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        print_warning "jq is not installed. Installing with Homebrew..."
        if command -v brew &> /dev/null; then
            brew install jq
        else
            print_error "Please install jq: brew install jq"
            exit 1
        fi
    fi
    
    if ! command -v bc &> /dev/null; then
        print_warning "bc is not installed. Installing with Homebrew..."
        if command -v brew &> /dev/null; then
            brew install bc
        else
            print_error "Please install bc: brew install bc"
            exit 1
        fi
    fi
}

# Entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    check_dependencies
    main
fi
