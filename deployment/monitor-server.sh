#!/bin/bash

# 📊 IntelliFlow AI Platform - Server Monitoring Script
# Real-time monitoring and health checks for production deployment

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Configuration
SERVER_USER="${SERVER_USER:-root}"
SERVER_HOST="${SERVER_HOST}"
SERVER_PORT="${SERVER_PORT:-22}"
DEPLOYMENT_PATH="${DEPLOYMENT_PATH:-/opt/intelliflow}"
REFRESH_INTERVAL="${REFRESH_INTERVAL:-5}"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --host HOST          Server hostname or IP address"
    echo "  -u, --user USER          SSH username (default: root)"
    echo "  -p, --port PORT          SSH port (default: 22)"
    echo "  -i, --interval SECONDS   Refresh interval (default: 5)"
    echo "  --deployment-path PATH   Deployment path (default: /opt/intelliflow)"
    echo "  --help                   Show this help message"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            SERVER_HOST="$2"
            shift 2
            ;;
        -u|--user)
            SERVER_USER="$2"
            shift 2
            ;;
        -p|--port)
            SERVER_PORT="$2"
            shift 2
            ;;
        -i|--interval)
            REFRESH_INTERVAL="$2"
            shift 2
            ;;
        --deployment-path)
            DEPLOYMENT_PATH="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$SERVER_HOST" ]]; then
    echo -e "${RED}❌ Error: Server host is required${NC}"
    usage
    exit 1
fi

# Function to run SSH commands
ssh_exec() {
    ssh -p "$SERVER_PORT" "$SERVER_USER@$SERVER_HOST" "$1" 2>/dev/null
}

# Function to clear screen
clear_screen() {
    clear
    echo -e "${BLUE}"
    echo "  📊 IntelliFlow AI Platform - Live Monitoring"
    echo "  ============================================="
    echo -e "${NC}"
    echo "  Server: $SERVER_USER@$SERVER_HOST:$SERVER_PORT"
    echo "  Refresh: Every ${REFRESH_INTERVAL}s (Ctrl+C to exit)"
    echo ""
}

# Function to check service health
check_service_health() {
    local service_name="$1"
    local port="$2"
    local endpoint="${3:-health}"
    
    if ssh_exec "curl -s -f http://localhost:$port/$endpoint >/dev/null 2>&1"; then
        echo -e "${GREEN}✅${NC}"
    else
        echo -e "${RED}❌${NC}"
    fi
}

# Function to get container status
get_container_status() {
    local container_name="$1"
    local status=$(ssh_exec "docker inspect --format='{{.State.Status}}' $container_name 2>/dev/null" || echo "not found")
    
    case $status in
        "running")
            echo -e "${GREEN}🟢 Running${NC}"
            ;;
        "exited")
            echo -e "${RED}🔴 Exited${NC}"
            ;;
        "restarting")
            echo -e "${YELLOW}🟡 Restarting${NC}"
            ;;
        "paused")
            echo -e "${YELLOW}⏸️  Paused${NC}"
            ;;
        "not found")
            echo -e "${RED}❓ Not Found${NC}"
            ;;
        *)
            echo -e "${YELLOW}❓ $status${NC}"
            ;;
    esac
}

# Function to get memory usage
get_memory_usage() {
    ssh_exec "free | grep Mem | awk '{printf \"%.1f%%\", \$3/\$2 * 100.0}'"
}

# Function to get CPU usage
get_cpu_usage() {
    ssh_exec "top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\\([0-9.]*\\)%* id.*/\\1/' | awk '{printf \"%.1f%%\", 100 - \$1}'"
}

# Function to get disk usage
get_disk_usage() {
    ssh_exec "df -h / | awk 'NR==2{printf \"%s\", \$5}'"
}

# Function to get uptime
get_uptime() {
    ssh_exec "uptime | awk '{print \$3,\$4}' | sed 's/,//'"
}

# Function to get load average
get_load_average() {
    ssh_exec "uptime | awk -F'load average:' '{print \$2}'"
}

# Function to get container resource usage
get_container_resources() {
    local container_name="$1"
    ssh_exec "docker stats $container_name --no-stream --format 'table {{.CPUPerc}}\t{{.MemUsage}}' 2>/dev/null | tail -n 1" || echo "N/A	N/A"
}

# Main monitoring loop
monitor_server() {
    while true; do
        clear_screen
        
        # System Overview
        echo -e "${PURPLE}🖥️  System Overview${NC}"
        echo "────────────────────────────────────────────────────────────"
        printf "%-15s %s\n" "CPU Usage:" "$(get_cpu_usage)"
        printf "%-15s %s\n" "Memory Usage:" "$(get_memory_usage)"
        printf "%-15s %s\n" "Disk Usage:" "$(get_disk_usage)"
        printf "%-15s %s\n" "Uptime:" "$(get_uptime)"
        printf "%-15s %s\n" "Load Average:" "$(get_load_average)"
        echo ""
        
        # Docker Services Status
        echo -e "${PURPLE}🐳 Docker Services${NC}"
        echo "────────────────────────────────────────────────────────────"
        printf "%-25s %-15s %-10s %s\n" "Service" "Status" "Health" "Resources"
        echo "────────────────────────────────────────────────────────────"
        
        services=(
            "intelliflow-web-ui-prod:3000:health"
            "intelliflow-user-service-prod:8081:actuator/health"
            "intelliflow-transaction-service-prod:8082:actuator/health"
            "intelliflow-fraud-detection-prod:8083:actuator/health"
            "intelliflow-analytics-service-prod:8084:actuator/health"
            "intelliflow-notification-service-prod:8085:actuator/health"
            "intelliflow-postgres-prod:5432:"
            "intelliflow-redis-prod:6379:"
            "intelliflow-mongodb-prod:27017:"
            "intelliflow-kafka-prod:9092:"
            "intelliflow-grafana-prod:3000:"
            "intelliflow-nginx-prod:80:health"
        )
        
        for service_info in "${services[@]}"; do
            IFS=':' read -r container_name port endpoint <<< "$service_info"
            display_name=$(echo "$container_name" | sed 's/intelliflow-//g' | sed 's/-prod//g')
            
            status=$(get_container_status "$container_name")
            
            if [[ -n "$endpoint" ]]; then
                health=$(check_service_health "$display_name" "$port" "$endpoint")
            else
                health="-"
            fi
            
            resources=$(get_container_resources "$container_name")
            cpu=$(echo "$resources" | awk '{print $1}')
            memory=$(echo "$resources" | awk '{print $2}')
            
            printf "%-25s %-15s %-10s %s %s\n" "$display_name" "$status" "$health" "$cpu" "$memory"
        done
        
        echo ""
        
        # Application Metrics
        echo -e "${PURPLE}📊 Application Metrics${NC}"
        echo "────────────────────────────────────────────────────────────"
        
        # Get logs from the last minute
        recent_errors=$(ssh_exec "cd $DEPLOYMENT_PATH && docker-compose -f docker-compose.production.yml logs --since=1m 2>/dev/null | grep -i error | wc -l" || echo "0")
        recent_warnings=$(ssh_exec "cd $DEPLOYMENT_PATH && docker-compose -f docker-compose.production.yml logs --since=1m 2>/dev/null | grep -i warn | wc -l" || echo "0")
        
        printf "%-20s %s\n" "Recent Errors:" "$recent_errors"
        printf "%-20s %s\n" "Recent Warnings:" "$recent_warnings"
        
        # Check if services are responding
        web_ui_status=$(ssh_exec "curl -s -o /dev/null -w '%{http_code}' http://localhost:3000 2>/dev/null" || echo "000")
        api_status=$(ssh_exec "curl -s -o /dev/null -w '%{http_code}' http://localhost:8081/actuator/health 2>/dev/null" || echo "000")
        
        if [[ "$web_ui_status" == "200" ]]; then
            web_ui_display="${GREEN}✅ Online (200)${NC}"
        else
            web_ui_display="${RED}❌ Error ($web_ui_status)${NC}"
        fi
        
        if [[ "$api_status" == "200" ]]; then
            api_display="${GREEN}✅ Online (200)${NC}"
        else
            api_display="${RED}❌ Error ($api_status)${NC}"
        fi
        
        printf "%-20s %s\n" "Web UI:" "$web_ui_display"
        printf "%-20s %s\n" "API Gateway:" "$api_display"
        
        echo ""
        
        # Quick Actions
        echo -e "${PURPLE}🔧 Quick Actions${NC}"
        echo "────────────────────────────────────────────────────────────"
        echo "Press Ctrl+C to exit monitoring"
        echo "For detailed logs: ssh $SERVER_USER@$SERVER_HOST 'cd $DEPLOYMENT_PATH && docker-compose -f docker-compose.production.yml logs -f'"
        echo "To restart services: ssh $SERVER_USER@$SERVER_HOST 'cd $DEPLOYMENT_PATH && docker-compose -f docker-compose.production.yml restart'"
        
        # Wait for next refresh
        sleep "$REFRESH_INTERVAL"
    done
}

# Test connection first
echo -e "${BLUE}📡 Testing connection to $SERVER_HOST...${NC}"
if ! ssh_exec "echo 'Connected'" >/dev/null 2>&1; then
    echo -e "${RED}❌ Cannot connect to server${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Connected successfully${NC}"
echo "Starting monitoring in 3 seconds..."
sleep 3

# Start monitoring
monitor_server
