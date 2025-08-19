#!/bin/bash

# 🚀 IntelliFlow AI Platform - Server Deployment Script
# Deploy the complete platform to a production server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOYMENT_DIR="${SCRIPT_DIR}"

# Default values
SERVER_USER="${SERVER_USER:-root}"
SERVER_HOST="${SERVER_HOST}"
SERVER_PORT="${SERVER_PORT:-22}"
DEPLOYMENT_PATH="${DEPLOYMENT_PATH:-/opt/intelliflow}"
BACKUP_ENABLED="${BACKUP_ENABLED:-true}"

echo -e "${BLUE}"
echo "  ___       _       _ _ _ _____ _                "
echo " |_ _|_ __ | |_ ___| | (_)  ___| | _____      __"
echo "  | || '_ \| __/ _ \ | | | |_  | |/ _ \ \ /\ / /"
echo "  | || | | | ||  __/ | | |  _| | | (_) \ V  V / "
echo " |___|_| |_|\__\___|_|_|_|_|   |_|\___/ \_/\_/  "
echo ""
echo "        🚀 Server Deployment Script"
echo -e "${NC}"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --host HOST          Server hostname or IP address (required)"
    echo "  -u, --user USER          SSH username (default: root)"
    echo "  -p, --port PORT          SSH port (default: 22)"
    echo "  -d, --deployment-path    Deployment path on server (default: /opt/intelliflow)"
    echo "  --no-backup             Skip backup creation"
    echo "  --ssl                   Enable SSL configuration"
    echo "  --domain DOMAIN         Set domain name for SSL"
    echo "  --help                  Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  SERVER_HOST             Server hostname or IP"
    echo "  SERVER_USER             SSH username"
    echo "  SERVER_PORT             SSH port"
    echo "  DEPLOYMENT_PATH         Deployment path on server"
    echo ""
    echo "Examples:"
    echo "  $0 --host 192.168.1.100 --user ubuntu"
    echo "  $0 --host myserver.com --ssl --domain intelliflow.myserver.com"
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
        -d|--deployment-path)
            DEPLOYMENT_PATH="$2"
            shift 2
            ;;
        --no-backup)
            BACKUP_ENABLED="false"
            shift
            ;;
        --ssl)
            SSL_ENABLED="true"
            shift
            ;;
        --domain)
            DOMAIN="$2"
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
    echo "Use --host option or set SERVER_HOST environment variable"
    usage
    exit 1
fi

# Function to run SSH commands
ssh_exec() {
    ssh -p "$SERVER_PORT" "$SERVER_USER@$SERVER_HOST" "$1"
}

# Function to copy files to server
scp_copy() {
    scp -P "$SERVER_PORT" -r "$1" "$SERVER_USER@$SERVER_HOST:$2"
}

echo -e "${BLUE}🔧 Deployment Configuration:${NC}"
echo "  Server: $SERVER_USER@$SERVER_HOST:$SERVER_PORT"
echo "  Deployment Path: $DEPLOYMENT_PATH"
echo "  Backup Enabled: $BACKUP_ENABLED"
echo "  SSL Enabled: ${SSL_ENABLED:-false}"
if [[ -n "$DOMAIN" ]]; then
    echo "  Domain: $DOMAIN"
fi
echo ""

# Step 1: Check server connection
echo -e "${BLUE}📡 Testing server connection...${NC}"
if ! ssh -p "$SERVER_PORT" -o ConnectTimeout=10 "$SERVER_USER@$SERVER_HOST" "echo 'Connection successful'" >/dev/null 2>&1; then
    echo -e "${RED}❌ Cannot connect to server $SERVER_HOST${NC}"
    echo "Please check:"
    echo "  - Server is running and accessible"
    echo "  - SSH credentials are correct"
    echo "  - SSH key is properly configured"
    exit 1
fi
echo -e "${GREEN}✅ Server connection successful${NC}"

# Step 2: Check server requirements
echo -e "${BLUE}🔍 Checking server requirements...${NC}"
ssh_exec "command -v docker >/dev/null 2>&1" || {
    echo -e "${YELLOW}⚠️  Docker not found. Installing Docker...${NC}"
    ssh_exec "curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh"
    ssh_exec "systemctl enable docker && systemctl start docker"
    ssh_exec "usermod -aG docker $SERVER_USER" || true
}

ssh_exec "command -v docker-compose >/dev/null 2>&1" || {
    echo -e "${YELLOW}⚠️  Docker Compose not found. Installing...${NC}"
    ssh_exec "curl -L \"https://github.com/docker/compose/releases/latest/download/docker-compose-\$(uname -s)-\$(uname -m)\" -o /usr/local/bin/docker-compose"
    ssh_exec "chmod +x /usr/local/bin/docker-compose"
}

echo -e "${GREEN}✅ Server requirements met${NC}"

# Step 3: Create backup if enabled
if [[ "$BACKUP_ENABLED" == "true" ]]; then
    echo -e "${BLUE}💾 Creating backup...${NC}"
    BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
    ssh_exec "if [ -d '$DEPLOYMENT_PATH' ]; then tar -czf /tmp/intelliflow_backup_$BACKUP_DATE.tar.gz -C '$DEPLOYMENT_PATH' . && echo 'Backup created: /tmp/intelliflow_backup_$BACKUP_DATE.tar.gz'; else echo 'No existing deployment to backup'; fi"
fi

# Step 4: Prepare deployment directory
echo -e "${BLUE}📁 Preparing deployment directory...${NC}"
ssh_exec "mkdir -p $DEPLOYMENT_PATH"
ssh_exec "mkdir -p $DEPLOYMENT_PATH/logs"
ssh_exec "mkdir -p $DEPLOYMENT_PATH/data"
ssh_exec "mkdir -p $DEPLOYMENT_PATH/ssl"

# Step 5: Copy project files
echo -e "${BLUE}📤 Copying project files to server...${NC}"
echo "  Copying deployment configurations..."
scp_copy "$DEPLOYMENT_DIR/docker-compose.production.yml" "$DEPLOYMENT_PATH/"
scp_copy "$DEPLOYMENT_DIR/env.production.template" "$DEPLOYMENT_PATH/"
scp_copy "$DEPLOYMENT_DIR/nginx/" "$DEPLOYMENT_PATH/"

echo "  Copying application source code..."
scp_copy "$PROJECT_ROOT/user-service" "$DEPLOYMENT_PATH/"
scp_copy "$PROJECT_ROOT/transaction-service" "$DEPLOYMENT_PATH/"
scp_copy "$PROJECT_ROOT/fraud-detection-service" "$DEPLOYMENT_PATH/"
scp_copy "$PROJECT_ROOT/analytics-service" "$DEPLOYMENT_PATH/"
scp_copy "$PROJECT_ROOT/notification-service" "$DEPLOYMENT_PATH/"
scp_copy "$PROJECT_ROOT/web-ui" "$DEPLOYMENT_PATH/"
scp_copy "$PROJECT_ROOT/shared-lib" "$DEPLOYMENT_PATH/"

echo "  Copying infrastructure configurations..."
scp_copy "$PROJECT_ROOT/docker" "$DEPLOYMENT_PATH/"

# Step 6: Set up environment configuration
echo -e "${BLUE}⚙️  Setting up environment configuration...${NC}"
if ssh_exec "[ ! -f '$DEPLOYMENT_PATH/.env' ]"; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from template...${NC}"
    ssh_exec "cp '$DEPLOYMENT_PATH/env.production.template' '$DEPLOYMENT_PATH/.env'"
    echo -e "${YELLOW}📝 Please edit $DEPLOYMENT_PATH/.env with your production values${NC}"
    echo -e "${BLUE}Required configuration:${NC}"
    echo "  - Database passwords"
    echo "  - JWT secret"
    echo "  - SMTP settings"
    echo "  - Domain configuration"
fi

# Step 7: Configure SSL if enabled
if [[ "$SSL_ENABLED" == "true" ]]; then
    echo -e "${BLUE}🔒 Setting up SSL configuration...${NC}"
    if [[ -n "$DOMAIN" ]]; then
        echo "  Installing Certbot for Let's Encrypt..."
        ssh_exec "apt-get update && apt-get install -y certbot python3-certbot-nginx"
        ssh_exec "certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN"
    else
        echo -e "${YELLOW}⚠️  SSL enabled but no domain specified. Please configure SSL manually.${NC}"
    fi
fi

# Step 8: Build and start services
echo -e "${BLUE}🏗️  Building and starting services...${NC}"
ssh_exec "cd '$DEPLOYMENT_PATH' && docker-compose -f docker-compose.production.yml build --no-cache"
ssh_exec "cd '$DEPLOYMENT_PATH' && docker-compose -f docker-compose.production.yml up -d"

# Step 9: Wait for services to be ready
echo -e "${BLUE}⏳ Waiting for services to start...${NC}"
sleep 30

# Step 10: Health check
echo -e "${BLUE}🔍 Performing health checks...${NC}"
HEALTH_CHECK_FAILED=false

services=("web-ui:3000" "user-service:8081" "transaction-service:8082" "fraud-detection-service:8083" "analytics-service:8084" "notification-service:8085")

for service in "${services[@]}"; do
    service_name=${service%:*}
    port=${service#*:}
    
    if ssh_exec "docker ps | grep -q '$service_name.*Up'"; then
        echo -e "${GREEN}✅ $service_name is running${NC}"
    else
        echo -e "${RED}❌ $service_name is not running${NC}"
        HEALTH_CHECK_FAILED=true
    fi
done

# Step 11: Display access information
echo ""
echo -e "${PURPLE}🎉 Deployment completed!${NC}"
echo ""

if [[ "$HEALTH_CHECK_FAILED" == "false" ]]; then
    echo -e "${GREEN}✅ All services are healthy${NC}"
else
    echo -e "${YELLOW}⚠️  Some services may need attention. Check logs:${NC}"
    echo "  ssh $SERVER_USER@$SERVER_HOST 'cd $DEPLOYMENT_PATH && docker-compose -f docker-compose.production.yml logs'"
fi

echo ""
echo -e "${BLUE}🌐 Access URLs:${NC}"
if [[ "$SSL_ENABLED" == "true" && -n "$DOMAIN" ]]; then
    echo "  🎨 Web UI:     https://$DOMAIN"
    echo "  📊 Grafana:    https://$DOMAIN/grafana/"
else
    echo "  🎨 Web UI:     http://$SERVER_HOST:3000"
    echo "  📊 Grafana:    http://$SERVER_HOST:3001"
fi

echo ""
echo -e "${BLUE}🔧 Management Commands:${NC}"
echo "  View logs:     ssh $SERVER_USER@$SERVER_HOST 'cd $DEPLOYMENT_PATH && docker-compose -f docker-compose.production.yml logs -f'"
echo "  Stop services: ssh $SERVER_USER@$SERVER_HOST 'cd $DEPLOYMENT_PATH && docker-compose -f docker-compose.production.yml down'"
echo "  Start services:ssh $SERVER_USER@$SERVER_HOST 'cd $DEPLOYMENT_PATH && docker-compose -f docker-compose.production.yml up -d'"
echo "  Update:        ./deploy-to-server.sh --host $SERVER_HOST --user $SERVER_USER"

echo ""
echo -e "${PURPLE}📋 Next Steps:${NC}"
echo "1. Edit environment file: ssh $SERVER_USER@$SERVER_HOST 'nano $DEPLOYMENT_PATH/.env'"
echo "2. Configure firewall: Allow ports 80, 443, 3000-3001"
echo "3. Set up monitoring: Configure alerts and backups"
echo "4. Test the application: Open the Web UI and create test transactions"

if [[ -n "$DOMAIN" ]]; then
    echo "5. Update DNS: Point $DOMAIN to $SERVER_HOST"
fi

echo ""
echo -e "${GREEN}🚀 IntelliFlow AI Platform is now running on your server!${NC}"
