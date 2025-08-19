#!/bin/bash

# 🖥️ IntelliFlow AI Platform - Server Setup Script
# Prepare a fresh Ubuntu/Debian server for IntelliFlow deployment

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
echo "🖥️  IntelliFlow AI Platform - Server Setup"
echo "=========================================="
echo -e "${NC}"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}❌ This script must be run as root (use sudo)${NC}"
   exit 1
fi

# Detect OS
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
else
    echo -e "${RED}❌ Cannot detect OS version${NC}"
    exit 1
fi

echo -e "${BLUE}🔍 Detected OS: $OS $VER${NC}"

# Update system
echo -e "${BLUE}📦 Updating system packages...${NC}"
apt-get update && apt-get upgrade -y

# Install essential packages
echo -e "${BLUE}📦 Installing essential packages...${NC}"
apt-get install -y \
    curl \
    wget \
    git \
    vim \
    htop \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    ufw \
    fail2ban \
    certbot \
    python3-certbot-nginx

# Install Docker
echo -e "${BLUE}🐳 Installing Docker...${NC}"
if ! command -v docker &> /dev/null; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    systemctl enable docker
    systemctl start docker
    echo -e "${GREEN}✅ Docker installed successfully${NC}"
else
    echo -e "${GREEN}✅ Docker already installed${NC}"
fi

# Install Docker Compose
echo -e "${BLUE}🔧 Installing Docker Compose...${NC}"
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    echo -e "${GREEN}✅ Docker Compose installed successfully${NC}"
else
    echo -e "${GREEN}✅ Docker Compose already installed${NC}"
fi

# Install Node.js (for web UI builds)
echo -e "${BLUE}📦 Installing Node.js...${NC}"
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
    apt-get install -y nodejs
    echo -e "${GREEN}✅ Node.js installed successfully${NC}"
else
    echo -e "${GREEN}✅ Node.js already installed${NC}"
fi

# Install Java (for microservices)
echo -e "${BLUE}☕ Installing Java 17...${NC}"
if ! command -v java &> /dev/null; then
    apt-get install -y openjdk-17-jdk
    echo -e "${GREEN}✅ Java 17 installed successfully${NC}"
else
    echo -e "${GREEN}✅ Java already installed${NC}"
fi

# Configure firewall
echo -e "${BLUE}🔥 Configuring firewall...${NC}"
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 3000/tcp  # Web UI
ufw allow 3001/tcp  # Grafana
ufw allow 9090/tcp  # Prometheus (optional, for monitoring)
ufw --force enable
echo -e "${GREEN}✅ Firewall configured${NC}"

# Configure fail2ban
echo -e "${BLUE}🛡️  Configuring fail2ban...${NC}"
systemctl enable fail2ban
systemctl start fail2ban

# Create fail2ban jail for SSH
cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600
EOF

systemctl restart fail2ban
echo -e "${GREEN}✅ Fail2ban configured${NC}"

# Create deployment directory
echo -e "${BLUE}📁 Creating deployment directory...${NC}"
mkdir -p /opt/intelliflow
mkdir -p /opt/intelliflow/logs
mkdir -p /opt/intelliflow/data
mkdir -p /opt/intelliflow/ssl
mkdir -p /opt/intelliflow/backups
chown -R $SUDO_USER:$SUDO_USER /opt/intelliflow 2>/dev/null || true

# Configure log rotation
echo -e "${BLUE}📋 Configuring log rotation...${NC}"
cat > /etc/logrotate.d/intelliflow << EOF
/opt/intelliflow/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 root root
    postrotate
        docker exec intelliflow-nginx-prod nginx -s reload 2>/dev/null || true
    endscript
}
EOF

# Install monitoring tools
echo -e "${BLUE}📊 Installing monitoring tools...${NC}"
apt-get install -y \
    htop \
    iotop \
    nethogs \
    ncdu \
    tree

# Configure swap (if not exists)
if [[ ! -f /swapfile ]]; then
    echo -e "${BLUE}💾 Creating swap file...${NC}"
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo -e "${GREEN}✅ Swap file created (2GB)${NC}"
fi

# Set up automatic security updates
echo -e "${BLUE}🔒 Configuring automatic security updates...${NC}"
apt-get install -y unattended-upgrades
echo 'Unattended-Upgrade::Automatic-Reboot "false";' >> /etc/apt/apt.conf.d/50unattended-upgrades
systemctl enable unattended-upgrades

# Create a systemd service for IntelliFlow
echo -e "${BLUE}⚙️  Creating IntelliFlow systemd service...${NC}"
cat > /etc/systemd/system/intelliflow.service << EOF
[Unit]
Description=IntelliFlow AI Platform
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/intelliflow
ExecStart=/usr/local/bin/docker-compose -f docker-compose.production.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.production.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable intelliflow

# Create backup script
echo -e "${BLUE}💾 Creating backup script...${NC}"
cat > /opt/intelliflow/backup.sh << 'EOF'
#!/bin/bash

# IntelliFlow Platform Backup Script
BACKUP_DIR="/opt/intelliflow/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="intelliflow_backup_$DATE.tar.gz"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Stop services
cd /opt/intelliflow
docker-compose -f docker-compose.production.yml down

# Create backup
tar -czf "$BACKUP_DIR/$BACKUP_FILE" \
    --exclude='backups' \
    --exclude='logs' \
    --exclude='*.log' \
    -C /opt/intelliflow .

# Start services
docker-compose -f docker-compose.production.yml up -d

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DIR" -name "intelliflow_backup_*.tar.gz" -mtime +7 -delete

echo "Backup created: $BACKUP_DIR/$BACKUP_FILE"
EOF

chmod +x /opt/intelliflow/backup.sh

# Create daily backup cron job
echo -e "${BLUE}⏰ Setting up daily backups...${NC}"
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/intelliflow/backup.sh") | crontab -

# Create monitoring script
echo -e "${BLUE}📊 Creating monitoring script...${NC}"
cat > /opt/intelliflow/monitor.sh << 'EOF'
#!/bin/bash

# IntelliFlow Platform Monitoring Script
echo "=== IntelliFlow Platform Status ==="
echo "Date: $(date)"
echo ""

echo "=== System Resources ==="
echo "Memory Usage:"
free -h
echo ""
echo "Disk Usage:"
df -h /
echo ""
echo "CPU Load:"
uptime
echo ""

echo "=== Docker Services ==="
cd /opt/intelliflow
docker-compose -f docker-compose.production.yml ps
echo ""

echo "=== Service Health Checks ==="
services=("web-ui:3000" "user-service:8081" "transaction-service:8082" "fraud-detection-service:8083")

for service in "${services[@]}"; do
    service_name=${service%:*}
    port=${service#*:}
    
    if curl -s -f "http://localhost:$port/health" >/dev/null 2>&1; then
        echo "✅ $service_name is healthy"
    else
        echo "❌ $service_name is not responding"
    fi
done
EOF

chmod +x /opt/intelliflow/monitor.sh

# Display summary
echo ""
echo -e "${GREEN}🎉 Server setup completed successfully!${NC}"
echo ""
echo -e "${BLUE}📋 Summary:${NC}"
echo "  ✅ System packages updated"
echo "  ✅ Docker and Docker Compose installed"
echo "  ✅ Node.js and Java 17 installed"
echo "  ✅ Firewall configured (ports 22, 80, 443, 3000, 3001, 9090)"
echo "  ✅ Fail2ban configured for SSH protection"
echo "  ✅ Deployment directory created: /opt/intelliflow"
echo "  ✅ Log rotation configured"
echo "  ✅ Swap file created (2GB)"
echo "  ✅ Automatic security updates enabled"
echo "  ✅ IntelliFlow systemd service created"
echo "  ✅ Daily backup script configured"
echo "  ✅ Monitoring script created"
echo ""

echo -e "${BLUE}🔧 Useful Commands:${NC}"
echo "  Check services:     systemctl status intelliflow"
echo "  Start platform:     systemctl start intelliflow"
echo "  Stop platform:      systemctl stop intelliflow"
echo "  Monitor platform:   /opt/intelliflow/monitor.sh"
echo "  Create backup:      /opt/intelliflow/backup.sh"
echo "  View logs:          journalctl -u intelliflow -f"
echo ""

echo -e "${BLUE}📊 System Information:${NC}"
echo "  OS: $OS $VER"
echo "  Docker: $(docker --version)"
echo "  Docker Compose: $(docker-compose --version)"
echo "  Node.js: $(node --version)"
echo "  Java: $(java --version | head -1)"
echo ""

echo -e "${YELLOW}⚠️  Next Steps:${NC}"
echo "1. Reboot the server to ensure all configurations are applied"
echo "2. Run the deployment script to install IntelliFlow"
echo "3. Configure SSL certificates for your domain"
echo "4. Test the platform and monitoring"
echo ""

echo -e "${GREEN}✨ Your server is ready for IntelliFlow AI Platform deployment!${NC}"
