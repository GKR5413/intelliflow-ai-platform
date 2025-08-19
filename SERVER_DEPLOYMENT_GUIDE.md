# 🚀 IntelliFlow AI Platform - Server Deployment Guide

## 🌐 **Deploy to Production Server**

This guide provides everything you need to deploy the IntelliFlow AI Platform to a production server with enterprise-grade security, monitoring, and scalability.

---

## 📋 **Prerequisites**

### **Server Requirements**
- **OS**: Ubuntu 20.04+ / Debian 11+ / CentOS 8+
- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 8GB+ (16GB+ recommended)
- **Storage**: 50GB+ SSD (100GB+ recommended)
- **Network**: Public IP address with ports 80, 443, 22 open

### **Local Requirements**
- SSH access to the server
- SSH key authentication configured
- Docker and Docker Compose on local machine (for building)

---

## 🚀 **Quick Deployment (3 Commands)**

### **1. Prepare Your Server**
```bash
# Copy setup script to your server
scp deployment/server-setup.sh root@YOUR_SERVER_IP:/tmp/

# Run the setup script on your server
ssh root@YOUR_SERVER_IP "chmod +x /tmp/server-setup.sh && /tmp/server-setup.sh"
```

### **2. Deploy the Platform**
```bash
# Deploy from your local machine
cd "/Users/spider_myan/Documents/IntelliFlow AI Platform"
./deployment/deploy-to-server.sh --host YOUR_SERVER_IP --user root
```

### **3. Access Your Platform**
- **Web UI**: http://YOUR_SERVER_IP:3000
- **Login**: admin / admin
- **Grafana**: http://YOUR_SERVER_IP:3001

---

## 📖 **Detailed Deployment Steps**

### **Step 1: Server Preparation**

#### **Automated Setup (Recommended)**
```bash
# Download and run the automated setup script
curl -fsSL https://raw.githubusercontent.com/your-repo/intelliflow/main/deployment/server-setup.sh -o setup.sh
chmod +x setup.sh
sudo ./setup.sh
```

#### **Manual Setup**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo systemctl enable docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Configure firewall
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 3000
sudo ufw allow 3001
sudo ufw --force enable
```

### **Step 2: Environment Configuration**

#### **Create Production Environment File**
```bash
# On your server
cd /opt/intelliflow
cp env.production.template .env

# Edit with your production values
nano .env
```

#### **Required Environment Variables**
```env
# Database Security
DB_PASSWORD=your_very_secure_database_password_here
REDIS_PASSWORD=your_secure_redis_password_here
MONGO_PASSWORD=your_secure_mongo_password_here

# JWT Security
JWT_SECRET=your_extremely_long_jwt_secret_minimum_256_characters_here

# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_specific_password

# Domain Configuration
SERVER_DOMAIN=your-domain.com
```

### **Step 3: Deploy the Application**

#### **Using the Deployment Script**
```bash
# From your local machine
./deployment/deploy-to-server.sh \
  --host your-server-ip \
  --user root \
  --ssl \
  --domain your-domain.com
```

#### **Manual Deployment**
```bash
# Copy files to server
scp -r . root@your-server-ip:/opt/intelliflow/

# SSH to server and deploy
ssh root@your-server-ip
cd /opt/intelliflow
docker-compose -f deployment/docker-compose.production.yml up -d
```

### **Step 4: SSL Configuration (HTTPS)**

#### **Automatic SSL with Let's Encrypt**
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

#### **Manual SSL Configuration**
```bash
# Copy your SSL certificates to the server
scp your-cert.pem root@your-server-ip:/opt/intelliflow/ssl/cert.pem
scp your-key.pem root@your-server-ip:/opt/intelliflow/ssl/key.pem

# Update nginx configuration
# Edit: /opt/intelliflow/nginx/nginx.conf
# Uncomment the SSL server block
```

---

## 🔧 **Configuration Options**

### **Production Docker Compose**
The `docker-compose.production.yml` includes:
- **Resource Limits**: CPU and memory constraints
- **Health Checks**: Automatic container monitoring
- **Restart Policies**: Automatic recovery from failures
- **Security Context**: Non-root users and read-only filesystems
- **Logging**: Centralized log aggregation
- **Networking**: Isolated container networks

### **Nginx Reverse Proxy**
Features included:
- **SSL Termination**: HTTPS encryption
- **Rate Limiting**: API protection
- **Load Balancing**: Future scalability
- **Security Headers**: XSS, CSRF protection
- **Compression**: Gzip for faster responses
- **Caching**: Static asset optimization

### **Resource Requirements by Service**

| Service | CPU | Memory | Purpose |
|---------|-----|--------|---------|
| **Web UI** | 0.5 core | 512MB | React frontend |
| **User Service** | 1.0 core | 1GB | Authentication |
| **Transaction Service** | 1.0 core | 1.5GB | Payment processing |
| **Fraud Detection** | 2.0 cores | 2GB | ML analysis |
| **Analytics Service** | 1.0 core | 1.5GB | Reporting |
| **Notification Service** | 0.5 core | 1GB | Alerts |
| **PostgreSQL** | 1.0 core | 2GB | Primary database |
| **Redis** | 0.5 core | 1GB | Caching |
| **MongoDB** | 1.0 core | 2GB | Analytics data |
| **Kafka** | 1.0 core | 2GB | Message streaming |

**Total Recommended**: 8+ cores, 16GB+ RAM

---

## 📊 **Monitoring & Management**

### **Live Monitoring**
```bash
# Real-time server monitoring
./deployment/monitor-server.sh --host your-server-ip

# Docker container status
ssh root@your-server-ip 'docker ps'

# Service logs
ssh root@your-server-ip 'cd /opt/intelliflow && docker-compose -f docker-compose.production.yml logs -f'
```

### **Health Checks**
```bash
# Web UI health
curl http://your-server-ip:3000/health

# API health
curl http://your-server-ip:8081/actuator/health

# All services
curl http://your-server-ip/api/health
```

### **Management Commands**
```bash
# Start services
sudo systemctl start intelliflow

# Stop services
sudo systemctl stop intelliflow

# Restart specific service
ssh root@your-server-ip 'cd /opt/intelliflow && docker-compose -f docker-compose.production.yml restart user-service'

# View service status
sudo systemctl status intelliflow
```

---

## 🔒 **Security Configuration**

### **Firewall Rules**
```bash
# Essential ports only
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw allow 3000  # Web UI (can be removed if using SSL)
sudo ufw deny 8081   # Block direct API access
sudo ufw deny 8082
sudo ufw deny 8083
sudo ufw deny 8084
sudo ufw deny 8085
```

### **SSH Hardening**
```bash
# Disable password authentication
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
# Set: PermitRootLogin no

# Restart SSH
sudo systemctl restart ssh
```

### **Fail2Ban Protection**
```bash
# Configure fail2ban for SSH protection
sudo nano /etc/fail2ban/jail.local

[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
```

### **Database Security**
- Strong passwords (min 32 characters)
- Network isolation (containers only)
- Regular backups with encryption
- Connection limits and timeouts

---

## 💾 **Backup & Recovery**

### **Automated Backups**
```bash
# Daily backup (already configured)
/opt/intelliflow/backup.sh

# Manual backup
cd /opt/intelliflow
docker-compose -f docker-compose.production.yml down
tar -czf backup_$(date +%Y%m%d).tar.gz .
docker-compose -f docker-compose.production.yml up -d
```

### **Database Backups**
```bash
# PostgreSQL backup
docker exec intelliflow-postgres-prod pg_dumpall -U intelliflow_user > backup_db.sql

# MongoDB backup
docker exec intelliflow-mongodb-prod mongodump --out /backup/

# Redis backup
docker exec intelliflow-redis-prod redis-cli BGSAVE
```

### **Recovery Process**
```bash
# Stop services
cd /opt/intelliflow
docker-compose -f docker-compose.production.yml down

# Restore from backup
tar -xzf backup_date.tar.gz

# Start services
docker-compose -f docker-compose.production.yml up -d
```

---

## 🌍 **Domain & DNS Configuration**

### **DNS Records**
```dns
# A Record
your-domain.com    A    YOUR_SERVER_IP

# CNAME for www
www.your-domain.com    CNAME    your-domain.com

# Optional subdomains
api.your-domain.com    A    YOUR_SERVER_IP
grafana.your-domain.com    A    YOUR_SERVER_IP
```

### **Subdomain Configuration**
```nginx
# In nginx configuration
server {
    listen 443 ssl;
    server_name api.your-domain.com;
    
    location / {
        proxy_pass http://user-service:8081;
    }
}

server {
    listen 443 ssl;
    server_name grafana.your-domain.com;
    
    location / {
        proxy_pass http://grafana:3000;
    }
}
```

---

## 🔧 **Troubleshooting**

### **Common Issues**

#### **Services Not Starting**
```bash
# Check Docker status
sudo systemctl status docker

# Check logs
docker-compose -f docker-compose.production.yml logs service-name

# Check resource usage
htop
df -h
```

#### **Connection Issues**
```bash
# Check firewall
sudo ufw status

# Check port availability
netstat -tulpn | grep :3000

# Test internal connectivity
docker exec intelliflow-web-ui-prod curl http://user-service:8081/actuator/health
```

#### **SSL Certificate Issues**
```bash
# Check certificate
sudo certbot certificates

# Renew certificate
sudo certbot renew

# Test SSL
openssl s_client -connect your-domain.com:443
```

#### **Performance Issues**
```bash
# Check system resources
htop
iotop
nethogs

# Check database performance
docker exec intelliflow-postgres-prod psql -U intelliflow_user -c "SELECT * FROM pg_stat_activity;"

# Optimize containers
docker system prune -a
```

### **Log Locations**
```bash
# Application logs
/opt/intelliflow/logs/

# Nginx logs
/var/log/nginx/

# System logs
journalctl -u intelliflow

# Docker logs
docker-compose -f docker-compose.production.yml logs
```

---

## 📈 **Scaling & Performance**

### **Horizontal Scaling**
```yaml
# Scale specific services
docker-compose -f docker-compose.production.yml up -d --scale user-service=3

# Load balancer configuration
upstream user-service {
    server user-service-1:8081;
    server user-service-2:8081;
    server user-service-3:8081;
}
```

### **Database Optimization**
```bash
# PostgreSQL tuning
# Edit postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
```

### **Caching Strategy**
- Redis for session data
- Nginx for static assets
- Application-level caching
- CDN for global distribution

---

## 🎯 **Production Checklist**

### **Before Going Live**
- [ ] Strong passwords for all services
- [ ] SSL certificates configured
- [ ] Firewall properly configured
- [ ] Backup system tested
- [ ] Monitoring alerts configured
- [ ] Log rotation configured
- [ ] Performance testing completed
- [ ] Security scan performed
- [ ] DNS records configured
- [ ] Email notifications working

### **Post-Deployment**
- [ ] Monitor resource usage
- [ ] Set up automated backups
- [ ] Configure monitoring alerts
- [ ] Document access credentials
- [ ] Train team on management
- [ ] Plan disaster recovery
- [ ] Schedule regular maintenance
- [ ] Monitor security logs

---

## 🌟 **Success! Your Platform is Live**

After successful deployment, you'll have:

✅ **Production-Ready Platform** running on your server  
✅ **Beautiful Web UI** accessible via your domain  
✅ **Enterprise Security** with SSL and firewall protection  
✅ **Automated Backups** and monitoring  
✅ **Scalable Architecture** ready for growth  
✅ **Full ML Capabilities** for fraud detection  

### **🚀 Access Your Platform:**
- **Main Interface**: https://your-domain.com
- **Login**: admin / admin (change immediately)
- **Monitoring**: https://your-domain.com/grafana/

Your IntelliFlow AI Platform is now running in production with enterprise-grade security and monitoring! 🎉
