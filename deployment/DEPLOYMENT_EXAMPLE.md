# 🚀 IntelliFlow AI Platform - Deployment Example

## **Real-World Deployment Example**

Here's exactly how to deploy the IntelliFlow AI Platform to a production server.

---

## 🌐 **Example: Deploying to Ubuntu Server**

### **Prerequisites**
- Ubuntu 20.04+ server with public IP
- SSH key access to the server
- Domain name pointed to your server (optional)

---

## **Step 1: Prepare Your Server (One Command)**

```bash
# SSH to your server
ssh root@YOUR_SERVER_IP

# Run the automated setup
curl -fsSL https://raw.githubusercontent.com/your-repo/intelliflow/main/deployment/server-setup.sh -o setup.sh
chmod +x setup.sh
sudo ./setup.sh

# Reboot to apply all changes
sudo reboot
```

---

## **Step 2: Deploy from Your Local Machine**

```bash
# Navigate to your IntelliFlow project
cd "/Users/spider_myan/Documents/IntelliFlow AI Platform"

# Deploy to your server (replace YOUR_SERVER_IP)
./deployment/deploy-to-server.sh --host YOUR_SERVER_IP --user root

# If you have a domain and want SSL:
./deployment/deploy-to-server.sh --host YOUR_SERVER_IP --user root --ssl --domain yourdomain.com
```

---

## **Step 3: Configure Production Settings**

```bash
# SSH to your server
ssh root@YOUR_SERVER_IP

# Edit the environment file
nano /opt/intelliflow/.env

# Update these critical values:
DB_PASSWORD=your_super_secure_database_password_123!
REDIS_PASSWORD=your_secure_redis_password_456!
MONGO_PASSWORD=your_secure_mongo_password_789!
JWT_SECRET=your_extremely_long_jwt_secret_key_minimum_256_characters_for_security_012345
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_specific_password
SERVER_DOMAIN=yourdomain.com

# Save and restart services
cd /opt/intelliflow
docker-compose -f docker-compose.production.yml restart
```

---

## **Step 4: Access Your Platform**

### **Web Interface**
- **URL**: http://YOUR_SERVER_IP:3000 (or https://yourdomain.com if SSL)
- **Login**: admin / admin
- **First Action**: Change admin password immediately

### **Monitoring**
- **Grafana**: http://YOUR_SERVER_IP:3001/grafana/
- **Login**: admin / your_grafana_password

---

## **Step 5: Real-Time Monitoring**

```bash
# Monitor your server in real-time
./deployment/monitor-server.sh --host YOUR_SERVER_IP --user root

# Check individual service logs
ssh root@YOUR_SERVER_IP 'cd /opt/intelliflow && docker-compose -f docker-compose.production.yml logs -f user-service'
```

---

## 📱 **Example Usage After Deployment**

### **1. Test Transaction Creation**
1. Open http://YOUR_SERVER_IP:3000
2. Login with admin/admin
3. Click "New Transaction"
4. Enter:
   - Amount: $250.00
   - Type: Payment
   - Merchant: Amazon Store
   - Description: Test purchase
5. Watch real-time fraud detection!

### **2. Monitor with Grafana**
1. Open http://YOUR_SERVER_IP:3001
2. Login with your Grafana credentials
3. View platform metrics and performance

### **3. Check System Health**
```bash
# Health check all services
curl http://YOUR_SERVER_IP/health

# Check API status
curl http://YOUR_SERVER_IP:8081/actuator/health

# Web UI status
curl http://YOUR_SERVER_IP:3000/health
```

---

## 🛠️ **Common Management Tasks**

### **Restart Services**
```bash
ssh root@YOUR_SERVER_IP 'systemctl restart intelliflow'
```

### **View Logs**
```bash
ssh root@YOUR_SERVER_IP 'cd /opt/intelliflow && docker-compose -f docker-compose.production.yml logs --tail=100'
```

### **Create Backup**
```bash
ssh root@YOUR_SERVER_IP '/opt/intelliflow/backup.sh'
```

### **Update Platform**
```bash
# Re-run deployment script
./deployment/deploy-to-server.sh --host YOUR_SERVER_IP --user root
```

---

## 🔒 **Security Best Practices**

### **Immediate Actions After Deployment**
1. **Change Default Passwords**:
   ```bash
   # Login to Web UI and change admin password
   # Update .env file with strong passwords
   ```

2. **Configure Firewall**:
   ```bash
   ssh root@YOUR_SERVER_IP
   ufw deny 8081  # Block direct API access
   ufw deny 8082
   ufw deny 8083
   ufw deny 8084
   ufw deny 8085
   ```

3. **Set Up SSH Keys** (disable password auth):
   ```bash
   nano /etc/ssh/sshd_config
   # Set: PasswordAuthentication no
   systemctl restart ssh
   ```

---

## 📊 **Performance Optimization**

### **For High Traffic**
```bash
# Scale critical services
ssh root@YOUR_SERVER_IP 'cd /opt/intelliflow && docker-compose -f docker-compose.production.yml up -d --scale user-service=3 --scale transaction-service=2'
```

### **Database Optimization**
```bash
# Monitor database performance
ssh root@YOUR_SERVER_IP 'docker exec intelliflow-postgres-prod psql -U intelliflow_user -c "SELECT * FROM pg_stat_activity;"'
```

---

## 🚨 **Troubleshooting Quick Fixes**

### **Services Not Starting**
```bash
# Check Docker
ssh root@YOUR_SERVER_IP 'systemctl status docker'

# Check resources
ssh root@YOUR_SERVER_IP 'htop'
ssh root@YOUR_SERVER_IP 'df -h'

# Restart everything
ssh root@YOUR_SERVER_IP 'cd /opt/intelliflow && docker-compose -f docker-compose.production.yml down && docker-compose -f docker-compose.production.yml up -d'
```

### **Web UI Not Loading**
```bash
# Check nginx
ssh root@YOUR_SERVER_IP 'docker logs intelliflow-nginx-prod'

# Check web-ui container
ssh root@YOUR_SERVER_IP 'docker logs intelliflow-web-ui-prod'
```

### **API Errors**
```bash
# Check user service logs
ssh root@YOUR_SERVER_IP 'docker logs intelliflow-user-service-prod'

# Test database connection
ssh root@YOUR_SERVER_IP 'docker exec intelliflow-postgres-prod pg_isready -U intelliflow_user'
```

---

## 🎯 **Success Indicators**

After deployment, you should see:

✅ **All containers running**: `docker ps` shows all services UP  
✅ **Web UI accessible**: Login page loads at your server IP  
✅ **API responding**: Health checks return 200 OK  
✅ **Fraud detection working**: Transaction creation triggers ML analysis  
✅ **Monitoring active**: Grafana shows system metrics  
✅ **Logs flowing**: All services generating logs  

---

## 📞 **Need Help?**

### **Check Status**
```bash
# Quick status check
./deployment/monitor-server.sh --host YOUR_SERVER_IP

# Detailed health check
ssh root@YOUR_SERVER_IP '/opt/intelliflow/monitor.sh'
```

### **Common Commands Reference**
```bash
# Start platform
ssh root@YOUR_SERVER_IP 'systemctl start intelliflow'

# Stop platform  
ssh root@YOUR_SERVER_IP 'systemctl stop intelliflow'

# View all logs
ssh root@YOUR_SERVER_IP 'cd /opt/intelliflow && docker-compose -f docker-compose.production.yml logs'

# Check resource usage
ssh root@YOUR_SERVER_IP 'docker stats'
```

---

**🎉 Your IntelliFlow AI Platform is now running in production!**

**Access**: http://YOUR_SERVER_IP:3000  
**Login**: admin / admin  
**Start**: Create your first transaction and watch the AI at work! 🚀
