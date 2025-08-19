# 🚀 IntelliFlow Platform Performance Testing

Comprehensive performance testing suite for the IntelliFlow AI Platform using Apache JMeter.

## 📋 Overview

This performance testing framework provides comprehensive load, stress, spike, volume, and endurance testing capabilities for all IntelliFlow Platform microservices including:

- **User Service** - Authentication, registration, profile management
- **Transaction Service** - Payment processing, transaction management
- **Fraud Detection Service** - ML-based fraud prediction
- **Analytics Service** - Business intelligence and reporting
- **API Gateway** - Traffic routing and rate limiting

## 🛠 Prerequisites

### Required Software

1. **Apache JMeter 5.5+**
   ```bash
   # Download from https://jmeter.apache.org/
   # Or install via package manager
   brew install jmeter  # macOS
   sudo apt install jmeter  # Ubuntu
   ```

2. **Java 8+**
   ```bash
   java -version
   # Ensure JMeter can find Java
   ```

3. **curl** (for connectivity tests)
   ```bash
   curl --version
   ```

### Environment Setup

1. **Set JMeter Home** (if not in PATH):
   ```bash
   export JMETER_HOME=/path/to/jmeter
   ```

2. **Configure API Access**:
   ```bash
   export INTELLIFLOW_ADMIN_USER=admin
   export INTELLIFLOW_ADMIN_PASS=admin
   export INTELLIFLOW_API_KEY=your-api-key  # Optional
   ```

## 🎯 Test Types

### Load Testing
Tests system performance under expected user load.
```bash
./scripts/run-performance-tests.sh --test-type load --users 100 --duration 300
```

### Stress Testing
Finds system breaking points under increased load.
```bash
./scripts/run-performance-tests.sh --test-type stress --users 500 --duration 300
```

### Spike Testing
Tests system behavior under sudden load spikes.
```bash
./scripts/run-performance-tests.sh --test-type spike --users 300 --duration 180
```

### Volume Testing
Tests system with large data volumes and user counts.
```bash
./scripts/run-performance-tests.sh --test-type volume --users 400 --duration 600
```

### Endurance Testing
Tests system stability over extended periods.
```bash
./scripts/run-performance-tests.sh --test-type endurance --users 100 --duration 3600
```

## 🎭 Test Scenarios

### All Services (Default)
Comprehensive testing of all microservices.
```bash
./scripts/run-performance-tests.sh --scenario all
```

### User Service Only
Focus on authentication and user management.
```bash
./scripts/run-performance-tests.sh --scenario user --users 200
```

### Transaction Service Only
Focus on payment processing and transaction handling.
```bash
./scripts/run-performance-tests.sh --scenario transaction --users 150
```

### Fraud Detection Only
Focus on ML model performance and prediction latency.
```bash
./scripts/run-performance-tests.sh --scenario fraud --users 100
```

### Analytics Service Only
Focus on reporting and business intelligence queries.
```bash
./scripts/run-performance-tests.sh --scenario analytics --users 50
```

## 🚀 Quick Start

### Basic Load Test
```bash
# Test staging environment with default settings
./scripts/run-performance-tests.sh

# Test with custom parameters
./scripts/run-performance-tests.sh \
    --base-url https://api-staging.intelliflow.com \
    --users 50 \
    --duration 300 \
    --ramp-up 60
```

### Production Testing
```bash
# Careful load test on production
./scripts/run-performance-tests.sh \
    --base-url https://api.intelliflow.com \
    --test-type load \
    --users 25 \
    --duration 180 \
    --environment production
```

### Fraud Detection Focus
```bash
# Focus on fraud detection performance
./scripts/run-performance-tests.sh \
    --scenario fraud \
    --users 50 \
    --duration 600 \
    --test-type endurance
```

## 📊 Test Results

### Result Files

After test execution, results are saved in timestamped directories:
```
performance-testing/results/YYYYMMDD-HHMMSS/
├── index.html              # Main results dashboard
├── reports/                 # Detailed HTML reports
│   ├── index.html          # JMeter generated report
│   ├── content/            # Charts and graphs
│   └── sbadmin2-1.0.7/     # Report styling
├── results.jtl             # Raw test results
├── aggregate-report.jtl    # Aggregated metrics
├── summary-report.jtl      # Summary statistics
├── test-plan.txt           # Test configuration
├── execution-summary.txt   # Execution details
├── analysis-summary.txt    # Quick analysis
└── jmeter.log             # JMeter execution log
```

### Key Metrics

The test reports include:

- **Response Time Percentiles** (P50, P90, P95, P99)
- **Throughput** (Requests per second)
- **Error Rate** (Percentage of failed requests)
- **Concurrent Users** (Active user simulation)
- **Resource Utilization** (if monitoring enabled)

### Performance Targets

| Service | P95 Response Time | Error Rate | Throughput |
|---------|------------------|------------|------------|
| User Service | < 500ms | < 0.1% | > 100 RPS |
| Transaction Service | < 1000ms | < 0.5% | > 50 TPS |
| Fraud Detection | < 100ms | < 0.1% | > 200 RPS |
| Analytics Service | < 2000ms | < 1% | > 20 RPS |
| API Gateway | < 50ms | < 0.01% | > 500 RPS |

## 🔧 Advanced Configuration

### Custom JMeter Properties

Create `jmeter.properties` file for advanced configuration:
```properties
# Thread pool settings
jmeterengine.threadpool.size=50

# HTTP settings
httpclient4.retrycount=1
httpclient4.request.timeout=30000

# Results file settings
jmeter.save.saveservice.output_format=csv
jmeter.save.saveservice.response_data=false
```

### Environment-Specific Configuration

#### Staging Environment
```bash
./scripts/run-performance-tests.sh \
    --base-url https://api-staging.intelliflow.com \
    --environment staging \
    --users 100 \
    --duration 300
```

#### Production Environment
```bash
./scripts/run-performance-tests.sh \
    --base-url https://api.intelliflow.com \
    --environment production \
    --users 50 \
    --duration 180 \
    --ramp-up 120  # Slower ramp-up for production
```

#### Local Development
```bash
./scripts/run-performance-tests.sh \
    --base-url http://localhost:8080 \
    --environment local \
    --users 10 \
    --duration 60
```

### Custom Test Data

Modify `jmeter/test-data/` files for custom test data:
- `users.csv` - User credentials
- `transactions.csv` - Transaction templates
- `merchants.csv` - Merchant data

## 🐛 Troubleshooting

### Common Issues

#### JMeter Not Found
```bash
# Set JMeter home explicitly
export JMETER_HOME=/path/to/jmeter
./scripts/run-performance-tests.sh --jmeter-home /path/to/jmeter
```

#### API Connection Failed
```bash
# Test connectivity
curl -v https://api-staging.intelliflow.com/health

# Check firewall/proxy settings
export https_proxy=http://proxy:8080
```

#### Out of Memory Errors
```bash
# Increase JMeter heap size
export HEAP="-Xms1g -Xmx4g"
export JVM_ARGS="-XX:MaxMetaspaceSize=256m"
```

#### SSL Certificate Issues
```bash
# Disable SSL verification (testing only)
export JVM_ARGS="-Dtrust_all_cert=true"
```

### Performance Tuning

#### For High Load Tests
```bash
# Optimize JMeter for high load
export HEAP="-Xms2g -Xmx8g"
export JVM_ARGS="-XX:+UseG1GC -XX:MaxGCPauseMillis=100"

# Increase system limits
ulimit -n 65536
```

#### For Long Duration Tests
```bash
# Prevent memory leaks in long tests
export JVM_ARGS="-XX:+UseG1GC -XX:G1HeapRegionSize=16m"

# Enable GC logging
export JVM_ARGS="$JVM_ARGS -Xloggc:gc.log -XX:+PrintGCDetails"
```

## 📈 Monitoring Integration

### Prometheus Metrics

During tests, monitor these key metrics:
```promql
# Response time P95
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Throughput
rate(http_requests_total[5m])
```

### Grafana Dashboard

Import the provided Grafana dashboard for real-time monitoring:
```bash
# Dashboard location
monitoring/grafana/dashboards/performance-testing-dashboard.json
```

### Alert Configuration

Set up alerts for performance degradation:
```yaml
# Alertmanager rule
- alert: PerformanceTestHighLatency
  expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: High latency detected during performance test
```

## 🔄 CI/CD Integration

### Jenkins Pipeline

```groovy
stage('Performance Testing') {
    steps {
        script {
            sh '''
                cd performance-testing
                ./scripts/run-performance-tests.sh \
                    --base-url ${API_URL} \
                    --users 100 \
                    --duration 300 \
                    --no-reports
            '''
            
            // Archive results
            archiveArtifacts artifacts: 'performance-testing/results/**/*'
            
            // Publish performance report
            performanceReport sourceDataFiles: 'performance-testing/results/**/results.jtl'
        }
    }
}
```

### GitHub Actions

```yaml
name: Performance Tests
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  performance-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup JMeter
        run: |
          wget https://archive.apache.org/dist/jmeter/binaries/apache-jmeter-5.6.2.tgz
          tar -xzf apache-jmeter-5.6.2.tgz
          export JMETER_HOME=$PWD/apache-jmeter-5.6.2
      
      - name: Run Performance Tests
        run: |
          cd performance-testing
          ./scripts/run-performance-tests.sh \
            --base-url ${{ secrets.API_URL }} \
            --users 50 \
            --duration 180
      
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: performance-results
          path: performance-testing/results/
```

## 📝 Test Plan Details

### User Workflow Simulation

1. **User Registration**
   - Create new user account
   - Validate email format
   - Password strength verification

2. **Authentication**
   - Login with credentials
   - JWT token extraction
   - Session management

3. **Transaction Processing**
   - Create transaction
   - Fraud detection check
   - Payment processing
   - Status verification

4. **Analytics Queries**
   - Transaction statistics
   - Fraud trend analysis
   - Performance metrics

### Load Patterns

#### Steady Load
```
Users: ████████████████████ (Constant)
Time:  0    60   120  180  240  300s
```

#### Spike Load
```
Users: ████████████████████
       ██              ██
       ██              ██
Time:  0   30  60  90  120  150s
```

#### Gradual Ramp
```
Users:               ████████
                ██████
           ██████
      ██████
 ██████
Time: 0   60  120  180  240  300s
```

## 📞 Support

### Getting Help

1. **Check Logs**: Review `jmeter.log` for detailed error information
2. **Validate Config**: Use `--dry-run` to test configuration
3. **API Status**: Verify target API health endpoints
4. **Resource Limits**: Check system resources and JMeter memory settings

### Reporting Issues

When reporting performance test issues, include:
- Test configuration (users, duration, scenario)
- Error messages from logs
- System specifications
- Target environment details
- JMeter version and Java version

### Performance Baseline

Maintain performance baselines for comparison:
```bash
# Save baseline results
cp performance-testing/results/latest/analysis-summary.txt baselines/v1.0.0-baseline.txt

# Compare with baseline
diff baselines/v1.0.0-baseline.txt performance-testing/results/latest/analysis-summary.txt
```

---

**🚀 Happy Performance Testing!**

For additional questions or advanced configurations, consult the [IntelliFlow Platform Documentation](../README.md) or contact the platform team.
