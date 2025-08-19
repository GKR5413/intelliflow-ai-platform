pipeline {
    agent {
        kubernetes {
            yaml """
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: trivy
    image: aquasec/trivy:0.48.1
    command:
    - cat
    tty: true
  - name: owasp-zap
    image: owasp/zap2docker-stable:2.14.0
    command:
    - cat
    tty: true
    securityContext:
      runAsUser: 1000
  - name: nuclei
    image: projectdiscovery/nuclei:v3.1.0
    command:
    - cat
    tty: true
  - name: kubectl
    image: bitnami/kubectl:1.28.4
    command:
    - cat
    tty: true
  - name: kube-bench
    image: aquasec/kube-bench:v0.7.1
    command:
    - cat
    tty: true
    securityContext:
      privileged: true
  - name: kube-hunter
    image: aquasec/kube-hunter:0.6.8
    command:
    - cat
    tty: true
"""
        }
    }

    environment {
        DOCKER_REGISTRY = "intelliflow-registry.com"
        TARGET_NAMESPACE = "${params.TARGET_NAMESPACE ?: 'intelliflow-staging'}"
        GATEWAY_URL = "${params.GATEWAY_URL ?: 'https://api-staging.intelliflow.com'}"
        SECURITY_SCAN_TOKEN = credentials('security-scan-token')
        DEFECTDOJO_API_KEY = credentials('defectdojo-api-key')
        SLACK_WEBHOOK = credentials('slack-security-webhook')
    }

    parameters {
        choice(
            name: 'SCAN_TYPE',
            choices: ['full', 'quick', 'container-only', 'api-only', 'infrastructure-only'],
            description: 'Type of security scan to perform'
        )
        string(
            name: 'TARGET_NAMESPACE',
            defaultValue: 'intelliflow-staging',
            description: 'Kubernetes namespace to scan'
        )
        string(
            name: 'GATEWAY_URL',
            defaultValue: 'https://api-staging.intelliflow.com',
            description: 'API Gateway URL for dynamic testing'
        )
        booleanParam(
            name: 'PASSIVE_SCAN_ONLY',
            defaultValue: false,
            description: 'Perform only passive security scans (safer for production)'
        )
        choice(
            name: 'SEVERITY_THRESHOLD',
            choices: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
            description: 'Minimum severity level to report'
        )
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: '50', daysToKeepStr: '30'))
        timeout(time: 120, unit: 'MINUTES')
        timestamps()
        ansiColor('xterm')
    }

    stages {
        stage('Preparation') {
            steps {
                script {
                    echo """
====================================
Security Scan Configuration
====================================
Scan Type: ${params.SCAN_TYPE}
Target Namespace: ${env.TARGET_NAMESPACE}
Gateway URL: ${env.GATEWAY_URL}
Passive Only: ${params.PASSIVE_SCAN_ONLY}
Severity Threshold: ${params.SEVERITY_THRESHOLD}
====================================
"""
                    
                    sendSlackNotification(
                        "🔒 Starting security scan for IntelliFlow Platform - Type: ${params.SCAN_TYPE}",
                        "info"
                    )
                }
            }
        }

        stage('Container Security Scans') {
            when {
                anyOf {
                    params.SCAN_TYPE == 'full'
                    params.SCAN_TYPE == 'container-only'
                }
            }
            parallel {
                stage('Container Vulnerability Scan') {
                    steps {
                        container('trivy') {
                            script {
                                def services = [
                                    'user-service',
                                    'transaction-service', 
                                    'fraud-detection-service',
                                    'analytics-service',
                                    'notification-service',
                                    'api-gateway'
                                ]
                                
                                services.each { service ->
                                    sh """
                                        echo "Scanning ${service}..."
                                        trivy image --security-checks vuln \
                                            --severity ${params.SEVERITY_THRESHOLD},CRITICAL \
                                            --format sarif \
                                            --output ${service}-vulnerabilities.sarif \
                                            ${env.DOCKER_REGISTRY}/${service}:latest
                                        
                                        trivy image --security-checks vuln \
                                            --severity ${params.SEVERITY_THRESHOLD},CRITICAL \
                                            --format table \
                                            --output ${service}-vulnerabilities.txt \
                                            ${env.DOCKER_REGISTRY}/${service}:latest
                                    """
                                }
                            }
                        }
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: '*-vulnerabilities.*', allowEmptyArchive: true
                        }
                    }
                }

                stage('Container Configuration Scan') {
                    steps {
                        container('trivy') {
                            sh """
                                # Scan Kubernetes manifests
                                trivy config k8s/ \
                                    --severity ${params.SEVERITY_THRESHOLD},CRITICAL \
                                    --format sarif \
                                    --output k8s-misconfigurations.sarif
                                
                                trivy config k8s/ \
                                    --severity ${params.SEVERITY_THRESHOLD},CRITICAL \
                                    --format table \
                                    --output k8s-misconfigurations.txt
                                
                                # Scan Docker files
                                find . -name "Dockerfile*" -exec trivy config {} \
                                    --severity ${params.SEVERITY_THRESHOLD},CRITICAL \
                                    --format table \; > dockerfile-scan-results.txt
                            """
                        }
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: 'k8s-misconfigurations.*,dockerfile-scan-results.txt', allowEmptyArchive: true
                        }
                    }
                }

                stage('Secret Detection') {
                    steps {
                        container('trivy') {
                            sh """
                                # Scan for secrets in source code
                                trivy fs --security-checks secret \
                                    --format sarif \
                                    --output secrets-scan.sarif \
                                    .
                                
                                trivy fs --security-checks secret \
                                    --format table \
                                    --output secrets-scan.txt \
                                    .
                            """
                        }
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: 'secrets-scan.*', allowEmptyArchive: true
                        }
                    }
                }
            }
        }

        stage('Infrastructure Security Scans') {
            when {
                anyOf {
                    params.SCAN_TYPE == 'full'
                    params.SCAN_TYPE == 'infrastructure-only'
                }
            }
            parallel {
                stage('Kubernetes Cluster Security') {
                    steps {
                        container('kube-bench') {
                            sh """
                                # Run CIS Kubernetes Benchmark
                                kube-bench run \
                                    --targets master,node,etcd,policies \
                                    --json > kube-bench-results.json
                                
                                kube-bench run \
                                    --targets master,node,etcd,policies > kube-bench-results.txt
                            """
                        }
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: 'kube-bench-results.*', allowEmptyArchive: true
                        }
                    }
                }

                stage('Kubernetes Penetration Testing') {
                    when {
                        not { params.PASSIVE_SCAN_ONLY }
                    }
                    steps {
                        container('kube-hunter') {
                            sh """
                                # Run kube-hunter for Kubernetes penetration testing
                                kube-hunter --remote ${env.GATEWAY_URL} \
                                    --report json \
                                    --log INFO > kube-hunter-results.json
                                
                                kube-hunter --remote ${env.GATEWAY_URL} \
                                    --report yaml > kube-hunter-results.yaml
                            """
                        }
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: 'kube-hunter-results.*', allowEmptyArchive: true
                        }
                    }
                }

                stage('Network Security Scan') {
                    steps {
                        container('nuclei') {
                            sh """
                                # Update nuclei templates
                                nuclei -update-templates
                                
                                # Run network security scan
                                nuclei -u ${env.GATEWAY_URL} \
                                    -severity ${params.SEVERITY_THRESHOLD.toLowerCase()},critical \
                                    -json -o nuclei-network-scan.json
                                
                                nuclei -u ${env.GATEWAY_URL} \
                                    -severity ${params.SEVERITY_THRESHOLD.toLowerCase()},critical \
                                    -o nuclei-network-scan.txt
                            """
                        }
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: 'nuclei-network-scan.*', allowEmptyArchive: true
                        }
                    }
                }
            }
        }

        stage('Application Security Testing') {
            when {
                anyOf {
                    params.SCAN_TYPE == 'full'
                    params.SCAN_TYPE == 'api-only'
                }
            }
            parallel {
                stage('OWASP ZAP Baseline Scan') {
                    steps {
                        container('owasp-zap') {
                            sh """
                                # Create ZAP baseline scan
                                zap-baseline.py \
                                    -t ${env.GATEWAY_URL} \
                                    -J zap-baseline-report.json \
                                    -r zap-baseline-report.html \
                                    -x zap-baseline-report.xml \
                                    -I \
                                    -l WARN
                            """
                        }
                    }
                    post {
                        always {
                            publishHTML([
                                allowMissing: false,
                                alwaysLinkToLastBuild: true,
                                keepAll: true,
                                reportDir: '.',
                                reportFiles: 'zap-baseline-report.html',
                                reportName: 'OWASP ZAP Baseline Report'
                            ])
                            archiveArtifacts artifacts: 'zap-baseline-report.*', allowEmptyArchive: true
                        }
                    }
                }

                stage('OWASP ZAP Full Scan') {
                    when {
                        allOf {
                            not { params.PASSIVE_SCAN_ONLY }
                            anyOf {
                                params.SCAN_TYPE == 'full'
                                env.TARGET_NAMESPACE != 'intelliflow-prod'
                            }
                        }
                    }
                    steps {
                        container('owasp-zap') {
                            sh """
                                # Create ZAP full scan (active)
                                zap-full-scan.py \
                                    -t ${env.GATEWAY_URL} \
                                    -J zap-full-report.json \
                                    -r zap-full-report.html \
                                    -x zap-full-report.xml \
                                    -I \
                                    -l WARN
                            """
                        }
                    }
                    post {
                        always {
                            publishHTML([
                                allowMissing: false,
                                alwaysLinkToLastBuild: true,
                                keepAll: true,
                                reportDir: '.',
                                reportFiles: 'zap-full-report.html',
                                reportName: 'OWASP ZAP Full Scan Report'
                            ])
                            archiveArtifacts artifacts: 'zap-full-report.*', allowEmptyArchive: true
                        }
                    }
                }

                stage('API Security Testing') {
                    steps {
                        container('nuclei') {
                            sh """
                                # API-specific security tests
                                nuclei -u ${env.GATEWAY_URL}/api/v1 \
                                    -tags api,auth,jwt,injection \
                                    -severity ${params.SEVERITY_THRESHOLD.toLowerCase()},critical \
                                    -json -o api-security-scan.json
                                
                                nuclei -u ${env.GATEWAY_URL}/api/v1 \
                                    -tags api,auth,jwt,injection \
                                    -severity ${params.SEVERITY_THRESHOLD.toLowerCase()},critical \
                                    -o api-security-scan.txt
                            """
                        }
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: 'api-security-scan.*', allowEmptyArchive: true
                        }
                    }
                }
            }
        }

        stage('Runtime Security Analysis') {
            when {
                anyOf {
                    params.SCAN_TYPE == 'full'
                    params.SCAN_TYPE == 'infrastructure-only'
                }
            }
            steps {
                container('kubectl') {
                    sh """
                        # Check running container security
                        kubectl get pods -n ${env.TARGET_NAMESPACE} -o jsonpath='{range .items[*]}{.metadata.name}{"\\t"}{.spec.securityContext}{"\\n"}{end}' > pod-security-contexts.txt
                        
                        # Check service accounts and RBAC
                        kubectl get serviceaccounts -n ${env.TARGET_NAMESPACE} -o yaml > serviceaccounts.yaml
                        kubectl get roles,rolebindings,clusterroles,clusterrolebindings -n ${env.TARGET_NAMESPACE} -o yaml > rbac-config.yaml
                        
                        # Check network policies
                        kubectl get networkpolicies -n ${env.TARGET_NAMESPACE} -o yaml > network-policies.yaml
                        
                        # Check pod security policies/standards
                        kubectl get pods -n ${env.TARGET_NAMESPACE} -o yaml > running-pods.yaml
                    """
                }
                
                container('trivy') {
                    sh """
                        # Scan running cluster configuration
                        trivy k8s --severity ${params.SEVERITY_THRESHOLD},CRITICAL \
                            --format sarif \
                            --output k8s-runtime-scan.sarif \
                            cluster
                        
                        trivy k8s --severity ${params.SEVERITY_THRESHOLD},CRITICAL \
                            --format table \
                            --output k8s-runtime-scan.txt \
                            cluster
                    """
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'pod-security-contexts.txt,serviceaccounts.yaml,rbac-config.yaml,network-policies.yaml,running-pods.yaml,k8s-runtime-scan.*', allowEmptyArchive: true
                }
            }
        }

        stage('Security Report Generation') {
            steps {
                script {
                    sh """
                        # Create consolidated security report
                        echo "# IntelliFlow Platform Security Scan Report" > security-report.md
                        echo "Generated: \$(date)" >> security-report.md
                        echo "Scan Type: ${params.SCAN_TYPE}" >> security-report.md
                        echo "Target Namespace: ${env.TARGET_NAMESPACE}" >> security-report.md
                        echo "Severity Threshold: ${params.SEVERITY_THRESHOLD}" >> security-report.md
                        echo "" >> security-report.md
                        
                        # Count vulnerabilities by severity
                        echo "## Summary" >> security-report.md
                        if [ -f "*-vulnerabilities.txt" ]; then
                            echo "### Container Vulnerabilities" >> security-report.md
                            find . -name "*-vulnerabilities.txt" -exec echo "#### {}" \\; -exec head -20 {} \\; >> security-report.md
                        fi
                        
                        if [ -f "k8s-misconfigurations.txt" ]; then
                            echo "### Kubernetes Misconfigurations" >> security-report.md
                            head -50 k8s-misconfigurations.txt >> security-report.md
                        fi
                        
                        if [ -f "zap-baseline-report.html" ]; then
                            echo "### OWASP ZAP Findings" >> security-report.md
                            echo "See attached ZAP report for detailed findings" >> security-report.md
                        fi
                        
                        # Calculate security score
                        python3 -c "
import json, glob, sys
total_critical = 0
total_high = 0
total_medium = 0
total_low = 0

# Count SARIF findings
for sarif_file in glob.glob('*.sarif'):
    try:
        with open(sarif_file, 'r') as f:
            data = json.load(f)
            for run in data.get('runs', []):
                for result in run.get('results', []):
                    level = result.get('level', 'info')
                    if level == 'error':
                        total_critical += 1
                    elif level == 'warning':
                        total_high += 1
                    elif level == 'note':
                        total_medium += 1
                    else:
                        total_low += 1
    except:
        pass

# Calculate security score (0-100)
total_findings = total_critical + total_high + total_medium + total_low
if total_findings == 0:
    score = 100
else:
    # Weight critical/high more heavily
    weighted_score = (total_critical * 10 + total_high * 5 + total_medium * 2 + total_low * 1)
    score = max(0, 100 - (weighted_score / total_findings * 10))

print(f'Security Score: {score:.1f}/100')
print(f'Critical: {total_critical}')
print(f'High: {total_high}')
print(f'Medium: {total_medium}')
print(f'Low: {total_low}')

# Write to file for later use
with open('security-metrics.txt', 'w') as f:
    f.write(f'SECURITY_SCORE={score:.1f}\\n')
    f.write(f'CRITICAL_FINDINGS={total_critical}\\n')
    f.write(f'HIGH_FINDINGS={total_high}\\n')
    f.write(f'MEDIUM_FINDINGS={total_medium}\\n')
    f.write(f'LOW_FINDINGS={total_low}\\n')
" >> security-report.md
                    """
                    
                    // Read security metrics
                    def securityMetrics = readFile('security-metrics.txt')
                    def metrics = [:]
                    securityMetrics.split('\n').each { line ->
                        if (line.contains('=')) {
                            def parts = line.split('=')
                            metrics[parts[0]] = parts[1]
                        }
                    }
                    
                    env.SECURITY_SCORE = metrics.SECURITY_SCORE ?: '0'
                    env.CRITICAL_FINDINGS = metrics.CRITICAL_FINDINGS ?: '0'
                    env.HIGH_FINDINGS = metrics.HIGH_FINDINGS ?: '0'
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'security-report.md,security-metrics.txt', allowEmptyArchive: true
                    
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: '.',
                        reportFiles: 'security-report.md',
                        reportName: 'Security Scan Report'
                    ])
                }
            }
        }

        stage('DefectDojo Integration') {
            when {
                environment name: 'DEFECTDOJO_API_KEY', value: ''
            }
            steps {
                script {
                    // Upload findings to DefectDojo
                    def sarfFiles = sh(script: "find . -name '*.sarif' -type f", returnStdout: true).trim().split('\n')
                    
                    sarfFiles.each { file ->
                        if (file) {
                            sh """
                                curl -X POST "https://defectdojo.intelliflow.com/api/v2/import-scan/" \
                                    -H "Authorization: Token ${env.DEFECTDOJO_API_KEY}" \
                                    -F "scan_type=SARIF" \
                                    -F "file=@${file}" \
                                    -F "engagement=1" \
                                    -F "verified=true" \
                                    -F "active=true" \
                                    -F "scan_date=\$(date +%Y-%m-%d)"
                            """
                        }
                    }
                }
            }
        }

        stage('Security Gate Evaluation') {
            steps {
                script {
                    def securityScore = env.SECURITY_SCORE as Float
                    def criticalFindings = env.CRITICAL_FINDINGS as Integer
                    def highFindings = env.HIGH_FINDINGS as Integer
                    
                    echo """
Security Gate Evaluation:
- Security Score: ${securityScore}/100
- Critical Findings: ${criticalFindings}
- High Findings: ${highFindings}
"""
                    
                    // Define security gates
                    def criticalThreshold = 0
                    def highThreshold = 5
                    def scoreThreshold = 70.0
                    
                    if (criticalFindings > criticalThreshold) {
                        error "Security gate failed: ${criticalFindings} critical findings exceed threshold of ${criticalThreshold}"
                    }
                    
                    if (highFindings > highThreshold) {
                        error "Security gate failed: ${highFindings} high findings exceed threshold of ${highThreshold}"
                    }
                    
                    if (securityScore < scoreThreshold) {
                        error "Security gate failed: Security score ${securityScore} below threshold of ${scoreThreshold}"
                    }
                    
                    echo "✅ All security gates passed!"
                }
            }
        }
    }

    post {
        always {
            script {
                // Collect all SARIF files
                def sarfFiles = sh(script: "find . -name '*.sarif' -type f", returnStdout: true).trim()
                if (sarfFiles) {
                    archiveArtifacts artifacts: '*.sarif', allowEmptyArchive: true
                }
            }
        }
        
        success {
            script {
                sendSlackNotification(
                    "✅ Security scan completed successfully - Score: ${env.SECURITY_SCORE}/100, Critical: ${env.CRITICAL_FINDINGS}, High: ${env.HIGH_FINDINGS}",
                    "good"
                )
            }
        }
        
        failure {
            script {
                sendSlackNotification(
                    "❌ Security scan failed - Critical: ${env.CRITICAL_FINDINGS}, High: ${env.HIGH_FINDINGS}",
                    "danger"
                )
            }
        }
        
        unstable {
            script {
                sendSlackNotification(
                    "⚠️ Security scan completed with warnings - Score: ${env.SECURITY_SCORE}/100",
                    "warning"
                )
            }
        }
    }
}

def sendSlackNotification(message, color) {
    def payload = [
        channel: '#security-alerts',
        color: color,
        message: message,
        buildUrl: env.BUILD_URL,
        jobName: env.JOB_NAME,
        buildNumber: env.BUILD_NUMBER
    ]
    
    httpRequest(
        httpMode: 'POST',
        url: env.SLACK_WEBHOOK,
        contentType: 'APPLICATION_JSON',
        requestBody: groovy.json.JsonOutput.toJson(payload)
    )
}
