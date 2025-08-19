#!/bin/bash

# IntelliFlow Platform Performance Testing Script
# Comprehensive performance testing with multiple scenarios and environments

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
JMETER_DIR="${SCRIPT_DIR}/../jmeter"
RESULTS_DIR="${SCRIPT_DIR}/../results/$(date +%Y%m%d-%H%M%S)"
REPORTS_DIR="${RESULTS_DIR}/reports"

# Default configuration
DEFAULT_BASE_URL="https://api-staging.intelliflow.com"
DEFAULT_USERS="100"
DEFAULT_RAMP_UP="60"
DEFAULT_DURATION="300"
DEFAULT_ENVIRONMENT="staging"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

IntelliFlow Platform Performance Testing Script

OPTIONS:
    -u, --base-url URL          Base URL for the API (default: ${DEFAULT_BASE_URL})
    -n, --users NUMBER          Number of concurrent users (default: ${DEFAULT_USERS})
    -r, --ramp-up SECONDS       Ramp-up time in seconds (default: ${DEFAULT_RAMP_UP})
    -d, --duration SECONDS      Test duration in seconds (default: ${DEFAULT_DURATION})
    -e, --environment ENV       Environment name (default: ${DEFAULT_ENVIRONMENT})
    -t, --test-type TYPE        Test type: load|stress|spike|volume|endurance (default: load)
    -s, --scenario SCENARIO     Test scenario: all|user|transaction|fraud|analytics (default: all)
    -j, --jmeter-home PATH      JMeter home directory (auto-detected if not provided)
    -o, --output-dir PATH       Output directory for results (default: auto-generated)
    --admin-username USERNAME   Admin username (default: admin)
    --admin-password PASSWORD   Admin password (default: admin)
    --no-reports               Skip HTML report generation
    --dry-run                  Show what would be executed without running tests
    -h, --help                 Show this help message

TEST TYPES:
    load        Normal load testing with expected user volume
    stress      Stress testing with increased load to find breaking points
    spike       Spike testing with sudden load increases
    volume      Volume testing with large datasets
    endurance   Endurance testing for extended periods

SCENARIOS:
    all         Run all service tests (default)
    user        User service only
    transaction Transaction service only
    fraud       Fraud detection service only
    analytics   Analytics service only

EXAMPLES:
    # Basic load test
    $0 --users 50 --duration 300

    # Stress test on production
    $0 --base-url https://api.intelliflow.com --test-type stress --users 500

    # Fraud detection focus test
    $0 --scenario fraud --users 25 --duration 600

    # Extended endurance test
    $0 --test-type endurance --duration 3600 --users 100

ENVIRONMENT VARIABLES:
    JMETER_HOME                JMeter installation directory
    INTELLIFLOW_API_KEY       API key for authentication
    INTELLIFLOW_ADMIN_USER    Admin username
    INTELLIFLOW_ADMIN_PASS    Admin password

EOF
}

# Parse command line arguments
parse_args() {
    BASE_URL="${DEFAULT_BASE_URL}"
    USERS="${DEFAULT_USERS}"
    RAMP_UP="${DEFAULT_RAMP_UP}"
    DURATION="${DEFAULT_DURATION}"
    ENVIRONMENT="${DEFAULT_ENVIRONMENT}"
    TEST_TYPE="load"
    SCENARIO="all"
    JMETER_HOME=""
    OUTPUT_DIR=""
    ADMIN_USERNAME="${INTELLIFLOW_ADMIN_USER:-admin}"
    ADMIN_PASSWORD="${INTELLIFLOW_ADMIN_PASS:-admin}"
    GENERATE_REPORTS=true
    DRY_RUN=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -u|--base-url)
                BASE_URL="$2"
                shift 2
                ;;
            -n|--users)
                USERS="$2"
                shift 2
                ;;
            -r|--ramp-up)
                RAMP_UP="$2"
                shift 2
                ;;
            -d|--duration)
                DURATION="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--test-type)
                TEST_TYPE="$2"
                shift 2
                ;;
            -s|--scenario)
                SCENARIO="$2"
                shift 2
                ;;
            -j|--jmeter-home)
                JMETER_HOME="$2"
                shift 2
                ;;
            -o|--output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --admin-username)
                ADMIN_USERNAME="$2"
                shift 2
                ;;
            --admin-password)
                ADMIN_PASSWORD="$2"
                shift 2
                ;;
            --no-reports)
                GENERATE_REPORTS=false
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Override results directory if specified
    if [[ -n "${OUTPUT_DIR}" ]]; then
        RESULTS_DIR="${OUTPUT_DIR}"
        REPORTS_DIR="${RESULTS_DIR}/reports"
    fi
}

# Detect JMeter installation
detect_jmeter() {
    if [[ -n "${JMETER_HOME}" ]] && [[ -x "${JMETER_HOME}/bin/jmeter" ]]; then
        JMETER_CMD="${JMETER_HOME}/bin/jmeter"
        return 0
    fi

    # Try environment variable
    if [[ -n "${JMETER_HOME:-}" ]] && [[ -x "${JMETER_HOME}/bin/jmeter" ]]; then
        JMETER_CMD="${JMETER_HOME}/bin/jmeter"
        return 0
    fi

    # Try common installation paths
    local common_paths=(
        "/opt/jmeter/bin/jmeter"
        "/usr/local/jmeter/bin/jmeter"
        "/usr/share/jmeter/bin/jmeter"
        "$(which jmeter 2>/dev/null || true)"
    )

    for path in "${common_paths[@]}"; do
        if [[ -x "${path}" ]]; then
            JMETER_CMD="${path}"
            JMETER_HOME="$(dirname "$(dirname "${path}")")"
            return 0
        fi
    done

    log_error "JMeter not found. Please install JMeter or specify --jmeter-home"
    exit 1
}

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."

    # Check JMeter
    detect_jmeter
    log_info "Found JMeter at: ${JMETER_CMD}"

    # Check JMeter version
    local jmeter_version
    jmeter_version=$("${JMETER_CMD}" --version 2>&1 | head -n1 | awk '{print $3}')
    log_info "JMeter version: ${jmeter_version}"

    # Check if test plan exists
    local test_plan="${JMETER_DIR}/intelliflow-platform-test-plan.jmx"
    if [[ ! -f "${test_plan}" ]]; then
        log_error "Test plan not found: ${test_plan}"
        exit 1
    fi

    # Test API connectivity
    log_info "Testing API connectivity to ${BASE_URL}..."
    if ! curl -s --max-time 10 "${BASE_URL}/health" > /dev/null; then
        log_warning "Could not connect to ${BASE_URL}/health. Tests may fail."
    else
        log_success "API connectivity test passed"
    fi

    # Create results directory
    mkdir -p "${RESULTS_DIR}"
    mkdir -p "${REPORTS_DIR}"
    log_info "Results will be saved to: ${RESULTS_DIR}"
}

# Adjust test parameters based on test type
adjust_test_parameters() {
    case "${TEST_TYPE}" in
        load)
            # Normal load testing parameters (already set)
            ;;
        stress)
            # Increase load for stress testing
            USERS=$((USERS * 2))
            RAMP_UP=$((RAMP_UP / 2))
            log_info "Stress test: Users increased to ${USERS}, ramp-up reduced to ${RAMP_UP}s"
            ;;
        spike)
            # Very fast ramp-up for spike testing
            RAMP_UP=10
            USERS=$((USERS * 3))
            DURATION=$((DURATION / 2))
            log_info "Spike test: Users increased to ${USERS}, ramp-up set to ${RAMP_UP}s"
            ;;
        volume)
            # More users, longer duration
            USERS=$((USERS * 4))
            DURATION=$((DURATION * 2))
            log_info "Volume test: Users increased to ${USERS}, duration extended to ${DURATION}s"
            ;;
        endurance)
            # Moderate load but extended duration
            DURATION=$((DURATION * 4))
            log_info "Endurance test: Duration extended to ${DURATION}s"
            ;;
        *)
            log_error "Unknown test type: ${TEST_TYPE}"
            exit 1
            ;;
    esac
}

# Set scenario-specific parameters
set_scenario_parameters() {
    case "${SCENARIO}" in
        all)
            SCENARIO_PARAMS=""
            ;;
        user)
            SCENARIO_PARAMS="-Jtransaction.users=0 -Jfraud.users=0 -Janalytics.users=0"
            ;;
        transaction)
            SCENARIO_PARAMS="-Jusers=0 -Jfraud.users=0 -Janalytics.users=0 -Jtransaction.users=${USERS}"
            ;;
        fraud)
            SCENARIO_PARAMS="-Jusers=0 -Jtransaction.users=0 -Janalytics.users=0 -Jfraud.users=${USERS}"
            ;;
        analytics)
            SCENARIO_PARAMS="-Jusers=0 -Jtransaction.users=0 -Jfraud.users=0 -Janalytics.users=${USERS}"
            ;;
        *)
            log_error "Unknown scenario: ${SCENARIO}"
            exit 1
            ;;
    esac
}

# Generate test execution plan
generate_execution_plan() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    cat > "${RESULTS_DIR}/test-plan.txt" << EOF
IntelliFlow Platform Performance Test Plan
==========================================

Test Configuration:
- Timestamp: ${timestamp}
- Test Type: ${TEST_TYPE}
- Scenario: ${SCENARIO}
- Environment: ${ENVIRONMENT}
- Base URL: ${BASE_URL}
- Concurrent Users: ${USERS}
- Ramp-up Time: ${RAMP_UP} seconds
- Test Duration: ${DURATION} seconds
- JMeter Version: $(${JMETER_CMD} --version 2>&1 | head -n1)

Test Objectives:
EOF

    case "${TEST_TYPE}" in
        load)
            echo "- Validate system performance under expected load" >> "${RESULTS_DIR}/test-plan.txt"
            echo "- Measure response times and throughput" >> "${RESULTS_DIR}/test-plan.txt"
            echo "- Verify system stability under normal conditions" >> "${RESULTS_DIR}/test-plan.txt"
            ;;
        stress)
            echo "- Find system breaking point under increased load" >> "${RESULTS_DIR}/test-plan.txt"
            echo "- Identify performance degradation patterns" >> "${RESULTS_DIR}/test-plan.txt"
            echo "- Test system recovery after stress" >> "${RESULTS_DIR}/test-plan.txt"
            ;;
        spike)
            echo "- Test system behavior under sudden load spikes" >> "${RESULTS_DIR}/test-plan.txt"
            echo "- Verify auto-scaling effectiveness" >> "${RESULTS_DIR}/test-plan.txt"
            echo "- Measure impact on existing users" >> "${RESULTS_DIR}/test-plan.txt"
            ;;
        volume)
            echo "- Test system with large data volumes" >> "${RESULTS_DIR}/test-plan.txt"
            echo "- Verify database performance under load" >> "${RESULTS_DIR}/test-plan.txt"
            echo "- Test memory and storage usage" >> "${RESULTS_DIR}/test-plan.txt"
            ;;
        endurance)
            echo "- Test system stability over extended periods" >> "${RESULTS_DIR}/test-plan.txt"
            echo "- Identify memory leaks and resource issues" >> "${RESULTS_DIR}/test-plan.txt"
            echo "- Verify consistent performance over time" >> "${RESULTS_DIR}/test-plan.txt"
            ;;
    esac

    echo "" >> "${RESULTS_DIR}/test-plan.txt"
    echo "Expected Outcomes:" >> "${RESULTS_DIR}/test-plan.txt"
    echo "- Response time P95 < 2000ms" >> "${RESULTS_DIR}/test-plan.txt"
    echo "- Error rate < 1%" >> "${RESULTS_DIR}/test-plan.txt"
    echo "- Fraud detection latency < 100ms" >> "${RESULTS_DIR}/test-plan.txt"
    echo "- System availability > 99.9%" >> "${RESULTS_DIR}/test-plan.txt"
}

# Run JMeter performance test
run_jmeter_test() {
    local test_plan="${JMETER_DIR}/intelliflow-platform-test-plan.jmx"
    local results_file="${RESULTS_DIR}/results.jtl"
    local log_file="${RESULTS_DIR}/jmeter.log"

    log_info "Starting performance test..."
    log_info "Test configuration: ${USERS} users, ${RAMP_UP}s ramp-up, ${DURATION}s duration"

    # Build JMeter command
    local jmeter_cmd=(
        "${JMETER_CMD}"
        -n  # Non-GUI mode
        -t "${test_plan}"
        -l "${results_file}"
        -j "${log_file}"
        -Jbase.url="${BASE_URL}"
        -Jusers="${USERS}"
        -Jramp.up="${RAMP_UP}"
        -Jduration="${DURATION}"
        -Jadmin.username="${ADMIN_USERNAME}"
        -Jadmin.password="${ADMIN_PASSWORD}"
        -Jresults.file="${results_file}"
        -Jaggregate.file="${RESULTS_DIR}/aggregate-report.jtl"
        -Jsummary.file="${RESULTS_DIR}/summary-report.jtl"
    )

    # Add scenario-specific parameters
    if [[ -n "${SCENARIO_PARAMS}" ]]; then
        read -ra params <<< "${SCENARIO_PARAMS}"
        for param in "${params[@]}"; do
            jmeter_cmd+=("${param}")
        done
    fi

    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN - Would execute:"
        printf '%s ' "${jmeter_cmd[@]}"
        echo
        return 0
    fi

    # Execute JMeter test
    local start_time=$(date +%s)
    
    if "${jmeter_cmd[@]}"; then
        local end_time=$(date +%s)
        local test_duration=$((end_time - start_time))
        
        log_success "Performance test completed successfully"
        log_info "Actual test duration: ${test_duration} seconds"
        
        # Save execution summary
        echo "Test execution completed at $(date)" >> "${RESULTS_DIR}/execution-summary.txt"
        echo "Actual duration: ${test_duration} seconds" >> "${RESULTS_DIR}/execution-summary.txt"
        echo "JMeter exit code: 0" >> "${RESULTS_DIR}/execution-summary.txt"
        
        return 0
    else
        local exit_code=$?
        log_error "Performance test failed with exit code: ${exit_code}"
        
        # Save failure information
        echo "Test execution failed at $(date)" >> "${RESULTS_DIR}/execution-summary.txt"
        echo "JMeter exit code: ${exit_code}" >> "${RESULTS_DIR}/execution-summary.txt"
        
        # Show last few lines of JMeter log
        if [[ -f "${log_file}" ]]; then
            log_error "Last 10 lines of JMeter log:"
            tail -10 "${log_file}"
        fi
        
        return ${exit_code}
    fi
}

# Generate HTML reports
generate_reports() {
    if [[ "${GENERATE_REPORTS}" != true ]]; then
        log_info "Skipping HTML report generation"
        return 0
    fi

    local results_file="${RESULTS_DIR}/results.jtl"
    
    if [[ ! -f "${results_file}" ]] || [[ ! -s "${results_file}" ]]; then
        log_warning "No results file found or file is empty. Skipping report generation."
        return 0
    fi

    log_info "Generating HTML performance report..."

    local report_cmd=(
        "${JMETER_CMD}"
        -g "${results_file}"
        -o "${REPORTS_DIR}"
        -Jjmeter.reportgenerator.overall_granularity=1000
        -Jjmeter.reportgenerator.graph.responseTimeDistribution.property.set_granularity=100
    )

    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN - Would generate reports with:"
        printf '%s ' "${report_cmd[@]}"
        echo
        return 0
    fi

    if "${report_cmd[@]}"; then
        log_success "HTML report generated: ${REPORTS_DIR}/index.html"
        
        # Create a simple index page with links to all reports
        cat > "${RESULTS_DIR}/index.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>IntelliFlow Performance Test Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background-color: #f4f4f4; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; }
        .link { display: inline-block; margin: 10px; padding: 10px 20px; background-color: #007cba; color: white; text-decoration: none; border-radius: 5px; }
        .link:hover { background-color: #005a87; }
        .summary { background-color: #e8f4f8; padding: 15px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>IntelliFlow Platform Performance Test Results</h1>
        <p><strong>Test Type:</strong> ${TEST_TYPE} | <strong>Scenario:</strong> ${SCENARIO} | <strong>Environment:</strong> ${ENVIRONMENT}</p>
        <p><strong>Date:</strong> $(date) | <strong>Users:</strong> ${USERS} | <strong>Duration:</strong> ${DURATION}s</p>
    </div>
    
    <div class="section">
        <h2>Reports</h2>
        <a href="reports/index.html" class="link">📊 Detailed HTML Report</a>
        <a href="test-plan.txt" class="link">📋 Test Plan</a>
        <a href="execution-summary.txt" class="link">📄 Execution Summary</a>
        <a href="jmeter.log" class="link">📜 JMeter Log</a>
    </div>
    
    <div class="section summary">
        <h2>Quick Summary</h2>
        <p>For detailed metrics, charts, and analysis, click on the "Detailed HTML Report" link above.</p>
        <p>All test artifacts and logs are available in this directory.</p>
    </div>
</body>
</html>
EOF
        
        log_success "Main index page created: ${RESULTS_DIR}/index.html"
    else
        log_error "Failed to generate HTML report"
        return 1
    fi
}

# Analyze results and generate summary
analyze_results() {
    local results_file="${RESULTS_DIR}/results.jtl"
    
    if [[ ! -f "${results_file}" ]] || [[ ! -s "${results_file}" ]]; then
        log_warning "No results file found for analysis"
        return 0
    fi

    log_info "Analyzing test results..."

    # Generate basic statistics using awk
    awk -F, '
    BEGIN {
        total_requests = 0
        total_errors = 0
        sum_response_time = 0
        min_response_time = 999999
        max_response_time = 0
    }
    NR > 1 {  # Skip header
        total_requests++
        response_time = $2
        success = $8
        
        if (success == "false") {
            total_errors++
        }
        
        sum_response_time += response_time
        
        if (response_time < min_response_time) {
            min_response_time = response_time
        }
        if (response_time > max_response_time) {
            max_response_time = response_time
        }
    }
    END {
        if (total_requests > 0) {
            avg_response_time = sum_response_time / total_requests
            error_rate = (total_errors / total_requests) * 100
            
            printf "Total Requests: %d\n", total_requests
            printf "Total Errors: %d\n", total_errors
            printf "Error Rate: %.2f%%\n", error_rate
            printf "Average Response Time: %.2f ms\n", avg_response_time
            printf "Min Response Time: %.2f ms\n", min_response_time
            printf "Max Response Time: %.2f ms\n", max_response_time
        }
    }' "${results_file}" > "${RESULTS_DIR}/analysis-summary.txt"

    # Display summary
    log_info "Test Results Summary:"
    cat "${RESULTS_DIR}/analysis-summary.txt"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Add any cleanup logic here
}

# Main execution function
main() {
    echo "==============================================="
    echo "IntelliFlow Platform Performance Testing"
    echo "==============================================="
    echo

    parse_args "$@"
    validate_prerequisites
    adjust_test_parameters
    set_scenario_parameters
    generate_execution_plan

    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN MODE - No tests will be executed"
        log_info "Configuration validated successfully"
        exit 0
    fi

    # Set up cleanup trap
    trap cleanup EXIT

    # Run the performance test
    if run_jmeter_test; then
        analyze_results
        generate_reports
        
        echo
        log_success "Performance testing completed successfully!"
        log_info "Results directory: ${RESULTS_DIR}"
        
        if [[ "${GENERATE_REPORTS}" == true ]]; then
            log_info "Open ${RESULTS_DIR}/index.html in your browser to view results"
        fi
        
        exit 0
    else
        log_error "Performance testing failed!"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"
