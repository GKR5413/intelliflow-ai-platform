-- IntelliFlow Analytics Service - Initial Database Migration
-- Version: V1
-- Description: Create initial analytics tables and indexes

-- Create schema for analytics
CREATE SCHEMA IF NOT EXISTS analytics;

-- Set search path
SET search_path TO analytics, public;

-- Transaction Summary Table
CREATE TABLE transaction_summary (
    id BIGSERIAL PRIMARY KEY,
    transaction_date DATE NOT NULL,
    user_id BIGINT NOT NULL,
    transaction_count INTEGER NOT NULL DEFAULT 0,
    total_amount DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
    avg_amount DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
    fraud_count INTEGER NOT NULL DEFAULT 0,
    fraud_rate DECIMAL(5, 4) NOT NULL DEFAULT 0.0000,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    country_code VARCHAR(2),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Daily Business Metrics Table
CREATE TABLE daily_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_date DATE NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20, 6) NOT NULL,
    metric_type VARCHAR(50) NOT NULL, -- 'revenue', 'transactions', 'users', 'fraud_rate', etc.
    dimensions JSONB, -- Additional dimensions as JSON
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- User Behavior Analytics Table
CREATE TABLE user_behavior_analytics (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    analysis_date DATE NOT NULL,
    total_transactions INTEGER NOT NULL DEFAULT 0,
    total_amount DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
    avg_transaction_amount DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
    max_transaction_amount DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
    min_transaction_amount DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
    unique_merchants INTEGER NOT NULL DEFAULT 0,
    unique_categories INTEGER NOT NULL DEFAULT 0,
    weekend_transactions INTEGER NOT NULL DEFAULT 0,
    night_transactions INTEGER NOT NULL DEFAULT 0,
    international_transactions INTEGER NOT NULL DEFAULT 0,
    risk_score DECIMAL(5, 4) NOT NULL DEFAULT 0.0000,
    behavior_flags JSONB, -- JSON array of behavior flags
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Merchant Analytics Table
CREATE TABLE merchant_analytics (
    id BIGSERIAL PRIMARY KEY,
    merchant_id VARCHAR(100) NOT NULL,
    merchant_name VARCHAR(255),
    merchant_category VARCHAR(100),
    analysis_date DATE NOT NULL,
    transaction_count INTEGER NOT NULL DEFAULT 0,
    total_volume DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    avg_transaction_amount DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
    unique_customers INTEGER NOT NULL DEFAULT 0,
    fraud_incidents INTEGER NOT NULL DEFAULT 0,
    fraud_rate DECIMAL(5, 4) NOT NULL DEFAULT 0.0000,
    chargeback_count INTEGER NOT NULL DEFAULT 0,
    chargeback_amount DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
    risk_level VARCHAR(20) NOT NULL DEFAULT 'low',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Fraud Detection Analytics Table
CREATE TABLE fraud_detection_analytics (
    id BIGSERIAL PRIMARY KEY,
    analysis_date DATE NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    total_predictions INTEGER NOT NULL DEFAULT 0,
    fraud_predictions INTEGER NOT NULL DEFAULT 0,
    true_positives INTEGER NOT NULL DEFAULT 0,
    false_positives INTEGER NOT NULL DEFAULT 0,
    true_negatives INTEGER NOT NULL DEFAULT 0,
    false_negatives INTEGER NOT NULL DEFAULT 0,
    precision_score DECIMAL(5, 4) NOT NULL DEFAULT 0.0000,
    recall_score DECIMAL(5, 4) NOT NULL DEFAULT 0.0000,
    f1_score DECIMAL(5, 4) NOT NULL DEFAULT 0.0000,
    auc_score DECIMAL(5, 4) NOT NULL DEFAULT 0.0000,
    avg_prediction_time_ms DECIMAL(8, 3) NOT NULL DEFAULT 0.000,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Real-time Alerts Table
CREATE TABLE real_time_alerts (
    id BIGSERIAL PRIMARY KEY,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
    title VARCHAR(255) NOT NULL,
    description TEXT,
    entity_type VARCHAR(50), -- 'user', 'merchant', 'transaction', 'system'
    entity_id VARCHAR(100),
    alert_data JSONB,
    status VARCHAR(20) NOT NULL DEFAULT 'active', -- 'active', 'resolved', 'suppressed'
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- System Performance Metrics Table
CREATE TABLE system_performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    service_name VARCHAR(100) NOT NULL,
    metric_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    cpu_usage_percent DECIMAL(5, 2),
    memory_usage_percent DECIMAL(5, 2),
    request_count INTEGER,
    response_time_p50_ms DECIMAL(8, 3),
    response_time_p95_ms DECIMAL(8, 3),
    response_time_p99_ms DECIMAL(8, 3),
    error_count INTEGER,
    error_rate DECIMAL(5, 4),
    throughput_rps DECIMAL(10, 3),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create Indexes for Performance

-- Transaction Summary Indexes
CREATE INDEX idx_transaction_summary_date ON transaction_summary(transaction_date);
CREATE INDEX idx_transaction_summary_user_id ON transaction_summary(user_id);
CREATE INDEX idx_transaction_summary_date_user ON transaction_summary(transaction_date, user_id);
CREATE INDEX idx_transaction_summary_fraud_rate ON transaction_summary(fraud_rate) WHERE fraud_rate > 0;

-- Daily Metrics Indexes
CREATE INDEX idx_daily_metrics_date ON daily_metrics(metric_date);
CREATE INDEX idx_daily_metrics_name ON daily_metrics(metric_name);
CREATE INDEX idx_daily_metrics_date_name ON daily_metrics(metric_date, metric_name);
CREATE INDEX idx_daily_metrics_type ON daily_metrics(metric_type);

-- User Behavior Analytics Indexes
CREATE INDEX idx_user_behavior_user_id ON user_behavior_analytics(user_id);
CREATE INDEX idx_user_behavior_date ON user_behavior_analytics(analysis_date);
CREATE INDEX idx_user_behavior_user_date ON user_behavior_analytics(user_id, analysis_date);
CREATE INDEX idx_user_behavior_risk_score ON user_behavior_analytics(risk_score) WHERE risk_score > 0.5;

-- Merchant Analytics Indexes
CREATE INDEX idx_merchant_analytics_merchant_id ON merchant_analytics(merchant_id);
CREATE INDEX idx_merchant_analytics_date ON merchant_analytics(analysis_date);
CREATE INDEX idx_merchant_analytics_merchant_date ON merchant_analytics(merchant_id, analysis_date);
CREATE INDEX idx_merchant_analytics_fraud_rate ON merchant_analytics(fraud_rate) WHERE fraud_rate > 0;
CREATE INDEX idx_merchant_analytics_risk_level ON merchant_analytics(risk_level);

-- Fraud Detection Analytics Indexes
CREATE INDEX idx_fraud_detection_date ON fraud_detection_analytics(analysis_date);
CREATE INDEX idx_fraud_detection_model_version ON fraud_detection_analytics(model_version);
CREATE INDEX idx_fraud_detection_date_model ON fraud_detection_analytics(analysis_date, model_version);

-- Real-time Alerts Indexes
CREATE INDEX idx_alerts_created_at ON real_time_alerts(created_at);
CREATE INDEX idx_alerts_status ON real_time_alerts(status);
CREATE INDEX idx_alerts_severity ON real_time_alerts(severity);
CREATE INDEX idx_alerts_type ON real_time_alerts(alert_type);
CREATE INDEX idx_alerts_entity ON real_time_alerts(entity_type, entity_id);
CREATE INDEX idx_alerts_active ON real_time_alerts(created_at) WHERE status = 'active';

-- System Performance Metrics Indexes
CREATE INDEX idx_system_perf_service_name ON system_performance_metrics(service_name);
CREATE INDEX idx_system_perf_timestamp ON system_performance_metrics(metric_timestamp);
CREATE INDEX idx_system_perf_service_time ON system_performance_metrics(service_name, metric_timestamp);

-- Create Functions for Common Calculations

-- Function to calculate fraud rate
CREATE OR REPLACE FUNCTION calculate_fraud_rate(total_transactions INTEGER, fraud_transactions INTEGER)
RETURNS DECIMAL(5, 4) AS $$
BEGIN
    IF total_transactions = 0 THEN
        RETURN 0.0000;
    END IF;
    RETURN ROUND((fraud_transactions::DECIMAL / total_transactions::DECIMAL), 4);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at columns
CREATE TRIGGER trigger_transaction_summary_updated_at
    BEFORE UPDATE ON transaction_summary
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_user_behavior_analytics_updated_at
    BEFORE UPDATE ON user_behavior_analytics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_merchant_analytics_updated_at
    BEFORE UPDATE ON merchant_analytics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_real_time_alerts_updated_at
    BEFORE UPDATE ON real_time_alerts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create Views for Common Queries

-- Daily Transaction Summary View
CREATE VIEW daily_transaction_summary AS
SELECT 
    transaction_date,
    COUNT(*) as total_records,
    SUM(transaction_count) as total_transactions,
    SUM(total_amount) as total_volume,
    AVG(avg_amount) as avg_transaction_amount,
    SUM(fraud_count) as total_fraud,
    ROUND(SUM(fraud_count)::DECIMAL / SUM(transaction_count)::DECIMAL, 4) as overall_fraud_rate
FROM transaction_summary
GROUP BY transaction_date
ORDER BY transaction_date DESC;

-- High Risk Users View
CREATE VIEW high_risk_users AS
SELECT 
    user_id,
    analysis_date,
    total_transactions,
    total_amount,
    risk_score,
    behavior_flags
FROM user_behavior_analytics
WHERE risk_score > 0.7
ORDER BY risk_score DESC, analysis_date DESC;

-- Merchant Risk Assessment View
CREATE VIEW merchant_risk_assessment AS
SELECT 
    merchant_id,
    merchant_name,
    merchant_category,
    analysis_date,
    transaction_count,
    total_volume,
    fraud_rate,
    risk_level,
    CASE 
        WHEN fraud_rate > 0.05 THEN 'HIGH'
        WHEN fraud_rate > 0.02 THEN 'MEDIUM'
        ELSE 'LOW'
    END as calculated_risk_level
FROM merchant_analytics
WHERE analysis_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY fraud_rate DESC, total_volume DESC;

-- Model Performance Trends View
CREATE VIEW model_performance_trends AS
SELECT 
    analysis_date,
    model_version,
    total_predictions,
    fraud_predictions,
    precision_score,
    recall_score,
    f1_score,
    auc_score,
    avg_prediction_time_ms
FROM fraud_detection_analytics
ORDER BY analysis_date DESC, model_version;

-- Active Critical Alerts View
CREATE VIEW active_critical_alerts AS
SELECT 
    id,
    alert_type,
    severity,
    title,
    description,
    entity_type,
    entity_id,
    created_at,
    EXTRACT(HOUR FROM (CURRENT_TIMESTAMP - created_at)) as hours_open
FROM real_time_alerts
WHERE status = 'active' AND severity IN ('high', 'critical')
ORDER BY 
    CASE severity 
        WHEN 'critical' THEN 1 
        WHEN 'high' THEN 2 
        ELSE 3 
    END,
    created_at DESC;

-- Grant permissions (adjust schema and user as needed)
-- GRANT USAGE ON SCHEMA analytics TO analytics_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA analytics TO analytics_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA analytics TO analytics_user;

-- Insert initial reference data
INSERT INTO daily_metrics (metric_date, metric_name, metric_value, metric_type, dimensions)
VALUES 
    (CURRENT_DATE, 'system_initialized', 1, 'system', '{"version": "1.0.0", "environment": "production"}'),
    (CURRENT_DATE, 'database_migration', 1, 'system', '{"migration_version": "V1", "status": "completed"}');

-- Comments for documentation
COMMENT ON SCHEMA analytics IS 'Analytics schema for IntelliFlow AI Platform';
COMMENT ON TABLE transaction_summary IS 'Daily aggregated transaction data by user';
COMMENT ON TABLE daily_metrics IS 'Daily business and operational metrics';
COMMENT ON TABLE user_behavior_analytics IS 'User behavior analysis and risk scoring';
COMMENT ON TABLE merchant_analytics IS 'Merchant performance and risk analytics';
COMMENT ON TABLE fraud_detection_analytics IS 'ML model performance metrics';
COMMENT ON TABLE real_time_alerts IS 'Real-time alerts and notifications';
COMMENT ON TABLE system_performance_metrics IS 'System performance monitoring data';

COMMENT ON FUNCTION calculate_fraud_rate(INTEGER, INTEGER) IS 'Calculate fraud rate as percentage';
COMMENT ON FUNCTION update_updated_at_column() IS 'Trigger function to update timestamp columns';