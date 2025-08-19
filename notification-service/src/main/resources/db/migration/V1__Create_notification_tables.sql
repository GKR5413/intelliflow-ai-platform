-- IntelliFlow Notification Service - Initial Database Migration
-- Version: V1
-- Description: Create notification system tables and indexes

-- Create schema for notifications
CREATE SCHEMA IF NOT EXISTS notifications;

-- Set search path
SET search_path TO notifications, public;

-- Notification Templates Table
CREATE TABLE notification_templates (
    id BIGSERIAL PRIMARY KEY,
    template_name VARCHAR(100) NOT NULL UNIQUE,
    template_type VARCHAR(50) NOT NULL, -- 'email', 'sms', 'push', 'webhook', 'slack'
    subject_template TEXT, -- For email notifications
    body_template TEXT NOT NULL,
    template_variables JSONB, -- JSON array of variable names used in template
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1
);

-- Notification Channels Table
CREATE TABLE notification_channels (
    id BIGSERIAL PRIMARY KEY,
    channel_name VARCHAR(100) NOT NULL UNIQUE,
    channel_type VARCHAR(50) NOT NULL, -- 'email', 'sms', 'push', 'webhook', 'slack'
    configuration JSONB NOT NULL, -- Channel-specific configuration (SMTP settings, API keys, etc.)
    is_active BOOLEAN NOT NULL DEFAULT true,
    priority INTEGER NOT NULL DEFAULT 1, -- Channel priority for fallback
    rate_limit_per_minute INTEGER DEFAULT 60,
    created_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Notification Preferences Table
CREATE TABLE notification_preferences (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    notification_type VARCHAR(100) NOT NULL, -- 'fraud_alert', 'transaction_success', 'password_reset', etc.
    channel_type VARCHAR(50) NOT NULL, -- 'email', 'sms', 'push'
    is_enabled BOOLEAN NOT NULL DEFAULT true,
    delivery_time_start TIME DEFAULT '09:00:00', -- Preferred delivery window start
    delivery_time_end TIME DEFAULT '21:00:00', -- Preferred delivery window end
    timezone VARCHAR(50) DEFAULT 'UTC',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, notification_type, channel_type)
);

-- Notification Queue Table
CREATE TABLE notification_queue (
    id BIGSERIAL PRIMARY KEY,
    notification_type VARCHAR(100) NOT NULL,
    recipient_type VARCHAR(50) NOT NULL, -- 'user', 'admin', 'group'
    recipient_id VARCHAR(100) NOT NULL,
    channel_type VARCHAR(50) NOT NULL,
    channel_id BIGINT REFERENCES notification_channels(id),
    template_id BIGINT REFERENCES notification_templates(id),
    subject VARCHAR(255),
    message_body TEXT NOT NULL,
    template_variables JSONB, -- Variables used to render the template
    priority INTEGER NOT NULL DEFAULT 3, -- 1=high, 2=medium, 3=low
    scheduled_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    status VARCHAR(20) NOT NULL DEFAULT 'pending', -- 'pending', 'processing', 'sent', 'failed', 'cancelled'
    error_message TEXT,
    sent_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Notification Delivery Log Table
CREATE TABLE notification_delivery_log (
    id BIGSERIAL PRIMARY KEY,
    queue_id BIGINT NOT NULL REFERENCES notification_queue(id),
    channel_name VARCHAR(100) NOT NULL,
    channel_type VARCHAR(50) NOT NULL,
    recipient VARCHAR(255) NOT NULL,
    status VARCHAR(20) NOT NULL, -- 'sent', 'delivered', 'bounced', 'failed', 'clicked', 'opened'
    provider_message_id VARCHAR(255), -- External provider's message ID
    provider_response JSONB, -- Full response from the notification provider
    delivery_time_ms INTEGER, -- Time taken to deliver
    delivered_at TIMESTAMP WITH TIME ZONE,
    read_at TIMESTAMP WITH TIME ZONE, -- When notification was read/opened
    clicked_at TIMESTAMP WITH TIME ZONE, -- When any links were clicked
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Notification Subscribers Table (for group notifications)
CREATE TABLE notification_subscribers (
    id BIGSERIAL PRIMARY KEY,
    group_name VARCHAR(100) NOT NULL,
    subscriber_type VARCHAR(50) NOT NULL, -- 'user', 'email', 'phone'
    subscriber_identifier VARCHAR(255) NOT NULL, -- user_id, email, phone number
    subscription_types JSONB, -- JSON array of notification types they're subscribed to
    is_active BOOLEAN NOT NULL DEFAULT true,
    subscribed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    unsubscribed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(group_name, subscriber_type, subscriber_identifier)
);

-- Notification Webhooks Table
CREATE TABLE notification_webhooks (
    id BIGSERIAL PRIMARY KEY,
    webhook_name VARCHAR(100) NOT NULL UNIQUE,
    webhook_url VARCHAR(500) NOT NULL,
    http_method VARCHAR(10) NOT NULL DEFAULT 'POST',
    headers JSONB, -- HTTP headers to send
    authentication JSONB, -- Authentication configuration (API keys, tokens, etc.)
    event_types JSONB NOT NULL, -- JSON array of event types this webhook handles
    is_active BOOLEAN NOT NULL DEFAULT true,
    timeout_seconds INTEGER NOT NULL DEFAULT 30,
    retry_attempts INTEGER NOT NULL DEFAULT 3,
    created_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Webhook Delivery Log Table
CREATE TABLE webhook_delivery_log (
    id BIGSERIAL PRIMARY KEY,
    webhook_id BIGINT NOT NULL REFERENCES notification_webhooks(id),
    event_type VARCHAR(100) NOT NULL,
    payload JSONB NOT NULL,
    http_status_code INTEGER,
    response_body TEXT,
    response_headers JSONB,
    delivery_time_ms INTEGER,
    attempts INTEGER NOT NULL DEFAULT 1,
    status VARCHAR(20) NOT NULL, -- 'sent', 'failed', 'retry'
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    delivered_at TIMESTAMP WITH TIME ZONE
);

-- Create Indexes for Performance

-- Notification Templates Indexes
CREATE INDEX idx_notification_templates_type ON notification_templates(template_type);
CREATE INDEX idx_notification_templates_active ON notification_templates(is_active) WHERE is_active = true;

-- Notification Channels Indexes
CREATE INDEX idx_notification_channels_type ON notification_channels(channel_type);
CREATE INDEX idx_notification_channels_active ON notification_channels(is_active) WHERE is_active = true;
CREATE INDEX idx_notification_channels_priority ON notification_channels(priority);

-- Notification Preferences Indexes
CREATE INDEX idx_notification_preferences_user_id ON notification_preferences(user_id);
CREATE INDEX idx_notification_preferences_type ON notification_preferences(notification_type);
CREATE INDEX idx_notification_preferences_user_type ON notification_preferences(user_id, notification_type);

-- Notification Queue Indexes
CREATE INDEX idx_notification_queue_status ON notification_queue(status);
CREATE INDEX idx_notification_queue_scheduled_at ON notification_queue(scheduled_at);
CREATE INDEX idx_notification_queue_pending ON notification_queue(scheduled_at, priority) WHERE status = 'pending';
CREATE INDEX idx_notification_queue_processing ON notification_queue(status, created_at) WHERE status = 'processing';
CREATE INDEX idx_notification_queue_recipient ON notification_queue(recipient_type, recipient_id);
CREATE INDEX idx_notification_queue_channel_type ON notification_queue(channel_type);
CREATE INDEX idx_notification_queue_priority ON notification_queue(priority);

-- Notification Delivery Log Indexes
CREATE INDEX idx_delivery_log_queue_id ON notification_delivery_log(queue_id);
CREATE INDEX idx_delivery_log_status ON notification_delivery_log(status);
CREATE INDEX idx_delivery_log_channel_type ON notification_delivery_log(channel_type);
CREATE INDEX idx_delivery_log_created_at ON notification_delivery_log(created_at);
CREATE INDEX idx_delivery_log_recipient ON notification_delivery_log(recipient);

-- Notification Subscribers Indexes
CREATE INDEX idx_notification_subscribers_group ON notification_subscribers(group_name);
CREATE INDEX idx_notification_subscribers_identifier ON notification_subscribers(subscriber_identifier);
CREATE INDEX idx_notification_subscribers_active ON notification_subscribers(is_active) WHERE is_active = true;

-- Webhook Indexes
CREATE INDEX idx_notification_webhooks_active ON notification_webhooks(is_active) WHERE is_active = true;
CREATE INDEX idx_webhook_delivery_log_webhook_id ON webhook_delivery_log(webhook_id);
CREATE INDEX idx_webhook_delivery_log_status ON webhook_delivery_log(status);
CREATE INDEX idx_webhook_delivery_log_created_at ON webhook_delivery_log(created_at);

-- Create Functions

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to get user notification preferences
CREATE OR REPLACE FUNCTION get_user_notification_preference(
    p_user_id BIGINT,
    p_notification_type VARCHAR(100),
    p_channel_type VARCHAR(50)
)
RETURNS BOOLEAN AS $$
DECLARE
    preference_exists BOOLEAN;
    is_enabled BOOLEAN;
BEGIN
    SELECT EXISTS(
        SELECT 1 FROM notification_preferences 
        WHERE user_id = p_user_id 
        AND notification_type = p_notification_type 
        AND channel_type = p_channel_type
    ) INTO preference_exists;
    
    IF preference_exists THEN
        SELECT np.is_enabled INTO is_enabled
        FROM notification_preferences np
        WHERE np.user_id = p_user_id 
        AND np.notification_type = p_notification_type 
        AND np.channel_type = p_channel_type;
        
        RETURN is_enabled;
    ELSE
        -- Default to enabled if no preference is set
        RETURN TRUE;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to check if notification should be sent based on delivery window
CREATE OR REPLACE FUNCTION is_in_delivery_window(
    p_user_id BIGINT,
    p_notification_type VARCHAR(100),
    p_current_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
)
RETURNS BOOLEAN AS $$
DECLARE
    delivery_start TIME;
    delivery_end TIME;
    user_timezone VARCHAR(50);
    local_time TIME;
BEGIN
    SELECT delivery_time_start, delivery_time_end, timezone
    INTO delivery_start, delivery_end, user_timezone
    FROM notification_preferences
    WHERE user_id = p_user_id 
    AND notification_type = p_notification_type
    LIMIT 1;
    
    IF delivery_start IS NULL THEN
        -- No preference set, allow anytime
        RETURN TRUE;
    END IF;
    
    -- Convert current time to user's timezone
    local_time := (p_current_time AT TIME ZONE user_timezone)::TIME;
    
    -- Check if current time is within delivery window
    IF delivery_start <= delivery_end THEN
        RETURN local_time BETWEEN delivery_start AND delivery_end;
    ELSE
        -- Handle overnight windows (e.g., 22:00 to 06:00)
        RETURN local_time >= delivery_start OR local_time <= delivery_end;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at columns
CREATE TRIGGER trigger_notification_templates_updated_at
    BEFORE UPDATE ON notification_templates
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_notification_channels_updated_at
    BEFORE UPDATE ON notification_channels
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_notification_preferences_updated_at
    BEFORE UPDATE ON notification_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_notification_queue_updated_at
    BEFORE UPDATE ON notification_queue
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_notification_subscribers_updated_at
    BEFORE UPDATE ON notification_subscribers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_notification_webhooks_updated_at
    BEFORE UPDATE ON notification_webhooks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create Views for Common Queries

-- Pending Notifications View
CREATE VIEW pending_notifications AS
SELECT 
    nq.id,
    nq.notification_type,
    nq.recipient_type,
    nq.recipient_id,
    nq.channel_type,
    nc.channel_name,
    nq.subject,
    nq.priority,
    nq.scheduled_at,
    nq.attempts,
    nq.max_attempts,
    nq.created_at
FROM notification_queue nq
JOIN notification_channels nc ON nq.channel_id = nc.id
WHERE nq.status = 'pending' 
AND nq.scheduled_at <= CURRENT_TIMESTAMP
AND nc.is_active = true
ORDER BY nq.priority ASC, nq.scheduled_at ASC;

-- Notification Delivery Statistics View
CREATE VIEW notification_delivery_stats AS
SELECT 
    nq.notification_type,
    nq.channel_type,
    DATE(nq.created_at) as notification_date,
    COUNT(*) as total_sent,
    COUNT(CASE WHEN ndl.status = 'delivered' THEN 1 END) as delivered_count,
    COUNT(CASE WHEN ndl.status = 'bounced' THEN 1 END) as bounced_count,
    COUNT(CASE WHEN ndl.status = 'failed' THEN 1 END) as failed_count,
    COUNT(CASE WHEN ndl.status = 'opened' THEN 1 END) as opened_count,
    COUNT(CASE WHEN ndl.status = 'clicked' THEN 1 END) as clicked_count,
    ROUND(
        COUNT(CASE WHEN ndl.status = 'delivered' THEN 1 END)::DECIMAL / 
        COUNT(*)::DECIMAL * 100, 2
    ) as delivery_rate,
    AVG(ndl.delivery_time_ms) as avg_delivery_time_ms
FROM notification_queue nq
LEFT JOIN notification_delivery_log ndl ON nq.id = ndl.queue_id
WHERE nq.status = 'sent'
GROUP BY nq.notification_type, nq.channel_type, DATE(nq.created_at)
ORDER BY notification_date DESC, nq.notification_type;

-- Failed Notifications View
CREATE VIEW failed_notifications AS
SELECT 
    nq.id,
    nq.notification_type,
    nq.recipient_type,
    nq.recipient_id,
    nq.channel_type,
    nq.attempts,
    nq.max_attempts,
    nq.error_message,
    nq.created_at,
    nq.updated_at
FROM notification_queue nq
WHERE nq.status = 'failed'
ORDER BY nq.updated_at DESC;

-- Webhook Delivery Statistics View
CREATE VIEW webhook_delivery_stats AS
SELECT 
    nw.webhook_name,
    wdl.event_type,
    DATE(wdl.created_at) as delivery_date,
    COUNT(*) as total_deliveries,
    COUNT(CASE WHEN wdl.status = 'sent' THEN 1 END) as successful_deliveries,
    COUNT(CASE WHEN wdl.status = 'failed' THEN 1 END) as failed_deliveries,
    ROUND(
        COUNT(CASE WHEN wdl.status = 'sent' THEN 1 END)::DECIMAL / 
        COUNT(*)::DECIMAL * 100, 2
    ) as success_rate,
    AVG(wdl.delivery_time_ms) as avg_delivery_time_ms
FROM notification_webhooks nw
JOIN webhook_delivery_log wdl ON nw.id = wdl.webhook_id
GROUP BY nw.webhook_name, wdl.event_type, DATE(wdl.created_at)
ORDER BY delivery_date DESC, nw.webhook_name;

-- Insert default notification templates
INSERT INTO notification_templates (template_name, template_type, subject_template, body_template, template_variables) VALUES
('fraud_alert_email', 'email', 'Security Alert: Suspicious Activity Detected', 
 'Dear {{user_name}},\n\nWe detected suspicious activity on your account at {{transaction_time}}.\n\nTransaction Details:\n- Amount: {{amount}}\n- Merchant: {{merchant_name}}\n- Location: {{location}}\n\nIf this was not you, please contact us immediately.\n\nBest regards,\nIntelliFlow Security Team', 
 '["user_name", "transaction_time", "amount", "merchant_name", "location"]'),
 
('transaction_success_email', 'email', 'Transaction Successful', 
 'Dear {{user_name}},\n\nYour transaction has been processed successfully.\n\nDetails:\n- Amount: {{amount}}\n- Merchant: {{merchant_name}}\n- Date: {{transaction_date}}\n- Reference: {{transaction_id}}\n\nThank you for using IntelliFlow.\n\nBest regards,\nIntelliFlow Team', 
 '["user_name", "amount", "merchant_name", "transaction_date", "transaction_id"]'),
 
('password_reset_email', 'email', 'Password Reset Request', 
 'Dear {{user_name}},\n\nYou requested a password reset for your IntelliFlow account.\n\nClick the link below to reset your password:\n{{reset_link}}\n\nThis link will expire in {{expiry_hours}} hours.\n\nIf you did not request this, please ignore this email.\n\nBest regards,\nIntelliFlow Team', 
 '["user_name", "reset_link", "expiry_hours"]'),
 
('fraud_alert_sms', 'sms', NULL, 
 'IntelliFlow Security Alert: Suspicious {{amount}} transaction detected at {{merchant_name}}. If not you, call us immediately. Ref: {{transaction_id}}', 
 '["amount", "merchant_name", "transaction_id"]'),
 
('account_locked_slack', 'slack', NULL,
 '🚨 *Security Alert* 🚨\n\n*Account Locked*\nUser ID: {{user_id}}\nReason: {{lock_reason}}\nTime: {{lock_time}}\n\n*Action Required:* Security team review needed.',
 '["user_id", "lock_reason", "lock_time"]');

-- Insert default notification channels
INSERT INTO notification_channels (channel_name, channel_type, configuration, priority) VALUES
('primary_email', 'email', '{"smtp_host": "smtp.gmail.com", "smtp_port": 587, "use_tls": true}', 1),
('backup_email', 'email', '{"smtp_host": "smtp-backup.gmail.com", "smtp_port": 587, "use_tls": true}', 2),
('sms_provider', 'sms', '{"provider": "twilio", "account_sid": "", "auth_token": ""}', 1),
('push_notifications', 'push', '{"provider": "firebase", "server_key": ""}', 1),
('security_slack', 'slack', '{"webhook_url": "", "channel": "#security-alerts"}', 1);

-- Comments for documentation
COMMENT ON SCHEMA notifications IS 'Notification system schema for IntelliFlow AI Platform';
COMMENT ON TABLE notification_templates IS 'Template definitions for different notification types';
COMMENT ON TABLE notification_channels IS 'Configured notification delivery channels';
COMMENT ON TABLE notification_preferences IS 'User-specific notification preferences and settings';
COMMENT ON TABLE notification_queue IS 'Queue of notifications to be sent';
COMMENT ON TABLE notification_delivery_log IS 'Log of notification delivery attempts and status';
COMMENT ON TABLE notification_subscribers IS 'Group subscription management';
COMMENT ON TABLE notification_webhooks IS 'Webhook configurations for external integrations';
COMMENT ON TABLE webhook_delivery_log IS 'Log of webhook delivery attempts';

COMMENT ON FUNCTION get_user_notification_preference(BIGINT, VARCHAR, VARCHAR) IS 'Check if user has enabled a specific notification type for a channel';
COMMENT ON FUNCTION is_in_delivery_window(BIGINT, VARCHAR, TIMESTAMP WITH TIME ZONE) IS 'Check if current time is within user delivery preferences';