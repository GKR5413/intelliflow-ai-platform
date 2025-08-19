-- Create user_sessions table
CREATE TABLE user_sessions (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    ip_address VARCHAR(45),
    user_agent VARCHAR(500),
    device_info VARCHAR(200),
    location VARCHAR(100),
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
    expires_at TIMESTAMP NOT NULL,
    last_activity TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create user_activities table
CREATE TABLE user_activities (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    activity_type VARCHAR(50) NOT NULL,
    description VARCHAR(500),
    ip_address VARCHAR(45),
    user_agent VARCHAR(500),
    resource_id VARCHAR(255),
    resource_type VARCHAR(50),
    metadata TEXT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create indexes
CREATE INDEX idx_session_user ON user_sessions(user_id);
CREATE INDEX idx_session_token ON user_sessions(session_token);
CREATE INDEX idx_session_status ON user_sessions(status);
CREATE INDEX idx_session_expires ON user_sessions(expires_at);
CREATE INDEX idx_activity_user ON user_activities(user_id);
CREATE INDEX idx_activity_type ON user_activities(activity_type);
CREATE INDEX idx_activity_timestamp ON user_activities(timestamp);

-- Add constraints
ALTER TABLE user_sessions ADD CONSTRAINT chk_session_status 
    CHECK (status IN ('ACTIVE', 'EXPIRED', 'REVOKED'));

ALTER TABLE user_activities ADD CONSTRAINT chk_activity_type 
    CHECK (activity_type IN ('LOGIN', 'LOGOUT', 'REGISTRATION', 'PASSWORD_CHANGE', 
                             'PASSWORD_RESET', 'EMAIL_VERIFICATION', 'PROFILE_UPDATE', 
                             'ROLE_CHANGE', 'ACCOUNT_LOCK', 'ACCOUNT_UNLOCK', 
                             'TWO_FACTOR_ENABLE', 'TWO_FACTOR_DISABLE', 'API_ACCESS', 
                             'SUSPICIOUS_ACTIVITY'));

-- Add comments
COMMENT ON TABLE user_sessions IS 'Active user sessions for tracking login sessions';
COMMENT ON TABLE user_activities IS 'Audit log of user activities and system events';
COMMENT ON COLUMN user_sessions.session_token IS 'Unique session identifier';
COMMENT ON COLUMN user_sessions.expires_at IS 'Session expiration timestamp';
COMMENT ON COLUMN user_activities.activity_type IS 'Type of activity performed';
COMMENT ON COLUMN user_activities.metadata IS 'Additional activity data in JSON format';
