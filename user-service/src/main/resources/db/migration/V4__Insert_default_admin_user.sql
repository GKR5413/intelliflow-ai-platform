-- Insert default admin user for development  
-- Password is 'admin123' encoded with BCrypt
INSERT INTO users (
    username, 
    email, 
    password, 
    first_name, 
    last_name, 
    status, 
    email_verified, 
    created_at, 
    updated_at
) VALUES (
    'admin',
    'admin@intelliflow.com',
    '$2a$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/lewGsZY1saDNfq8G6', -- BCrypt hash for 'admin123'
    'Admin',
    'User',
    'ACTIVE',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (username) DO NOTHING;

-- Insert demo user for testing
INSERT INTO users (
    username, 
    email, 
    password, 
    first_name, 
    last_name, 
    status, 
    email_verified, 
    created_at, 
    updated_at
) VALUES (
    'demo',
    'demo@intelliflow.com',
    '$2a$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/lewGsZY1saDNfq8G6', -- BCrypt hash for 'admin123'
    'Demo',
    'User',
    'ACTIVE',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (username) DO NOTHING;

-- Add indexes for better performance
CREATE INDEX IF NOT EXISTS idx_user_email_verified ON users(email_verified);
CREATE INDEX IF NOT EXISTS idx_user_password_reset_token ON users(password_reset_token);
CREATE INDEX IF NOT EXISTS idx_user_email_verification_token ON users(email_verification_token);