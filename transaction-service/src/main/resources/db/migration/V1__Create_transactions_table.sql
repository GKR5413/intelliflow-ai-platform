-- Create transactions table
CREATE TABLE transactions (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    reference_number VARCHAR(50) UNIQUE NOT NULL,
    type VARCHAR(20) NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'PENDING',
    amount DECIMAL(15,2) NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    description VARCHAR(500),
    merchant_id VARCHAR(255),
    merchant_name VARCHAR(100),
    merchant_category VARCHAR(50),
    payment_method VARCHAR(30),
    payment_instrument_id VARCHAR(255),
    from_account_id VARCHAR(255),
    to_account_id VARCHAR(255),
    external_transaction_id VARCHAR(255),
    processing_fee DECIMAL(10,2) DEFAULT 0.00,
    exchange_rate DECIMAL(10,6),
    original_amount DECIMAL(15,2),
    original_currency VARCHAR(3),
    ip_address VARCHAR(45),
    user_agent VARCHAR(500),
    device_fingerprint VARCHAR(255),
    location_data VARCHAR(200),
    risk_score DECIMAL(5,2),
    fraud_check_status VARCHAR(20) DEFAULT 'PENDING',
    fraud_check_details TEXT,
    authorization_code VARCHAR(255),
    settlement_date TIMESTAMP,
    processed_at TIMESTAMP,
    failed_at TIMESTAMP,
    failure_reason VARCHAR(500),
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    notification_sent BOOLEAN DEFAULT FALSE,
    metadata TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version BIGINT DEFAULT 0
);

-- Create indexes for optimal query performance
CREATE INDEX idx_transaction_user_id ON transactions(user_id);
CREATE INDEX idx_transaction_status ON transactions(status);
CREATE INDEX idx_transaction_type ON transactions(type);
CREATE INDEX idx_transaction_reference ON transactions(reference_number);
CREATE INDEX idx_transaction_created_at ON transactions(created_at);
CREATE INDEX idx_transaction_amount ON transactions(amount);
CREATE INDEX idx_transaction_merchant ON transactions(merchant_id);
CREATE INDEX idx_transaction_payment_method ON transactions(payment_method);
CREATE INDEX idx_transaction_fraud_status ON transactions(fraud_check_status);
CREATE INDEX idx_transaction_processed_at ON transactions(processed_at);
CREATE INDEX idx_transaction_settlement_date ON transactions(settlement_date);

-- Composite indexes for common query patterns
CREATE INDEX idx_transaction_user_status ON transactions(user_id, status);
CREATE INDEX idx_transaction_user_type ON transactions(user_id, type);
CREATE INDEX idx_transaction_status_created ON transactions(status, created_at);
CREATE INDEX idx_transaction_amount_currency ON transactions(amount, currency);

-- Add constraints
ALTER TABLE transactions ADD CONSTRAINT chk_transaction_type 
    CHECK (type IN ('PAYMENT', 'REFUND', 'TRANSFER', 'DEPOSIT', 'WITHDRAWAL', 
                   'CHARGE', 'AUTHORIZATION', 'CAPTURE', 'VOID', 'SETTLEMENT'));

ALTER TABLE transactions ADD CONSTRAINT chk_transaction_status 
    CHECK (status IN ('PENDING', 'PROCESSING', 'SUCCESS', 'FAILED', 'CANCELLED', 
                     'EXPIRED', 'REFUNDED', 'PARTIALLY_REFUNDED', 'DISPUTED', 'CHARGEBACK'));

ALTER TABLE transactions ADD CONSTRAINT chk_payment_method 
    CHECK (payment_method IN ('CREDIT_CARD', 'DEBIT_CARD', 'BANK_TRANSFER', 'DIGITAL_WALLET', 
                              'CRYPTOCURRENCY', 'MOBILE_PAYMENT', 'PAYPAL', 'APPLE_PAY', 
                              'GOOGLE_PAY', 'ACH', 'WIRE_TRANSFER', 'CHECK', 'CASH'));

ALTER TABLE transactions ADD CONSTRAINT chk_fraud_check_status 
    CHECK (fraud_check_status IN ('PENDING', 'APPROVED', 'DECLINED', 'REVIEW_REQUIRED', 
                                  'MANUAL_REVIEW', 'WHITELISTED', 'BLACKLISTED', 'TIMEOUT'));

ALTER TABLE transactions ADD CONSTRAINT chk_amount_positive 
    CHECK (amount > 0);

ALTER TABLE transactions ADD CONSTRAINT chk_processing_fee_non_negative 
    CHECK (processing_fee >= 0);

ALTER TABLE transactions ADD CONSTRAINT chk_retry_count_non_negative 
    CHECK (retry_count >= 0);

ALTER TABLE transactions ADD CONSTRAINT chk_max_retries_positive 
    CHECK (max_retries > 0);

-- Add comments
COMMENT ON TABLE transactions IS 'Main transactions table storing all payment transactions';
COMMENT ON COLUMN transactions.reference_number IS 'Unique transaction reference number';
COMMENT ON COLUMN transactions.amount IS 'Transaction amount in the specified currency';
COMMENT ON COLUMN transactions.currency IS 'ISO 4217 currency code';
COMMENT ON COLUMN transactions.risk_score IS 'Fraud risk score from 0.00 to 1.00';
COMMENT ON COLUMN transactions.fraud_check_status IS 'Status of fraud detection check';
COMMENT ON COLUMN transactions.retry_count IS 'Number of retry attempts made';
COMMENT ON COLUMN transactions.version IS 'Version field for optimistic locking';
