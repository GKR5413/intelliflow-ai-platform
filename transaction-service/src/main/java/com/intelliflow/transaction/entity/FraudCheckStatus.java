package com.intelliflow.transaction.entity;

public enum FraudCheckStatus {
    PENDING,
    APPROVED,
    DECLINED,
    REVIEW_REQUIRED,
    MANUAL_REVIEW,
    WHITELISTED,
    BLACKLISTED,
    TIMEOUT
}
