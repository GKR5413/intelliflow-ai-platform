package com.intelliflow.transaction.entity;

public enum TransactionStatus {
    PENDING,
    PROCESSING,
    SUCCESS,
    FAILED,
    CANCELLED,
    EXPIRED,
    REFUNDED,
    PARTIALLY_REFUNDED,
    DISPUTED,
    CHARGEBACK
}
