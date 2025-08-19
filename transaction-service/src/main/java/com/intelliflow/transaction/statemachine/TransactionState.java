package com.intelliflow.transaction.statemachine;

/**
 * Transaction states for state machine
 */
public enum TransactionState {
    INITIATED,
    VALIDATED,
    FRAUD_CHECKING,
    FRAUD_APPROVED,
    FRAUD_DECLINED,
    BALANCE_CHECKING,
    BALANCE_VERIFIED,
    BALANCE_INSUFFICIENT,
    PROCESSING,
    PAYMENT_AUTHORIZED,
    PAYMENT_CAPTURED,
    COMPLETED,
    FAILED,
    CANCELLED,
    REFUND_INITIATED,
    REFUND_COMPLETED,
    DISPUTED
}
