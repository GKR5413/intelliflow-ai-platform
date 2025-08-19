package com.intelliflow.transaction.statemachine;

/**
 * Transaction events that trigger state transitions
 */
public enum TransactionEvent {
    INITIATE,
    VALIDATE,
    VALIDATION_SUCCESS,
    VALIDATION_FAILED,
    FRAUD_CHECK,
    FRAUD_APPROVED,
    FRAUD_DECLINED,
    BALANCE_CHECK,
    BALANCE_VERIFIED,
    BALANCE_INSUFFICIENT,
    PROCESS_PAYMENT,
    PAYMENT_AUTHORIZED,
    PAYMENT_FAILED,
    CAPTURE_PAYMENT,
    PAYMENT_CAPTURED,
    COMPLETE,
    FAIL,
    CANCEL,
    REFUND,
    REFUND_COMPLETED,
    DISPUTE
}
