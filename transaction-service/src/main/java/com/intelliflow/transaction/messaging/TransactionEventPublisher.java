package com.intelliflow.transaction.messaging;

import com.intelliflow.shared.messaging.EventMessage;
import com.intelliflow.shared.messaging.KafkaConfig;
import com.intelliflow.transaction.entity.Transaction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.SendResult;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Publisher for transaction events to Kafka
 */
@Service
public class TransactionEventPublisher {
    
    private static final Logger logger = LoggerFactory.getLogger(TransactionEventPublisher.class);
    
    private final KafkaTemplate<String, Object> kafkaTemplate;
    
    @Autowired
    public TransactionEventPublisher(KafkaTemplate<String, Object> kafkaTemplate) {
        this.kafkaTemplate = kafkaTemplate;
    }
    
    public void publishTransactionCreated(Transaction transaction) {
        Map<String, Object> payload = createTransactionPayload(transaction);
        EventMessage event = new EventMessage("TRANSACTION_CREATED", "transaction-service", payload, transaction.getCorrelationId());
        
        publishEvent(KafkaConfig.TRANSACTION_EVENTS_TOPIC, transaction.getReferenceNumber(), event);
        logger.info("Published TRANSACTION_CREATED event for: {}", transaction.getReferenceNumber());
    }
    
    public void publishTransactionValidated(Transaction transaction) {
        Map<String, Object> payload = createTransactionPayload(transaction);
        EventMessage event = new EventMessage("TRANSACTION_VALIDATED", "transaction-service", payload, transaction.getCorrelationId());
        
        publishEvent(KafkaConfig.TRANSACTION_EVENTS_TOPIC, transaction.getReferenceNumber(), event);
        logger.info("Published TRANSACTION_VALIDATED event for: {}", transaction.getReferenceNumber());
    }
    
    public void publishTransactionFailed(Transaction transaction) {
        Map<String, Object> payload = createTransactionPayload(transaction);
        payload.put("failureReason", transaction.getFailureReason());
        EventMessage event = new EventMessage("TRANSACTION_FAILED", "transaction-service", payload, transaction.getCorrelationId());
        
        publishEvent(KafkaConfig.TRANSACTION_EVENTS_TOPIC, transaction.getReferenceNumber(), event);
        logger.info("Published TRANSACTION_FAILED event for: {}", transaction.getReferenceNumber());
    }
    
    public void publishTransactionCompleted(Transaction transaction) {
        Map<String, Object> payload = createTransactionPayload(transaction);
        EventMessage event = new EventMessage("TRANSACTION_COMPLETED", "transaction-service", payload, transaction.getCorrelationId());
        
        publishEvent(KafkaConfig.TRANSACTION_EVENTS_TOPIC, transaction.getReferenceNumber(), event);
        logger.info("Published TRANSACTION_COMPLETED event for: {}", transaction.getReferenceNumber());
    }
    
    public void publishTransactionStatusChanged(Transaction transaction) {
        Map<String, Object> payload = createTransactionPayload(transaction);
        EventMessage event = new EventMessage("TRANSACTION_STATUS_CHANGED", "transaction-service", payload, transaction.getCorrelationId());
        
        publishEvent(KafkaConfig.TRANSACTION_EVENTS_TOPIC, transaction.getReferenceNumber(), event);
        logger.info("Published TRANSACTION_STATUS_CHANGED event for: {}", transaction.getReferenceNumber());
    }
    
    public void publishTransactionCancelled(Transaction transaction) {
        Map<String, Object> payload = createTransactionPayload(transaction);
        EventMessage event = new EventMessage("TRANSACTION_CANCELLED", "transaction-service", payload, transaction.getCorrelationId());
        
        publishEvent(KafkaConfig.TRANSACTION_EVENTS_TOPIC, transaction.getReferenceNumber(), event);
        logger.info("Published TRANSACTION_CANCELLED event for: {}", transaction.getReferenceNumber());
    }
    
    public void publishFraudCheckApproved(Transaction transaction) {
        Map<String, Object> payload = createTransactionPayload(transaction);
        EventMessage event = new EventMessage("FRAUD_CHECK_APPROVED", "transaction-service", payload, transaction.getCorrelationId());
        
        publishEvent(KafkaConfig.FRAUD_EVENTS_TOPIC, transaction.getReferenceNumber(), event);
        logger.info("Published FRAUD_CHECK_APPROVED event for: {}", transaction.getReferenceNumber());
    }
    
    public void publishFraudCheckDeclined(Transaction transaction) {
        Map<String, Object> payload = createTransactionPayload(transaction);
        EventMessage event = new EventMessage("FRAUD_CHECK_DECLINED", "transaction-service", payload, transaction.getCorrelationId());
        
        publishEvent(KafkaConfig.FRAUD_EVENTS_TOPIC, transaction.getReferenceNumber(), event);
        logger.info("Published FRAUD_CHECK_DECLINED event for: {}", transaction.getReferenceNumber());
    }
    
    public void publishBalanceVerified(Transaction transaction) {
        Map<String, Object> payload = createTransactionPayload(transaction);
        EventMessage event = new EventMessage("BALANCE_VERIFIED", "transaction-service", payload, transaction.getCorrelationId());
        
        publishEvent(KafkaConfig.TRANSACTION_EVENTS_TOPIC, transaction.getReferenceNumber(), event);
        logger.info("Published BALANCE_VERIFIED event for: {}", transaction.getReferenceNumber());
    }
    
    public void publishInsufficientBalance(Transaction transaction) {
        Map<String, Object> payload = createTransactionPayload(transaction);
        EventMessage event = new EventMessage("INSUFFICIENT_BALANCE", "transaction-service", payload, transaction.getCorrelationId());
        
        publishEvent(KafkaConfig.TRANSACTION_EVENTS_TOPIC, transaction.getReferenceNumber(), event);
        logger.info("Published INSUFFICIENT_BALANCE event for: {}", transaction.getReferenceNumber());
    }
    
    public void publishPaymentAuthorized(Transaction transaction) {
        Map<String, Object> payload = createTransactionPayload(transaction);
        payload.put("authorizationCode", transaction.getAuthorizationCode());
        EventMessage event = new EventMessage("PAYMENT_AUTHORIZED", "transaction-service", payload, transaction.getCorrelationId());
        
        publishEvent(KafkaConfig.TRANSACTION_EVENTS_TOPIC, transaction.getReferenceNumber(), event);
        logger.info("Published PAYMENT_AUTHORIZED event for: {}", transaction.getReferenceNumber());
    }
    
    public void publishPaymentCaptured(Transaction transaction) {
        Map<String, Object> payload = createTransactionPayload(transaction);
        EventMessage event = new EventMessage("PAYMENT_CAPTURED", "transaction-service", payload, transaction.getCorrelationId());
        
        publishEvent(KafkaConfig.TRANSACTION_EVENTS_TOPIC, transaction.getReferenceNumber(), event);
        logger.info("Published PAYMENT_CAPTURED event for: {}", transaction.getReferenceNumber());
    }
    
    public void publishPaymentFailed(Transaction transaction) {
        Map<String, Object> payload = createTransactionPayload(transaction);
        payload.put("failureReason", transaction.getFailureReason());
        EventMessage event = new EventMessage("PAYMENT_FAILED", "transaction-service", payload, transaction.getCorrelationId());
        
        publishEvent(KafkaConfig.TRANSACTION_EVENTS_TOPIC, transaction.getReferenceNumber(), event);
        logger.info("Published PAYMENT_FAILED event for: {}", transaction.getReferenceNumber());
    }
    
    public void publishRefundInitiated(Transaction transaction) {
        Map<String, Object> payload = createTransactionPayload(transaction);
        EventMessage event = new EventMessage("REFUND_INITIATED", "transaction-service", payload, transaction.getCorrelationId());
        
        publishEvent(KafkaConfig.TRANSACTION_EVENTS_TOPIC, transaction.getReferenceNumber(), event);
        logger.info("Published REFUND_INITIATED event for: {}", transaction.getReferenceNumber());
    }
    
    public void publishRefundCompleted(Transaction transaction) {
        Map<String, Object> payload = createTransactionPayload(transaction);
        EventMessage event = new EventMessage("REFUND_COMPLETED", "transaction-service", payload, transaction.getCorrelationId());
        
        publishEvent(KafkaConfig.TRANSACTION_EVENTS_TOPIC, transaction.getReferenceNumber(), event);
        logger.info("Published REFUND_COMPLETED event for: {}", transaction.getReferenceNumber());
    }
    
    public void publishRefundFailed(Transaction transaction) {
        Map<String, Object> payload = createTransactionPayload(transaction);
        payload.put("failureReason", transaction.getFailureReason());
        EventMessage event = new EventMessage("REFUND_FAILED", "transaction-service", payload, transaction.getCorrelationId());
        
        publishEvent(KafkaConfig.TRANSACTION_EVENTS_TOPIC, transaction.getReferenceNumber(), event);
        logger.info("Published REFUND_FAILED event for: {}", transaction.getReferenceNumber());
    }
    
    public void publishTransactionDisputed(Transaction transaction) {
        Map<String, Object> payload = createTransactionPayload(transaction);
        EventMessage event = new EventMessage("TRANSACTION_DISPUTED", "transaction-service", payload, transaction.getCorrelationId());
        
        publishEvent(KafkaConfig.TRANSACTION_EVENTS_TOPIC, transaction.getReferenceNumber(), event);
        logger.info("Published TRANSACTION_DISPUTED event for: {}", transaction.getReferenceNumber());
    }
    
    private Map<String, Object> createTransactionPayload(Transaction transaction) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("transactionId", transaction.getId());
        payload.put("referenceNumber", transaction.getReferenceNumber());
        payload.put("userId", transaction.getUserId());
        payload.put("type", transaction.getType().toString());
        payload.put("status", transaction.getStatus().toString());
        payload.put("currentState", transaction.getCurrentState().toString());
        payload.put("amount", transaction.getAmount());
        payload.put("currency", transaction.getCurrency());
        payload.put("merchantId", transaction.getMerchantId());
        payload.put("paymentMethod", transaction.getPaymentMethod().toString());
        payload.put("correlationId", transaction.getCorrelationId());
        
        if (transaction.getRiskScore() != null) {
            payload.put("riskScore", transaction.getRiskScore());
        }
        
        if (transaction.getFraudCheckStatus() != null) {
            payload.put("fraudCheckStatus", transaction.getFraudCheckStatus().toString());
        }
        
        return payload;
    }
    
    private void publishEvent(String topic, String key, EventMessage event) {
        try {
            CompletableFuture<SendResult<String, Object>> future = kafkaTemplate.send(topic, key, event);
            
            future.thenAccept(result -> {
                logger.debug("Event published successfully to topic: {}, partition: {}, offset: {}", 
                    topic, result.getRecordMetadata().partition(), result.getRecordMetadata().offset());
            }).exceptionally(throwable -> {
                logger.error("Failed to publish event to topic: {}, key: {}", topic, key, throwable);
                
                // In production, you might want to implement retry logic or dead letter queue
                // For now, just log the error
                return null;
            });
            
        } catch (Exception e) {
            logger.error("Error publishing event to topic: {}, key: {}", topic, key, e);
        }
    }
}
