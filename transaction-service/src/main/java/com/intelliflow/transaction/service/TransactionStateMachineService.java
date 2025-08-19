package com.intelliflow.transaction.service;

import com.intelliflow.transaction.dto.TransactionValidationDto;
import com.intelliflow.transaction.entity.Transaction;
import com.intelliflow.transaction.repository.TransactionRepository;
import com.intelliflow.transaction.statemachine.TransactionEvent;
import com.intelliflow.transaction.statemachine.TransactionState;
import com.intelliflow.transaction.messaging.TransactionEventPublisher;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.statemachine.StateContext;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;

@Service
@Transactional
public class TransactionStateMachineService {
    
    private static final Logger logger = LoggerFactory.getLogger(TransactionStateMachineService.class);
    
    private final TransactionRepository transactionRepository;
    private final TransactionValidationService validationService;
    private final FraudCheckService fraudCheckService;
    private final BalanceVerificationService balanceService;
    private final PaymentProcessingService paymentService;
    private final TransactionEventPublisher eventPublisher;
    
    @Autowired
    public TransactionStateMachineService(
            TransactionRepository transactionRepository,
            TransactionValidationService validationService,
            FraudCheckService fraudCheckService,
            BalanceVerificationService balanceService,
            PaymentProcessingService paymentService,
            TransactionEventPublisher eventPublisher) {
        this.transactionRepository = transactionRepository;
        this.validationService = validationService;
        this.fraudCheckService = fraudCheckService;
        this.balanceService = balanceService;
        this.paymentService = paymentService;
        this.eventPublisher = eventPublisher;
    }
    
    // State machine action handlers
    
    public void onValidationSuccess(StateContext<TransactionState, TransactionEvent> context) {
        Long transactionId = context.getExtendedState().get("transactionId", Long.class);
        logger.info("Transaction {} validation successful", transactionId);
        
        try {
            Transaction transaction = getTransaction(transactionId);
            transaction.setStatus(com.intelliflow.transaction.entity.TransactionStatus.PROCESSING);
            transactionRepository.save(transaction);
            
            // Publish validation success event
            eventPublisher.publishTransactionValidated(transaction);
            
            // Trigger fraud check
            context.getStateMachine().sendEvent(TransactionEvent.FRAUD_CHECK);
            
        } catch (Exception e) {
            logger.error("Error in validation success handler for transaction {}", transactionId, e);
            context.getStateMachine().sendEvent(TransactionEvent.FAIL);
        }
    }
    
    public void onValidationFailed(StateContext<TransactionState, TransactionEvent> context) {
        Long transactionId = context.getExtendedState().get("transactionId", Long.class);
        String failureReason = context.getExtendedState().get("failureReason", String.class);
        
        logger.warn("Transaction {} validation failed: {}", transactionId, failureReason);
        
        try {
            Transaction transaction = getTransaction(transactionId);
            transaction.setStatus(com.intelliflow.transaction.entity.TransactionStatus.FAILED);
            transaction.setFailureReason(failureReason);
            transaction.setFailedAt(LocalDateTime.now());
            transactionRepository.save(transaction);
            
            // Publish validation failed event
            eventPublisher.publishTransactionFailed(transaction);
            
        } catch (Exception e) {
            logger.error("Error in validation failed handler for transaction {}", transactionId, e);
        }
    }
    
    public void onFraudCheck(StateContext<TransactionState, TransactionEvent> context) {
        Long transactionId = context.getExtendedState().get("transactionId", Long.class);
        logger.info("Starting fraud check for transaction {}", transactionId);
        
        try {
            Transaction transaction = getTransaction(transactionId);
            
            // Perform fraud check asynchronously
            fraudCheckService.performFraudCheck(transaction)
                .thenAccept(fraudResult -> {
                    if (fraudResult.isApproved()) {
                        context.getStateMachine().sendEvent(TransactionEvent.FRAUD_APPROVED);
                    } else {
                        context.getExtendedState().getVariables().put("fraudReason", fraudResult.getReason());
                        context.getStateMachine().sendEvent(TransactionEvent.FRAUD_DECLINED);
                    }
                })
                .exceptionally(throwable -> {
                    logger.error("Fraud check failed for transaction {}", transactionId, throwable);
                    context.getStateMachine().sendEvent(TransactionEvent.FAIL);
                    return null;
                });
                
        } catch (Exception e) {
            logger.error("Error initiating fraud check for transaction {}", transactionId, e);
            context.getStateMachine().sendEvent(TransactionEvent.FAIL);
        }
    }
    
    public void onFraudApproved(StateContext<TransactionState, TransactionEvent> context) {
        Long transactionId = context.getExtendedState().get("transactionId", Long.class);
        logger.info("Fraud check approved for transaction {}", transactionId);
        
        try {
            Transaction transaction = getTransaction(transactionId);
            transaction.setFraudCheckStatus(com.intelliflow.transaction.entity.FraudCheckStatus.APPROVED);
            transactionRepository.save(transaction);
            
            // Publish fraud approved event
            eventPublisher.publishFraudCheckApproved(transaction);
            
            // Trigger balance check
            context.getStateMachine().sendEvent(TransactionEvent.BALANCE_CHECK);
            
        } catch (Exception e) {
            logger.error("Error in fraud approved handler for transaction {}", transactionId, e);
            context.getStateMachine().sendEvent(TransactionEvent.FAIL);
        }
    }
    
    public void onFraudDeclined(StateContext<TransactionState, TransactionEvent> context) {
        Long transactionId = context.getExtendedState().get("transactionId", Long.class);
        String fraudReason = context.getExtendedState().get("fraudReason", String.class);
        
        logger.warn("Fraud check declined for transaction {}: {}", transactionId, fraudReason);
        
        try {
            Transaction transaction = getTransaction(transactionId);
            transaction.setStatus(com.intelliflow.transaction.entity.TransactionStatus.FAILED);
            transaction.setFraudCheckStatus(com.intelliflow.transaction.entity.FraudCheckStatus.DECLINED);
            transaction.setFailureReason("Fraud check declined: " + fraudReason);
            transaction.setFailedAt(LocalDateTime.now());
            transactionRepository.save(transaction);
            
            // Publish fraud declined event
            eventPublisher.publishFraudCheckDeclined(transaction);
            
        } catch (Exception e) {
            logger.error("Error in fraud declined handler for transaction {}", transactionId, e);
        }
    }
    
    public void onBalanceCheck(StateContext<TransactionState, TransactionEvent> context) {
        Long transactionId = context.getExtendedState().get("transactionId", Long.class);
        logger.info("Starting balance check for transaction {}", transactionId);
        
        try {
            Transaction transaction = getTransaction(transactionId);
            
            // Perform balance verification asynchronously
            balanceService.verifyBalance(transaction)
                .thenAccept(balanceResult -> {
                    if (balanceResult.isSufficient()) {
                        context.getStateMachine().sendEvent(TransactionEvent.BALANCE_VERIFIED);
                    } else {
                        context.getExtendedState().getVariables().put("balanceReason", balanceResult.getReason());
                        context.getStateMachine().sendEvent(TransactionEvent.BALANCE_INSUFFICIENT);
                    }
                })
                .exceptionally(throwable -> {
                    logger.error("Balance check failed for transaction {}", transactionId, throwable);
                    context.getStateMachine().sendEvent(TransactionEvent.FAIL);
                    return null;
                });
                
        } catch (Exception e) {
            logger.error("Error initiating balance check for transaction {}", transactionId, e);
            context.getStateMachine().sendEvent(TransactionEvent.FAIL);
        }
    }
    
    public void onBalanceVerified(StateContext<TransactionState, TransactionEvent> context) {
        Long transactionId = context.getExtendedState().get("transactionId", Long.class);
        logger.info("Balance verified for transaction {}", transactionId);
        
        try {
            Transaction transaction = getTransaction(transactionId);
            
            // Publish balance verified event
            eventPublisher.publishBalanceVerified(transaction);
            
            // Trigger payment processing
            context.getStateMachine().sendEvent(TransactionEvent.PROCESS_PAYMENT);
            
        } catch (Exception e) {
            logger.error("Error in balance verified handler for transaction {}", transactionId, e);
            context.getStateMachine().sendEvent(TransactionEvent.FAIL);
        }
    }
    
    public void onInsufficientBalance(StateContext<TransactionState, TransactionEvent> context) {
        Long transactionId = context.getExtendedState().get("transactionId", Long.class);
        String balanceReason = context.getExtendedState().get("balanceReason", String.class);
        
        logger.warn("Insufficient balance for transaction {}: {}", transactionId, balanceReason);
        
        try {
            Transaction transaction = getTransaction(transactionId);
            transaction.setStatus(com.intelliflow.transaction.entity.TransactionStatus.FAILED);
            transaction.setFailureReason("Insufficient balance: " + balanceReason);
            transaction.setFailedAt(LocalDateTime.now());
            transactionRepository.save(transaction);
            
            // Publish insufficient balance event
            eventPublisher.publishInsufficientBalance(transaction);
            
        } catch (Exception e) {
            logger.error("Error in insufficient balance handler for transaction {}", transactionId, e);
        }
    }
    
    public void onProcessPayment(StateContext<TransactionState, TransactionEvent> context) {
        Long transactionId = context.getExtendedState().get("transactionId", Long.class);
        logger.info("Processing payment for transaction {}", transactionId);
        
        try {
            Transaction transaction = getTransaction(transactionId);
            
            // Process payment asynchronously
            paymentService.processPayment(transaction)
                .thenAccept(paymentResult -> {
                    if (paymentResult.isAuthorized()) {
                        context.getExtendedState().getVariables().put("authorizationCode", paymentResult.getAuthorizationCode());
                        context.getStateMachine().sendEvent(TransactionEvent.PAYMENT_AUTHORIZED);
                    } else {
                        context.getExtendedState().getVariables().put("paymentFailureReason", paymentResult.getFailureReason());
                        context.getStateMachine().sendEvent(TransactionEvent.PAYMENT_FAILED);
                    }
                })
                .exceptionally(throwable -> {
                    logger.error("Payment processing failed for transaction {}", transactionId, throwable);
                    context.getStateMachine().sendEvent(TransactionEvent.PAYMENT_FAILED);
                    return null;
                });
                
        } catch (Exception e) {
            logger.error("Error initiating payment processing for transaction {}", transactionId, e);
            context.getStateMachine().sendEvent(TransactionEvent.FAIL);
        }
    }
    
    public void onPaymentAuthorized(StateContext<TransactionState, TransactionEvent> context) {
        Long transactionId = context.getExtendedState().get("transactionId", Long.class);
        String authorizationCode = context.getExtendedState().get("authorizationCode", String.class);
        
        logger.info("Payment authorized for transaction {} with code {}", transactionId, authorizationCode);
        
        try {
            Transaction transaction = getTransaction(transactionId);
            transaction.setAuthorizationCode(authorizationCode);
            transactionRepository.save(transaction);
            
            // Publish payment authorized event
            eventPublisher.publishPaymentAuthorized(transaction);
            
            // Trigger payment capture
            context.getStateMachine().sendEvent(TransactionEvent.CAPTURE_PAYMENT);
            
        } catch (Exception e) {
            logger.error("Error in payment authorized handler for transaction {}", transactionId, e);
            context.getStateMachine().sendEvent(TransactionEvent.FAIL);
        }
    }
    
    public void onPaymentFailed(StateContext<TransactionState, TransactionEvent> context) {
        Long transactionId = context.getExtendedState().get("transactionId", Long.class);
        String failureReason = context.getExtendedState().get("paymentFailureReason", String.class);
        
        logger.warn("Payment failed for transaction {}: {}", transactionId, failureReason);
        
        try {
            Transaction transaction = getTransaction(transactionId);
            transaction.setStatus(com.intelliflow.transaction.entity.TransactionStatus.FAILED);
            transaction.setFailureReason("Payment failed: " + failureReason);
            transaction.setFailedAt(LocalDateTime.now());
            transactionRepository.save(transaction);
            
            // Publish payment failed event
            eventPublisher.publishPaymentFailed(transaction);
            
        } catch (Exception e) {
            logger.error("Error in payment failed handler for transaction {}", transactionId, e);
        }
    }
    
    public void onPaymentCaptured(StateContext<TransactionState, TransactionEvent> context) {
        Long transactionId = context.getExtendedState().get("transactionId", Long.class);
        logger.info("Payment captured for transaction {}", transactionId);
        
        try {
            Transaction transaction = getTransaction(transactionId);
            transaction.setProcessedAt(LocalDateTime.now());
            transactionRepository.save(transaction);
            
            // Publish payment captured event
            eventPublisher.publishPaymentCaptured(transaction);
            
            // Complete the transaction
            context.getStateMachine().sendEvent(TransactionEvent.COMPLETE);
            
        } catch (Exception e) {
            logger.error("Error in payment captured handler for transaction {}", transactionId, e);
            context.getStateMachine().sendEvent(TransactionEvent.FAIL);
        }
    }
    
    public void onTransactionCompleted(StateContext<TransactionState, TransactionEvent> context) {
        Long transactionId = context.getExtendedState().get("transactionId", Long.class);
        logger.info("Transaction {} completed successfully", transactionId);
        
        try {
            Transaction transaction = getTransaction(transactionId);
            transaction.setStatus(com.intelliflow.transaction.entity.TransactionStatus.SUCCESS);
            transaction.setProcessedAt(LocalDateTime.now());
            transactionRepository.save(transaction);
            
            // Publish transaction completed event
            eventPublisher.publishTransactionCompleted(transaction);
            
        } catch (Exception e) {
            logger.error("Error in transaction completed handler for transaction {}", transactionId, e);
        }
    }
    
    public void onTransactionCancelled(StateContext<TransactionState, TransactionEvent> context) {
        Long transactionId = context.getExtendedState().get("transactionId", Long.class);
        logger.info("Transaction {} cancelled", transactionId);
        
        try {
            Transaction transaction = getTransaction(transactionId);
            transaction.setStatus(com.intelliflow.transaction.entity.TransactionStatus.CANCELLED);
            transactionRepository.save(transaction);
            
            // Publish transaction cancelled event
            eventPublisher.publishTransactionCancelled(transaction);
            
        } catch (Exception e) {
            logger.error("Error in transaction cancelled handler for transaction {}", transactionId, e);
        }
    }
    
    public void onRefundInitiated(StateContext<TransactionState, TransactionEvent> context) {
        Long transactionId = context.getExtendedState().get("transactionId", Long.class);
        logger.info("Refund initiated for transaction {}", transactionId);
        
        try {
            Transaction transaction = getTransaction(transactionId);
            
            // Publish refund initiated event
            eventPublisher.publishRefundInitiated(transaction);
            
        } catch (Exception e) {
            logger.error("Error in refund initiated handler for transaction {}", transactionId, e);
        }
    }
    
    public void onRefundCompleted(StateContext<TransactionState, TransactionEvent> context) {
        Long transactionId = context.getExtendedState().get("transactionId", Long.class);
        logger.info("Refund completed for transaction {}", transactionId);
        
        try {
            Transaction transaction = getTransaction(transactionId);
            transaction.setStatus(com.intelliflow.transaction.entity.TransactionStatus.REFUNDED);
            transactionRepository.save(transaction);
            
            // Publish refund completed event
            eventPublisher.publishRefundCompleted(transaction);
            
        } catch (Exception e) {
            logger.error("Error in refund completed handler for transaction {}", transactionId, e);
        }
    }
    
    public void onTransactionDisputed(StateContext<TransactionState, TransactionEvent> context) {
        Long transactionId = context.getExtendedState().get("transactionId", Long.class);
        logger.info("Transaction {} disputed", transactionId);
        
        try {
            Transaction transaction = getTransaction(transactionId);
            transaction.setStatus(com.intelliflow.transaction.entity.TransactionStatus.DISPUTED);
            transactionRepository.save(transaction);
            
            // Publish transaction disputed event
            eventPublisher.publishTransactionDisputed(transaction);
            
        } catch (Exception e) {
            logger.error("Error in transaction disputed handler for transaction {}", transactionId, e);
        }
    }
    
    private Transaction getTransaction(Long transactionId) {
        return transactionRepository.findById(transactionId)
            .orElseThrow(() -> new RuntimeException("Transaction not found: " + transactionId));
    }
}
