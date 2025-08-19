package com.intelliflow.transaction.statemachine;

import com.intelliflow.transaction.service.TransactionStateMachineService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.statemachine.config.EnableStateMachine;
import org.springframework.statemachine.config.EnumStateMachineConfigurerAdapter;
import org.springframework.statemachine.config.builders.StateMachineConfigurationConfigurer;
import org.springframework.statemachine.config.builders.StateMachineStateConfigurer;
import org.springframework.statemachine.config.builders.StateMachineTransitionConfigurer;
import org.springframework.statemachine.listener.StateMachineListener;
import org.springframework.statemachine.listener.StateMachineListenerAdapter;
import org.springframework.statemachine.state.State;
import org.springframework.statemachine.transition.Transition;

import java.util.EnumSet;

@Configuration
@EnableStateMachine
public class TransactionStateMachineConfig extends EnumStateMachineConfigurerAdapter<TransactionState, TransactionEvent> {
    
    private static final Logger logger = LoggerFactory.getLogger(TransactionStateMachineConfig.class);
    
    @Autowired
    private TransactionStateMachineService stateMachineService;
    
    @Override
    public void configure(StateMachineConfigurationConfigurer<TransactionState, TransactionEvent> config) throws Exception {
        config
            .withConfiguration()
            .autoStartup(true)
            .listener(stateMachineListener());
    }
    
    @Override
    public void configure(StateMachineStateConfigurer<TransactionState, TransactionEvent> states) throws Exception {
        states
            .withStates()
            .initial(TransactionState.INITIATED)
            .states(EnumSet.allOf(TransactionState.class))
            .end(TransactionState.COMPLETED)
            .end(TransactionState.FAILED)
            .end(TransactionState.CANCELLED)
            .end(TransactionState.REFUND_COMPLETED)
            .end(TransactionState.DISPUTED);
    }
    
    @Override
    public void configure(StateMachineTransitionConfigurer<TransactionState, TransactionEvent> transitions) throws Exception {
        transitions
            // Initial validation flow
            .withExternal()
                .source(TransactionState.INITIATED)
                .target(TransactionState.VALIDATED)
                .event(TransactionEvent.VALIDATION_SUCCESS)
                .action(stateMachineService::onValidationSuccess)
            .and()
            .withExternal()
                .source(TransactionState.INITIATED)
                .target(TransactionState.FAILED)
                .event(TransactionEvent.VALIDATION_FAILED)
                .action(stateMachineService::onValidationFailed)
            .and()
            
            // Fraud checking flow
            .withExternal()
                .source(TransactionState.VALIDATED)
                .target(TransactionState.FRAUD_CHECKING)
                .event(TransactionEvent.FRAUD_CHECK)
                .action(stateMachineService::onFraudCheck)
            .and()
            .withExternal()
                .source(TransactionState.FRAUD_CHECKING)
                .target(TransactionState.FRAUD_APPROVED)
                .event(TransactionEvent.FRAUD_APPROVED)
                .action(stateMachineService::onFraudApproved)
            .and()
            .withExternal()
                .source(TransactionState.FRAUD_CHECKING)
                .target(TransactionState.FRAUD_DECLINED)
                .event(TransactionEvent.FRAUD_DECLINED)
                .action(stateMachineService::onFraudDeclined)
            .and()
            
            // Balance verification flow
            .withExternal()
                .source(TransactionState.FRAUD_APPROVED)
                .target(TransactionState.BALANCE_CHECKING)
                .event(TransactionEvent.BALANCE_CHECK)
                .action(stateMachineService::onBalanceCheck)
            .and()
            .withExternal()
                .source(TransactionState.BALANCE_CHECKING)
                .target(TransactionState.BALANCE_VERIFIED)
                .event(TransactionEvent.BALANCE_VERIFIED)
                .action(stateMachineService::onBalanceVerified)
            .and()
            .withExternal()
                .source(TransactionState.BALANCE_CHECKING)
                .target(TransactionState.BALANCE_INSUFFICIENT)
                .event(TransactionEvent.BALANCE_INSUFFICIENT)
                .action(stateMachineService::onInsufficientBalance)
            .and()
            
            // Payment processing flow
            .withExternal()
                .source(TransactionState.BALANCE_VERIFIED)
                .target(TransactionState.PROCESSING)
                .event(TransactionEvent.PROCESS_PAYMENT)
                .action(stateMachineService::onProcessPayment)
            .and()
            .withExternal()
                .source(TransactionState.PROCESSING)
                .target(TransactionState.PAYMENT_AUTHORIZED)
                .event(TransactionEvent.PAYMENT_AUTHORIZED)
                .action(stateMachineService::onPaymentAuthorized)
            .and()
            .withExternal()
                .source(TransactionState.PROCESSING)
                .target(TransactionState.FAILED)
                .event(TransactionEvent.PAYMENT_FAILED)
                .action(stateMachineService::onPaymentFailed)
            .and()
            
            // Payment capture flow
            .withExternal()
                .source(TransactionState.PAYMENT_AUTHORIZED)
                .target(TransactionState.PAYMENT_CAPTURED)
                .event(TransactionEvent.CAPTURE_PAYMENT)
                .action(stateMachineService::onPaymentCaptured)
            .and()
            .withExternal()
                .source(TransactionState.PAYMENT_CAPTURED)
                .target(TransactionState.COMPLETED)
                .event(TransactionEvent.COMPLETE)
                .action(stateMachineService::onTransactionCompleted)
            .and()
            
            // Cancellation flow
            .withExternal()
                .source(TransactionState.INITIATED)
                .target(TransactionState.CANCELLED)
                .event(TransactionEvent.CANCEL)
                .action(stateMachineService::onTransactionCancelled)
            .and()
            .withExternal()
                .source(TransactionState.VALIDATED)
                .target(TransactionState.CANCELLED)
                .event(TransactionEvent.CANCEL)
                .action(stateMachineService::onTransactionCancelled)
            .and()
            .withExternal()
                .source(TransactionState.FRAUD_CHECKING)
                .target(TransactionState.CANCELLED)
                .event(TransactionEvent.CANCEL)
                .action(stateMachineService::onTransactionCancelled)
            .and()
            
            // Refund flow
            .withExternal()
                .source(TransactionState.COMPLETED)
                .target(TransactionState.REFUND_INITIATED)
                .event(TransactionEvent.REFUND)
                .action(stateMachineService::onRefundInitiated)
            .and()
            .withExternal()
                .source(TransactionState.REFUND_INITIATED)
                .target(TransactionState.REFUND_COMPLETED)
                .event(TransactionEvent.REFUND_COMPLETED)
                .action(stateMachineService::onRefundCompleted)
            .and()
            
            // Dispute flow
            .withExternal()
                .source(TransactionState.COMPLETED)
                .target(TransactionState.DISPUTED)
                .event(TransactionEvent.DISPUTE)
                .action(stateMachineService::onTransactionDisputed);
    }
    
    @Bean
    public StateMachineListener<TransactionState, TransactionEvent> stateMachineListener() {
        return new StateMachineListenerAdapter<TransactionState, TransactionEvent>() {
            @Override
            public void stateChanged(State<TransactionState, TransactionEvent> from, State<TransactionState, TransactionEvent> to) {
                logger.info("Transaction state changed from {} to {}", 
                    from != null ? from.getId() : "null", 
                    to != null ? to.getId() : "null");
            }
            
            @Override
            public void transition(Transition<TransactionState, TransactionEvent> transition) {
                logger.info("Transaction transition: {} -> {} on event {}", 
                    transition.getSource().getId(), 
                    transition.getTarget().getId(), 
                    transition.getTrigger().getEvent());
            }
        };
    }
}
