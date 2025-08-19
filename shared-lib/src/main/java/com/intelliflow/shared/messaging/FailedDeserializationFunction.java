package com.intelliflow.shared.messaging;

import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.common.header.Header;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.support.serializer.FailedDeserializationInfo;

import java.util.function.Function;

/**
 * Function to handle failed deserialization in Kafka consumers
 */
public class FailedDeserializationFunction implements Function<FailedDeserializationInfo, Object> {
    
    private static final Logger logger = LoggerFactory.getLogger(FailedDeserializationFunction.class);
    
    @Override
    public Object apply(FailedDeserializationInfo failedDeserializationInfo) {
        logger.error("Failed to deserialize message from topic: {}, partition: {}, offset: {}",
            failedDeserializationInfo.getTopic(),
            failedDeserializationInfo.getPartition(),
            failedDeserializationInfo.getOffset(),
            failedDeserializationInfo.getException());
        
        // Create a fallback message object
        FailedMessage failedMessage = new FailedMessage();
        failedMessage.setTopic(failedDeserializationInfo.getTopic());
        failedMessage.setPartition(failedDeserializationInfo.getPartition());
        failedMessage.setOffset(failedDeserializationInfo.getOffset());
        failedMessage.setOriginalData(failedDeserializationInfo.getData());
        failedMessage.setErrorMessage(failedDeserializationInfo.getException().getMessage());
        
        return failedMessage;
    }
    
    public static class FailedMessage {
        private String topic;
        private int partition;
        private long offset;
        private byte[] originalData;
        private String errorMessage;
        
        // Getters and Setters
        public String getTopic() { return topic; }
        public void setTopic(String topic) { this.topic = topic; }
        
        public int getPartition() { return partition; }
        public void setPartition(int partition) { this.partition = partition; }
        
        public long getOffset() { return offset; }
        public void setOffset(long offset) { this.offset = offset; }
        
        public byte[] getOriginalData() { return originalData; }
        public void setOriginalData(byte[] originalData) { this.originalData = originalData; }
        
        public String getErrorMessage() { return errorMessage; }
        public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
    }
}
