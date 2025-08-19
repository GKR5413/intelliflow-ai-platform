package com.intelliflow.shared.messaging;

import com.fasterxml.jackson.annotation.JsonFormat;

import java.time.LocalDateTime;
import java.util.Map;
import java.util.UUID;

/**
 * Base event message for Kafka messaging
 */
public class EventMessage {
    
    private String eventId;
    private String eventType;
    private String source;
    private String correlationId;
    
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime timestamp;
    
    private Object payload;
    private Map<String, Object> metadata;
    
    public EventMessage() {
        this.eventId = UUID.randomUUID().toString();
        this.timestamp = LocalDateTime.now();
    }
    
    public EventMessage(String eventType, String source, Object payload) {
        this();
        this.eventType = eventType;
        this.source = source;
        this.payload = payload;
    }
    
    public EventMessage(String eventType, String source, Object payload, String correlationId) {
        this(eventType, source, payload);
        this.correlationId = correlationId;
    }
    
    // Getters and Setters
    public String getEventId() { return eventId; }
    public void setEventId(String eventId) { this.eventId = eventId; }
    
    public String getEventType() { return eventType; }
    public void setEventType(String eventType) { this.eventType = eventType; }
    
    public String getSource() { return source; }
    public void setSource(String source) { this.source = source; }
    
    public String getCorrelationId() { return correlationId; }
    public void setCorrelationId(String correlationId) { this.correlationId = correlationId; }
    
    public LocalDateTime getTimestamp() { return timestamp; }
    public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }
    
    public Object getPayload() { return payload; }
    public void setPayload(Object payload) { this.payload = payload; }
    
    public Map<String, Object> getMetadata() { return metadata; }
    public void setMetadata(Map<String, Object> metadata) { this.metadata = metadata; }
    
    @Override
    public String toString() {
        return "EventMessage{" +
               "eventId='" + eventId + '\'' +
               ", eventType='" + eventType + '\'' +
               ", source='" + source + '\'' +
               ", correlationId='" + correlationId + '\'' +
               ", timestamp=" + timestamp +
               '}';
    }
}
