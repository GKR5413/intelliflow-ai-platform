package com.intelliflow.shared.health;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.actuator.health.Health;
import org.springframework.boot.actuator.health.HealthIndicator;
import org.springframework.boot.actuator.health.Status;
import org.springframework.data.redis.connection.RedisConnection;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

import javax.sql.DataSource;
import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * Comprehensive health indicator registry for IntelliFlow Platform
 * Provides detailed health checks for all critical dependencies
 */
@Component
public class HealthIndicatorRegistry {

    @Autowired(required = false)
    private DataSource dataSource;

    @Autowired(required = false)
    private RedisConnectionFactory redisConnectionFactory;

    @Autowired(required = false)
    private KafkaTemplate<String, Object> kafkaTemplate;

    /**
     * Database health indicator
     */
    @Component("database")
    public class DatabaseHealthIndicator implements HealthIndicator {

        @Override
        public Health health() {
            if (dataSource == null) {
                return Health.down()
                    .withDetail("reason", "DataSource not configured")
                    .build();
            }

            try {
                Instant start = Instant.now();
                JdbcTemplate jdbcTemplate = new JdbcTemplate(dataSource);
                
                // Test query with timeout
                CompletableFuture<Integer> queryFuture = CompletableFuture.supplyAsync(() -> {
                    return jdbcTemplate.queryForObject("SELECT 1", Integer.class);
                });
                
                Integer result = queryFuture.get(5, TimeUnit.SECONDS);
                Duration responseTime = Duration.between(start, Instant.now());
                
                if (result != null && result == 1) {
                    Health.Builder builder = Health.up()
                        .withDetail("database", "PostgreSQL")
                        .withDetail("responseTime", responseTime.toMillis() + "ms")
                        .withDetail("validationQuery", "SELECT 1");
                    
                    // Add connection pool details if available
                    try {
                        if (dataSource instanceof com.zaxxer.hikari.HikariDataSource) {
                            com.zaxxer.hikari.HikariDataSource hikariDS = 
                                (com.zaxxer.hikari.HikariDataSource) dataSource;
                            builder.withDetail("activeConnections", hikariDS.getHikariPoolMXBean().getActiveConnections())
                                   .withDetail("totalConnections", hikariDS.getHikariPoolMXBean().getTotalConnections())
                                   .withDetail("maxPoolSize", hikariDS.getMaximumPoolSize());
                        }
                    } catch (Exception e) {
                        // Ignore if connection pool details are not available
                    }
                    
                    return builder.build();
                } else {
                    return Health.down()
                        .withDetail("reason", "Validation query failed")
                        .withDetail("responseTime", responseTime.toMillis() + "ms")
                        .build();
                }
                
            } catch (Exception e) {
                return Health.down()
                    .withDetail("reason", "Database connection failed")
                    .withDetail("error", e.getMessage())
                    .withDetail("type", e.getClass().getSimpleName())
                    .build();
            }
        }
    }

    /**
     * Redis health indicator
     */
    @Component("redis")
    public class RedisHealthIndicator implements HealthIndicator {

        @Override
        public Health health() {
            if (redisConnectionFactory == null) {
                return Health.down()
                    .withDetail("reason", "Redis connection factory not configured")
                    .build();
            }

            try {
                Instant start = Instant.now();
                
                CompletableFuture<String> redisFuture = CompletableFuture.supplyAsync(() -> {
                    try (RedisConnection connection = redisConnectionFactory.getConnection()) {
                        return connection.ping();
                    }
                });
                
                String pongResult = redisFuture.get(3, TimeUnit.SECONDS);
                Duration responseTime = Duration.between(start, Instant.now());
                
                if ("PONG".equals(pongResult)) {
                    Health.Builder builder = Health.up()
                        .withDetail("responseTime", responseTime.toMillis() + "ms")
                        .withDetail("command", "PING")
                        .withDetail("result", pongResult);
                    
                    // Add Redis info if available
                    try (RedisConnection connection = redisConnectionFactory.getConnection()) {
                        String info = new String(connection.info("server"));
                        String[] lines = info.split("\r\n");
                        for (String line : lines) {
                            if (line.startsWith("redis_version:")) {
                                builder.withDetail("version", line.split(":")[1]);
                            } else if (line.startsWith("connected_clients:")) {
                                builder.withDetail("connectedClients", line.split(":")[1]);
                            } else if (line.startsWith("used_memory_human:")) {
                                builder.withDetail("usedMemory", line.split(":")[1]);
                            }
                        }
                    } catch (Exception e) {
                        // Ignore if additional info is not available
                    }
                    
                    return builder.build();
                } else {
                    return Health.down()
                        .withDetail("reason", "PING command failed")
                        .withDetail("responseTime", responseTime.toMillis() + "ms")
                        .withDetail("result", pongResult)
                        .build();
                }
                
            } catch (Exception e) {
                return Health.down()
                    .withDetail("reason", "Redis connection failed")
                    .withDetail("error", e.getMessage())
                    .withDetail("type", e.getClass().getSimpleName())
                    .build();
            }
        }
    }

    /**
     * Kafka health indicator
     */
    @Component("kafka")
    public class KafkaHealthIndicator implements HealthIndicator {

        @Override
        public Health health() {
            if (kafkaTemplate == null) {
                return Health.down()
                    .withDetail("reason", "Kafka template not configured")
                    .build();
            }

            try {
                Instant start = Instant.now();
                
                // Test Kafka connectivity by checking cluster metadata
                CompletableFuture<Boolean> kafkaFuture = CompletableFuture.supplyAsync(() -> {
                    try {
                        // Get cluster metadata with timeout
                        kafkaTemplate.getProducerFactory().createProducer()
                            .partitionsFor("health-check-topic");
                        return true;
                    } catch (Exception e) {
                        return false;
                    }
                });
                
                Boolean isHealthy = kafkaFuture.get(5, TimeUnit.SECONDS);
                Duration responseTime = Duration.between(start, Instant.now());
                
                if (isHealthy) {
                    return Health.up()
                        .withDetail("responseTime", responseTime.toMillis() + "ms")
                        .withDetail("bootstrapServers", kafkaTemplate.getProducerFactory()
                            .getConfigurationProperties().get("bootstrap.servers"))
                        .build();
                } else {
                    return Health.down()
                        .withDetail("reason", "Kafka metadata request failed")
                        .withDetail("responseTime", responseTime.toMillis() + "ms")
                        .build();
                }
                
            } catch (Exception e) {
                return Health.down()
                    .withDetail("reason", "Kafka connection failed")
                    .withDetail("error", e.getMessage())
                    .withDetail("type", e.getClass().getSimpleName())
                    .build();
            }
        }
    }

    /**
     * External service health indicator (generic)
     */
    public static class ExternalServiceHealthIndicator implements HealthIndicator {
        
        private final String serviceName;
        private final String healthCheckUrl;
        private final Duration timeout;

        public ExternalServiceHealthIndicator(String serviceName, String healthCheckUrl, Duration timeout) {
            this.serviceName = serviceName;
            this.healthCheckUrl = healthCheckUrl;
            this.timeout = timeout;
        }

        @Override
        public Health health() {
            try {
                Instant start = Instant.now();
                
                // Use Spring's RestTemplate or WebClient for HTTP health checks
                // This is a simplified version - in production, use proper HTTP client
                CompletableFuture<Boolean> httpFuture = CompletableFuture.supplyAsync(() -> {
                    try {
                        java.net.http.HttpClient client = java.net.http.HttpClient.newBuilder()
                            .connectTimeout(timeout)
                            .build();
                        
                        java.net.http.HttpRequest request = java.net.http.HttpRequest.newBuilder()
                            .uri(java.net.URI.create(healthCheckUrl))
                            .timeout(timeout)
                            .GET()
                            .build();
                        
                        java.net.http.HttpResponse<String> response = client.send(request,
                            java.net.http.HttpResponse.BodyHandlers.ofString());
                        
                        return response.statusCode() >= 200 && response.statusCode() < 300;
                    } catch (Exception e) {
                        return false;
                    }
                });
                
                Boolean isHealthy = httpFuture.get(timeout.toSeconds(), TimeUnit.SECONDS);
                Duration responseTime = Duration.between(start, Instant.now());
                
                if (isHealthy) {
                    return Health.up()
                        .withDetail("service", serviceName)
                        .withDetail("url", healthCheckUrl)
                        .withDetail("responseTime", responseTime.toMillis() + "ms")
                        .build();
                } else {
                    return Health.down()
                        .withDetail("service", serviceName)
                        .withDetail("url", healthCheckUrl)
                        .withDetail("reason", "HTTP health check failed")
                        .withDetail("responseTime", responseTime.toMillis() + "ms")
                        .build();
                }
                
            } catch (Exception e) {
                return Health.down()
                    .withDetail("service", serviceName)
                    .withDetail("url", healthCheckUrl)
                    .withDetail("reason", "External service check failed")
                    .withDetail("error", e.getMessage())
                    .withDetail("type", e.getClass().getSimpleName())
                    .build();
            }
        }
    }

    /**
     * Memory health indicator
     */
    @Component("memory")
    public class MemoryHealthIndicator implements HealthIndicator {

        @Override
        public Health health() {
            Runtime runtime = Runtime.getRuntime();
            long maxMemory = runtime.maxMemory();
            long totalMemory = runtime.totalMemory();
            long freeMemory = runtime.freeMemory();
            long usedMemory = totalMemory - freeMemory;
            
            double memoryUsagePercentage = (double) usedMemory / maxMemory * 100;
            
            Health.Builder builder = Health.up()
                .withDetail("maxMemory", formatBytes(maxMemory))
                .withDetail("totalMemory", formatBytes(totalMemory))
                .withDetail("usedMemory", formatBytes(usedMemory))
                .withDetail("freeMemory", formatBytes(freeMemory))
                .withDetail("usagePercentage", String.format("%.2f%%", memoryUsagePercentage));
            
            // Determine health status based on memory usage
            if (memoryUsagePercentage > 90) {
                builder.status(Status.DOWN)
                       .withDetail("reason", "Memory usage critically high");
            } else if (memoryUsagePercentage > 80) {
                builder.status("WARNING")
                       .withDetail("reason", "Memory usage high");
            }
            
            return builder.build();
        }
        
        private String formatBytes(long bytes) {
            if (bytes < 1024) return bytes + " B";
            int exp = (int) (Math.log(bytes) / Math.log(1024));
            String pre = "KMGTPE".charAt(exp - 1) + "";
            return String.format("%.1f %sB", bytes / Math.pow(1024, exp), pre);
        }
    }

    /**
     * Disk space health indicator
     */
    @Component("diskSpace")
    public class DiskSpaceHealthIndicator implements HealthIndicator {

        @Override
        public Health health() {
            try {
                java.io.File root = new java.io.File("/");
                long totalSpace = root.getTotalSpace();
                long freeSpace = root.getFreeSpace();
                long usedSpace = totalSpace - freeSpace;
                
                double usagePercentage = (double) usedSpace / totalSpace * 100;
                
                Health.Builder builder = Health.up()
                    .withDetail("totalSpace", formatBytes(totalSpace))
                    .withDetail("freeSpace", formatBytes(freeSpace))
                    .withDetail("usedSpace", formatBytes(usedSpace))
                    .withDetail("usagePercentage", String.format("%.2f%%", usagePercentage))
                    .withDetail("path", root.getAbsolutePath());
                
                // Determine health status based on disk usage
                if (usagePercentage > 95) {
                    builder.status(Status.DOWN)
                           .withDetail("reason", "Disk space critically low");
                } else if (usagePercentage > 85) {
                    builder.status("WARNING")
                           .withDetail("reason", "Disk space low");
                }
                
                return builder.build();
                
            } catch (Exception e) {
                return Health.down()
                    .withDetail("reason", "Failed to check disk space")
                    .withDetail("error", e.getMessage())
                    .build();
            }
        }
        
        private String formatBytes(long bytes) {
            if (bytes < 1024) return bytes + " B";
            int exp = (int) (Math.log(bytes) / Math.log(1024));
            String pre = "KMGTPE".charAt(exp - 1) + "";
            return String.format("%.1f %sB", bytes / Math.pow(1024, exp), pre);
        }
    }
}
