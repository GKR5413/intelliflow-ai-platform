package com.intelliflow.gateway.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.reactive.EnableWebFluxSecurity;
import org.springframework.security.config.web.server.ServerHttpSecurity;
import org.springframework.security.web.server.SecurityWebFilterChain;
import org.springframework.security.web.server.header.ReferrerPolicyServerHttpHeadersWriter;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.reactive.CorsConfigurationSource;
import org.springframework.web.cors.reactive.UrlBasedCorsConfigurationSource;

import java.time.Duration;
import java.util.Arrays;
import java.util.List;

/**
 * Security configuration for API Gateway
 */
@Configuration
@EnableWebFluxSecurity
public class SecurityConfig {
    
    @Bean
    public SecurityWebFilterChain securityWebFilterChain(ServerHttpSecurity http) {
        return http
            // CSRF Configuration
            .csrf(csrf -> csrf.disable()) // Disabled for API Gateway as it handles stateless authentication
            
            // CORS Configuration
            .cors(cors -> cors.configurationSource(corsConfigurationSource()))
            
            // Security Headers
            .headers(headers -> headers
                .frameOptions(frameOptions -> frameOptions.deny())
                .contentTypeOptions(contentTypeOptions -> contentTypeOptions.and())
                .httpStrictTransportSecurity(hstsConfig -> hstsConfig
                    .maxAgeInSeconds(31536000)
                    .includeSubdomains(true)
                    .preload(true))
                .referrerPolicy(ReferrerPolicyServerHttpHeadersWriter.ReferrerPolicy.STRICT_ORIGIN_WHEN_CROSS_ORIGIN)
                .crossOriginEmbedderPolicy(crossOriginEmbedderPolicyConfig -> 
                    crossOriginEmbedderPolicyConfig.policy(org.springframework.security.web.server.header.CrossOriginEmbedderPolicyServerHttpHeadersWriter.CrossOriginEmbedderPolicy.REQUIRE_CORP))
                .crossOriginOpenerPolicy(crossOriginOpenerPolicyConfig ->
                    crossOriginOpenerPolicyConfig.policy(org.springframework.security.web.server.header.CrossOriginOpenerPolicyServerHttpHeadersWriter.CrossOriginOpenerPolicy.SAME_ORIGIN))
                .crossOriginResourcePolicy(crossOriginResourcePolicyConfig ->
                    crossOriginResourcePolicyConfig.policy(org.springframework.security.web.server.header.CrossOriginResourcePolicyServerHttpHeadersWriter.CrossOriginResourcePolicy.SAME_ORIGIN)))
            
            // Authorization Rules
            .authorizeExchange(exchanges -> exchanges
                // Public endpoints - no authentication required
                .pathMatchers(
                    "/actuator/health",
                    "/actuator/info",
                    "/actuator/metrics",
                    "/actuator/prometheus",
                    "/fallback/**",
                    "/api/v1/auth/login",
                    "/api/v1/auth/register",
                    "/api/v1/auth/verify-email",
                    "/api/v1/auth/forgot-password",
                    "/api/v1/auth/reset-password"
                ).permitAll()
                
                // Admin endpoints - require ADMIN role
                .pathMatchers(
                    "/actuator/**",
                    "/api/v1/admin/**",
                    "/api/v1/*/admin/**"
                ).hasRole("ADMIN")
                
                // Analytics endpoints - require ANALYTICS or ADMIN role
                .pathMatchers(
                    "/api/v1/analytics/**",
                    "/api/v1/*/stats/**"
                ).hasAnyRole("ANALYTICS", "ADMIN")
                
                // Transaction management - require OPERATOR or ADMIN role
                .pathMatchers(
                    "/api/v1/transactions/*/status",
                    "/api/v1/transactions/*/refund",
                    "/api/v1/transactions/*/cancel",
                    "/api/v1/transactions/search"
                ).hasAnyRole("OPERATOR", "ADMIN")
                
                // Fraud management - require FRAUD_ANALYST or ADMIN role
                .pathMatchers(
                    "/api/v1/fraud/**"
                ).hasAnyRole("FRAUD_ANALYST", "ADMIN")
                
                // User management - require appropriate roles
                .pathMatchers(
                    "/api/v1/users/*/admin/**"
                ).hasRole("ADMIN")
                
                // All other requests require authentication
                .anyExchange().authenticated())
            
            // OAuth2 Resource Server Configuration (if using OAuth2)
            .oauth2ResourceServer(oauth2 -> oauth2
                .jwt(jwt -> jwt
                    .jwtDecoder(jwtDecoder())
                    .jwtAuthenticationConverter(jwtAuthenticationConverter())))
            
            // Exception Handling
            .exceptionHandling(exceptions -> exceptions
                .authenticationEntryPoint(customAuthenticationEntryPoint())
                .accessDeniedHandler(customAccessDeniedHandler()))
            
            .build();
    }
    
    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration configuration = new CorsConfiguration();
        
        // Allow specific origins in production, * for development
        configuration.setAllowedOriginPatterns(List.of(
            "https://intelliflow.com",
            "https://*.intelliflow.com",
            "http://localhost:*",
            "https://localhost:*"
        ));
        
        configuration.setAllowedMethods(Arrays.asList(
            "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"
        ));
        
        configuration.setAllowedHeaders(Arrays.asList(
            "Authorization",
            "Content-Type",
            "X-Requested-With",
            "X-User-Id",
            "X-Correlation-ID",
            "X-Request-ID",
            "X-API-Key",
            "Cache-Control",
            "Pragma",
            "Origin",
            "Accept",
            "Accept-Language"
        ));
        
        configuration.setExposedHeaders(Arrays.asList(
            "X-Total-Count",
            "X-Page-Number",
            "X-Page-Size",
            "X-Rate-Limit-Remaining",
            "X-Rate-Limit-Limit",
            "X-Correlation-ID",
            "X-Request-ID"
        ));
        
        configuration.setAllowCredentials(true);
        configuration.setMaxAge(Duration.ofHours(1));
        
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", configuration);
        
        return source;
    }
    
    @Bean
    public org.springframework.security.oauth2.jwt.ReactiveJwtDecoder jwtDecoder() {
        // In production, this would be configured with the actual JWT issuer
        // For now, using a simple implementation
        return org.springframework.security.oauth2.jwt.ReactiveJwtDecoders
            .fromIssuerLocation("https://your-auth-server.com");
    }
    
    @Bean
    public org.springframework.core.convert.converter.Converter<org.springframework.security.oauth2.jwt.Jwt, org.springframework.security.core.Authentication> jwtAuthenticationConverter() {
        org.springframework.security.oauth2.server.resource.authentication.JwtAuthenticationConverter converter = 
            new org.springframework.security.oauth2.server.resource.authentication.JwtAuthenticationConverter();
        
        converter.setJwtGrantedAuthoritiesConverter(jwt -> {
            // Extract authorities from JWT claims
            return org.springframework.security.oauth2.server.resource.authentication.JwtGrantedAuthoritiesConverter
                .DEFAULT_AUTHORITIES_CLAIM_NAME.equals("authorities") ?
                new org.springframework.security.oauth2.server.resource.authentication.JwtGrantedAuthoritiesConverter()
                    .convert(jwt) : List.of();
        });
        
        return new org.springframework.security.oauth2.server.resource.authentication.ReactiveJwtAuthenticationConverterAdapter(converter);
    }
    
    @Bean
    public org.springframework.security.web.server.ServerAuthenticationEntryPoint customAuthenticationEntryPoint() {
        return (exchange, authentication) -> {
            org.springframework.http.server.reactive.ServerHttpResponse response = exchange.getResponse();
            response.setStatusCode(org.springframework.http.HttpStatus.UNAUTHORIZED);
            response.getHeaders().add("Content-Type", "application/json");
            
            String body = """
                {
                    "error": "Authentication Required",
                    "message": "You must be authenticated to access this resource",
                    "status": 401,
                    "timestamp": "%s",
                    "path": "%s"
                }
                """.formatted(
                    java.time.Instant.now().toString(),
                    exchange.getRequest().getPath().value()
                );
            
            org.springframework.core.io.buffer.DataBuffer buffer = response.bufferFactory().wrap(body.getBytes());
            return response.writeWith(reactor.core.publisher.Mono.just(buffer));
        };
    }
    
    @Bean
    public org.springframework.security.web.server.authorization.ServerAccessDeniedHandler customAccessDeniedHandler() {
        return (exchange, denied) -> {
            org.springframework.http.server.reactive.ServerHttpResponse response = exchange.getResponse();
            response.setStatusCode(org.springframework.http.HttpStatus.FORBIDDEN);
            response.getHeaders().add("Content-Type", "application/json");
            
            String body = """
                {
                    "error": "Access Denied",
                    "message": "You don't have permission to access this resource",
                    "status": 403,
                    "timestamp": "%s",
                    "path": "%s"
                }
                """.formatted(
                    java.time.Instant.now().toString(),
                    exchange.getRequest().getPath().value()
                );
            
            org.springframework.core.io.buffer.DataBuffer buffer = response.bufferFactory().wrap(body.getBytes());
            return response.writeWith(reactor.core.publisher.Mono.just(buffer));
        };
    }
}
