package com.intelliflow.shared.security;

import com.intelliflow.shared.exception.InvalidTokenException;
import io.jsonwebtoken.*;
import io.jsonwebtoken.security.Keys;
import io.jsonwebtoken.security.SignatureException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.stereotype.Component;

import javax.crypto.SecretKey;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;

@Component
public class JwtTokenProvider {
    
    private static final Logger logger = LoggerFactory.getLogger(JwtTokenProvider.class);
    
    @Value("${spring.security.jwt.secret:mySecretKey}")
    private String jwtSecret;
    
    @Value("${spring.security.jwt.expiration:86400000}")
    private long jwtExpirationMs;
    
    @Value("${spring.security.jwt.refresh-expiration:604800000}")
    private long jwtRefreshExpirationMs;
    
    @Value("${spring.application.name:intelliflow}")
    private String issuer;
    
    /**
     * Generate JWT token for authenticated user
     */
    public String generateToken(Authentication authentication) {
        UserPrincipal userPrincipal = (UserPrincipal) authentication.getPrincipal();
        return generateTokenFromUserId(userPrincipal.getId(), userPrincipal.getUsername(), 
                                     userPrincipal.getAuthorities().stream()
                                         .map(GrantedAuthority::getAuthority)
                                         .collect(Collectors.toList()));
    }
    
    /**
     * Generate JWT token from user details
     */
    public String generateTokenFromUserId(Long userId, String username, List<String> authorities) {
        Date expiryDate = Date.from(Instant.now().plus(jwtExpirationMs, ChronoUnit.MILLIS));
        
        return Jwts.builder()
                .setSubject(String.valueOf(userId))
                .claim("username", username)
                .claim("authorities", authorities)
                .setIssuer(issuer)
                .setIssuedAt(new Date())
                .setExpiration(expiryDate)
                .signWith(getSigningKey(), SignatureAlgorithm.HS512)
                .compact();
    }
    
    /**
     * Generate refresh token
     */
    public String generateRefreshToken(Long userId, String username) {
        Date expiryDate = Date.from(Instant.now().plus(jwtRefreshExpirationMs, ChronoUnit.MILLIS));
        
        return Jwts.builder()
                .setSubject(String.valueOf(userId))
                .claim("username", username)
                .claim("type", "refresh")
                .setIssuer(issuer)
                .setIssuedAt(new Date())
                .setExpiration(expiryDate)
                .signWith(getSigningKey(), SignatureAlgorithm.HS512)
                .compact();
    }
    
    /**
     * Get user ID from JWT token
     */
    public Long getUserIdFromToken(String token) {
        Claims claims = parseClaimsFromToken(token);
        return Long.parseLong(claims.getSubject());
    }
    
    /**
     * Get username from JWT token
     */
    public String getUsernameFromToken(String token) {
        Claims claims = parseClaimsFromToken(token);
        return claims.get("username", String.class);
    }
    
    /**
     * Get authorities from JWT token
     */
    @SuppressWarnings("unchecked")
    public List<String> getAuthoritiesFromToken(String token) {
        Claims claims = parseClaimsFromToken(token);
        return claims.get("authorities", List.class);
    }
    
    /**
     * Get token expiration date
     */
    public Date getExpirationDateFromToken(String token) {
        Claims claims = parseClaimsFromToken(token);
        return claims.getExpiration();
    }
    
    /**
     * Check if token is expired
     */
    public boolean isTokenExpired(String token) {
        try {
            Date expiration = getExpirationDateFromToken(token);
            return expiration.before(new Date());
        } catch (Exception e) {
            return true;
        }
    }
    
    /**
     * Validate JWT token
     */
    public boolean validateToken(String token) {
        try {
            parseClaimsFromToken(token);
            return true;
        } catch (SignatureException e) {
            logger.error("Invalid JWT signature: {}", e.getMessage());
        } catch (MalformedJwtException e) {
            logger.error("Invalid JWT token: {}", e.getMessage());
        } catch (ExpiredJwtException e) {
            logger.error("JWT token is expired: {}", e.getMessage());
        } catch (UnsupportedJwtException e) {
            logger.error("JWT token is unsupported: {}", e.getMessage());
        } catch (IllegalArgumentException e) {
            logger.error("JWT claims string is empty: {}", e.getMessage());
        }
        return false;
    }
    
    /**
     * Check if token is a refresh token
     */
    public boolean isRefreshToken(String token) {
        try {
            Claims claims = parseClaimsFromToken(token);
            return "refresh".equals(claims.get("type", String.class));
        } catch (Exception e) {
            return false;
        }
    }
    
    /**
     * Refresh access token using refresh token
     */
    public String refreshToken(String refreshToken) {
        if (!validateToken(refreshToken) || !isRefreshToken(refreshToken)) {
            throw new InvalidTokenException("Invalid refresh token");
        }
        
        Claims claims = parseClaimsFromToken(refreshToken);
        Long userId = Long.parseLong(claims.getSubject());
        String username = claims.get("username", String.class);
        
        // For refresh, we need to get authorities from the database
        // This is a simplified version - in practice, you'd fetch from UserService
        return generateTokenFromUserId(userId, username, List.of("ROLE_USER"));
    }
    
    /**
     * Get remaining validity period in milliseconds
     */
    public long getRemainingValidityMs(String token) {
        Date expiration = getExpirationDateFromToken(token);
        return expiration.getTime() - System.currentTimeMillis();
    }
    
    /**
     * Extract token from Authorization header
     */
    public String extractTokenFromHeader(String authHeader) {
        if (authHeader != null && authHeader.startsWith("Bearer ")) {
            return authHeader.substring(7);
        }
        return null;
    }
    
    /**
     * Generate token for service-to-service communication
     */
    public String generateServiceToken(String serviceName, List<String> permissions) {
        Date expiryDate = Date.from(Instant.now().plus(jwtExpirationMs, ChronoUnit.MILLIS));
        
        return Jwts.builder()
                .setSubject(serviceName)
                .claim("type", "service")
                .claim("permissions", permissions)
                .setIssuer(issuer)
                .setIssuedAt(new Date())
                .setExpiration(expiryDate)
                .signWith(getSigningKey(), SignatureAlgorithm.HS512)
                .compact();
    }
    
    /**
     * Validate service token
     */
    public boolean validateServiceToken(String token, String expectedService) {
        try {
            Claims claims = parseClaimsFromToken(token);
            return "service".equals(claims.get("type", String.class)) &&
                   expectedService.equals(claims.getSubject());
        } catch (Exception e) {
            logger.error("Service token validation failed", e);
            return false;
        }
    }
    
    /**
     * Parse claims from token
     */
    private Claims parseClaimsFromToken(String token) {
        try {
            return Jwts.parserBuilder()
                    .setSigningKey(getSigningKey())
                    .build()
                    .parseClaimsJws(token)
                    .getBody();
        } catch (Exception e) {
            logger.error("Failed to parse JWT token", e);
            throw new InvalidTokenException("Invalid token", e);
        }
    }
    
    /**
     * Get signing key
     */
    private SecretKey getSigningKey() {
        byte[] keyBytes = jwtSecret.getBytes(StandardCharsets.UTF_8);
        return Keys.hmacShaKeyFor(keyBytes);
    }
}
