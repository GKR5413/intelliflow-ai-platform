package com.intelliflow.user.security;

import com.intelliflow.user.entity.User;
import com.intelliflow.user.repository.UserRoleRepository;
import io.jsonwebtoken.*;
import io.jsonwebtoken.security.Keys;
import io.jsonwebtoken.security.SignatureException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import javax.crypto.SecretKey;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Date;
import java.util.List;

/**
 * JWT token provider for user service
 */
@Component
public class JwtTokenProvider {
    
    private static final Logger logger = LoggerFactory.getLogger(JwtTokenProvider.class);
    
    private final UserRoleRepository userRoleRepository;
    
    @Value("${spring.security.jwt.secret:mySecretKey}")
    private String jwtSecret;
    
    @Value("${spring.security.jwt.expiration:86400000}")
    private long jwtExpirationMs;
    
    @Autowired
    public JwtTokenProvider(UserRoleRepository userRoleRepository) {
        this.userRoleRepository = userRoleRepository;
    }
    
    /**
     * Generate JWT token for user
     */
    public String generateToken(User user) {
        List<String> roles = userRoleRepository.findRoleNamesByUserId(user.getId());
        return generateTokenFromUserId(user.getId(), user.getUsername(), roles);
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
                .setIssuer("intelliflow")
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
     * Parse claims from token
     */
    private Claims parseClaimsFromToken(String token) {
        return Jwts.parser()
                .setSigningKey(getSigningKey())
                .build()
                .parseClaimsJws(token)
                .getBody();
    }
    
    /**
     * Get signing key
     */
    private SecretKey getSigningKey() {
        byte[] keyBytes = jwtSecret.getBytes(StandardCharsets.UTF_8);
        return Keys.hmacShaKeyFor(keyBytes);
    }
}
