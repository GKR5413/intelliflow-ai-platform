#!/usr/bin/env python3
"""
IntelliFlow AI Platform - Mock API Server
Provides registration and login endpoints for testing the Web UI
"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class MockAPIHandler(BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        logger.info(f"{self.address_string()} - {format % args}")
    
    def _send_cors_headers(self):
        """Send CORS headers for all responses"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
        self.send_header('Access-Control-Max-Age', '86400')
    
    def _send_json_response(self, data, status_code=200):
        """Send JSON response with proper headers"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def do_OPTIONS(self):
        """Handle preflight requests"""
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/actuator/health':
            response = {
                'status': 'UP',
                'components': {
                    'mock-api': {'status': 'UP'},
                    'database': {'status': 'UP'},
                    'redis': {'status': 'UP'}
                }
            }
            self._send_json_response(response)
            
        elif path == '/api/v1/users/profile':
            # Mock user profile endpoint
            response = {
                'id': 1,
                'username': 'admin',
                'email': 'admin@intelliflow.com',
                'firstName': 'Admin',
                'lastName': 'User',
                'roles': ['ADMIN'],
                'emailVerified': True,
                'createdAt': '2024-01-01T00:00:00Z'
            }
            self._send_json_response(response)
            
        else:
            self._send_json_response({'error': 'Not Found'}, 404)
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
            except json.JSONDecodeError:
                self._send_json_response({'error': 'Invalid JSON'}, 400)
                return
        else:
            data = {}
        
        logger.info(f"POST {path} with data: {data}")
        
        if path == '/api/v1/auth/register':
            # Validate required fields
            required_fields = ['username', 'email', 'password', 'firstName', 'lastName']
            missing_fields = [field for field in required_fields if not data.get(field)]
            
            if missing_fields:
                response = {
                    'success': False,
                    'message': f'Missing required fields: {", ".join(missing_fields)}',
                    'errors': {field: f'{field} is required' for field in missing_fields}
                }
                self._send_json_response(response, 400)
                return
            
            # Simulate successful registration
            response = {
                'success': True,
                'message': 'Registration successful! You can now login.',
                'user': {
                    'id': 1,
                    'username': data.get('username'),
                    'email': data.get('email'),
                    'firstName': data.get('firstName'),
                    'lastName': data.get('lastName'),
                    'roles': ['USER'],
                    'emailVerified': False,
                    'createdAt': '2024-08-19T00:00:00Z'
                }
            }
            self._send_json_response(response)
            
        elif path == '/api/v1/auth/login':
            username_or_email = data.get('usernameOrEmail', '')
            password = data.get('password', '')
            
            if not username_or_email or not password:
                response = {
                    'success': False,
                    'message': 'Username/email and password are required'
                }
                self._send_json_response(response, 400)
                return
            
            # Mock successful login (accept admin/admin or any registered user)
            if (username_or_email == 'admin' and password == 'admin') or (username_or_email == 'testuser' and password == 'test') or len(password) >= 6:
                response = {
                    'accessToken': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIiwibmFtZSI6IkFkbWluIFVzZXIiLCJpYXQiOjE1MTYyMzkwMjJ9.mock-signature',
                    'user': {
                        'id': 1,
                        'username': username_or_email if username_or_email != 'admin' else 'admin',
                        'email': 'admin@intelliflow.com' if username_or_email == 'admin' else f'{username_or_email}@example.com',
                        'firstName': 'Admin' if username_or_email == 'admin' else 'User',
                        'lastName': 'User',
                        'roles': ['ADMIN'] if username_or_email == 'admin' else ['USER'],
                        'emailVerified': True,
                        'createdAt': '2024-01-01T00:00:00Z'
                    }
                }
                self._send_json_response(response)
            else:
                response = {
                    'success': False,
                    'message': 'Invalid credentials'
                }
                self._send_json_response(response, 401)
                
        else:
            self._send_json_response({'error': 'Endpoint not found'}, 404)

def main():
    """Start the mock API server"""
    host = 'localhost'
    port = 8081
    
    try:
        server = HTTPServer((host, port), MockAPIHandler)
        print(f"🚀 IntelliFlow Mock API Server starting...")
        print(f"📍 Server running on http://{host}:{port}")
        print(f"✅ Registration endpoint: POST /api/v1/auth/register")
        print(f"✅ Login endpoint: POST /api/v1/auth/login")
        print(f"✅ Health check: GET /actuator/health")
        print(f"🔧 CORS enabled for all origins")
        print(f"📝 Logs will show all requests")
        print(f"🛑 Press Ctrl+C to stop the server")
        print("-" * 50)
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        server.server_close()
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
