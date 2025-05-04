#!/usr/bin/env python3
"""
Dummy sender that simulates a sensor device HTTP server
providing gas, temperature and humidity data.
"""
import http.server
import socketserver
import json
import random
import threading
import time
import socket
import os
import signal
import sys

# Configuration
PORT = 8080  # Using 8080 as we can't bind to port 80 without root
HOST = "localhost"  # Use localhost for testing
UPDATE_INTERVAL = 1  # seconds between sensor data updates

# Global variables
running = True
sensor_data = {
    "gas": 0,
    "temp": 0,
    "humidity": 0
}

# Track current server instance
current_server = None

class SensorHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler that returns sensor data"""
    
    def do_GET(self):
        """Handle GET requests by returning current sensor data"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')  # CORS header
        self.end_headers()
        
        # Send the current sensor data as JSON
        self.wfile.write(json.dumps(sensor_data).encode())
    
    # Make the server quiet
    def log_message(self, format, *args):
        return

def update_sensor_data():
    """Continuously update sensor data with random values"""
    global sensor_data
    
    while running:
        # Generate random sensor data
        sensor_data = {
            "gas": random.uniform(100, 500),            # Gas resistance in ohms
            "temp": random.uniform(20, 30),             # Temperature in Celsius
            "humidity": random.uniform(30, 70)          # Humidity percentage
        }
        
        time.sleep(UPDATE_INTERVAL)

class ReuseAddressServer(socketserver.TCPServer):
    """TCP Server with SO_REUSEADDR option set"""
    allow_reuse_address = True

def free_port():
    """Attempt to free the port if it's in use"""
    global current_server
    
    # If we have a server running, stop it first
    if current_server:
        try:
            stop_server(current_server)
        except Exception as e:
            print(f"Error stopping existing server: {e}")
    
    # Try to forcibly free the port by finding and killing any process using it
    try:
        # This works on Unix-based systems
        if sys.platform != 'win32':
            os.system(f"lsof -ti:{PORT} | xargs kill -9 2>/dev/null")
        else:
            # For Windows
            os.system(f"for /f \"tokens=5\" %a in ('netstat -aon ^| findstr :{PORT}') do taskkill /f /pid %a")
    except Exception as e:
        print(f"Warning: Could not forcibly free port: {e}")
    
    # Add a small delay to ensure port is released
    time.sleep(1)

def start_server():
    """Start the HTTP server in a separate thread"""
    global current_server
    
    # Try to free the port first
    free_port()
    
    try:
        # Create and start the server with reuse address option
        server = ReuseAddressServer((HOST, PORT), SensorHandler)
        current_server = server
        
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        # Start the sensor data update thread
        update_thread = threading.Thread(target=update_sensor_data)
        update_thread.daemon = True
        update_thread.start()
        
        print(f"Dummy sensor server running at http://{HOST}:{PORT}/")
        return server
    except OSError as e:
        print(f"Error starting server: {e}")
        print("Could not start server even after trying to free the port.")
        return None

def stop_server(server):
    """Stop the HTTP server"""
    global running, current_server
    if server:
        running = False
        try:
            server.shutdown()
            server.server_close()
            current_server = None
            print("Sensor server stopped")
        except Exception as e:
            print(f"Error stopping server: {e}")

if __name__ == "__main__":
    try:
        server = start_server()
        if server:
            print("Press Ctrl+C to stop the server")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_server(server)
