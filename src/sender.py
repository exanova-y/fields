#!/usr/bin/env python3
"""
Dummy data sender that simulates BME688 sensor.
Creates a simple HTTP server at http://localhost:8080/ that serves simulated sensor data.
"""
import socket
import random
import time
import threading
import http.server
import socketserver
from urllib.parse import parse_qs, urlparse
import json
import numpy as np

# Configuration
HOST = "localhost"  # Interface to bind to
PORT = 8080          # Port to listen on

# Global server reference that can be closed if needed
current_server = None

class ReuseAddressServer(socketserver.TCPServer):
    """TCP Server that reuses port if it's in TIME_WAIT state"""
    allow_reuse_address = True

class SensorHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler that simulates a sensor device"""
    
    def do_GET(self):
        """Handle GET requests with simulated sensor data"""
        # Parse URL to check if JSON endpoint was requested
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # Generate random sensor values
        temp = 20 + 5 * random.random() + 5 * np.sin(time.time() / 10)
        humidity = 20 + 30 * random.random()
        gas = 1000 + 5000 * random.random()
        
        # Build sensor data dictionary
        sensor_data = {
            "temp": round(temp, 2),
            "humidity": round(humidity, 2),
            "gas": round(gas, 2)
        }
        
        # Check if JSON endpoint was requested (default for programmatic access)
        if path == "/" or path == "/json":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')  # Allow CORS
            self.end_headers()
            
            # Return JSON data
            self.wfile.write(json.dumps(sensor_data).encode('utf-8'))
            
            # For debugging
            print(f"Sent JSON data - Temp: {temp:.2f}°C, Humidity: {humidity:.2f}%, Gas: {gas:.2f}Ω")
            
        elif path == "/html":
            # HTML view for browser testing
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Format HTML response
            html = f"""
            <html>
            <head>
                <title>BME688 Sensor Dashboard</title>
                <meta http-equiv="refresh" content="2">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    .data {{ font-size: 1.2em; }}
                    .value {{ font-weight: bold; color: #0066cc; }}
                </style>
            </head>
            <body>
                <h1>BME688 Sensor Dashboard</h1>
                <div class="data">
                    Temperature: <span class="value">{temp:.2f} °C</span>
                </div>
                <div class="data">
                    Humidity: <span class="value">{humidity:.2f} %</span>
                </div>
                <div class="data">
                    Gas Resistance: <span class="value">{gas:.2f} Ω</span>
                </div>
                <p>
                    <a href="/">View JSON Data</a> | 
                    <a href="/config">View Configuration</a>
                </p>
                <p><small>Data updates automatically every 2 seconds</small></p>
            </body>
            </html>
            """
            
            # Convert to bytes and send
            self.wfile.write(html.encode('utf-8'))
            
        elif path == "/config":
            # Configuration view
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <html>
            <head>
                <title>BME688 Configuration</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #333; }
                    table { border-collapse: collapse; width: 100%; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <h1>BME688 Configuration</h1>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Endpoints</td>
                        <td>/ (JSON), /html (HTML), /config (This page)</td>
                    </tr>
                    <tr>
                        <td>Update Interval</td>
                        <td>~1 second</td>
                    </tr>
                    <tr>
                        <td>Sensor Type</td>
                        <td>BME688 (Simulated)</td>
                    </tr>
                </table>
                <p><a href="/html">Back to Dashboard</a></p>
            </body>
            </html>
            """
            
            self.wfile.write(html.encode('utf-8'))
        else:
            # Not found
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"404 Not Found")
    
    def log_message(self, format, *args):
        """Silence server logs"""
        return

def free_port():
    """Free the port if it's in use"""
    global current_server
    
    if current_server:
        print("Stopping existing server...")
        current_server.shutdown()
        current_server.server_close()
        current_server = None
        # Give it a moment to fully close
        time.sleep(0.5)

def start_server():
    """Start the HTTP server"""
    free_port()
    try:
        server = ReuseAddressServer((HOST, PORT), SensorHandler)
        global current_server
        current_server = server
        
        # Start server in a thread
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        print(f"Dummy sensor server running at http://{HOST}:{PORT}/")
        print("Press Ctrl+C to stop the server")
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            server.shutdown()
            server.server_close()
            print("\nServer stopped")
    
    except OSError as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    start_server()
