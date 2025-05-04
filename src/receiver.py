#!/usr/bin/env python3
"""
Minimal sensor data receiver for HTTP endpoint at 192.168.4.1
"""
import requests
import time
import os
import signal
import sys
from datetime import datetime
import threading

# Configuration
SENSOR_URL = "http://localhost:8080/"  # "http://192.168.4.1/" private ip address for microcontroller access point
DATA_DIR = "../data"
SAVE_INTERVAL = 1  # seconds
SENSOR_ID = "sensor1"  # Default sensor ID
running = True
_thread = None

def signal_handler(sig, frame):
    """Handle termination signals"""
    global running
    print("\nShutting down...")
    running = False
    sys.exit(0)

def get_data():
    """Fetch sensor data from HTTP endpoint"""
    try:
        response = requests.get(SENSOR_URL, timeout=3)
        if response.status_code == 200:
            data = response.json()
            return {
                'timestamp': datetime.now().isoformat(),
                'gas_resistance': data.get('gas', 0),
                'temperature': data.get('temp', 0),
                'humidity': data.get('humidity', 0),
                'sensor_id': SENSOR_ID
            }
    except Exception as e:
        print(f"Error: {e}")
    return None

def save_data(data):
    """Save sensor data to file"""
    if not data:
        return
        
    os.makedirs(DATA_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    filepath = os.path.join(DATA_DIR, f"sensor-{date_str}.txt")
    
    line = (f"{data['timestamp']},{data['sensor_id']},"
            f"{data['gas_resistance']},{data['temperature']},{data['humidity']}\n")
    
    with open(filepath, 'a') as f:
        f.write(line)
    
    print(f"Data saved at {datetime.now().strftime('%H:%M:%S')}")

def loop():
    """Main receiver loop"""
    print(f"Receiver started. Interval: {SAVE_INTERVAL}s, Dir: {DATA_DIR}")
    
    while running:
        data = get_data()
        if data:
            save_data(data)
        time.sleep(SAVE_INTERVAL)

def start():
    """Start in background thread"""
    global _thread, running
    running = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if not (_thread and _thread.is_alive()):
        _thread = threading.Thread(target=loop, daemon=True)
        _thread.start()
        print("Receiver running in background")
    return _thread

def stop():
    """Stop background thread"""
    global running
    running = False
    print("Stopping receiver...")

def is_running():
    """Check if running"""
    return _thread is not None and _thread.is_alive()

# When run as script
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--background":
        start()
        print("Running in background. Import to control.")
    else:
        loop()