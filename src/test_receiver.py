#!/usr/bin/env python3
"""
Test script demonstrating receiver and sender integration.
"""
import receiver
import sender
import time
import os

def main():
    # Override receiver URL to point to our local server
    receiver.SENSOR_URL = f"http://{sender.HOST}:{sender.PORT}/"
    
    print("=== Sensor Data Collection Test ===")
    
    # Start the dummy sensor server
    print("Starting dummy sensor server...")
    server = sender.start_server()
    if not server:
        print("Failed to start server. Exiting.")
        return
    
    # Start the receiver in background
    print("Starting sensor receiver in background...")
    receiver.start()
    
    # Run for a short while collecting data
    collection_time = 10  # seconds
    print(f"Collecting data for {collection_time} seconds...")
    
    for i in range(collection_time):
        # Display the current sensor values
        print(f"Current sensor data: {sender.sensor_data}")
        time.sleep(1)
    
    # Stop receiver and server
    receiver.stop()
    sender.stop_server(server)
    
    # Show collected data
    print("\nData collection complete!")
    
    # Find and display the saved data file
    data_files = os.listdir(receiver.DATA_DIR)
    data_files = [f for f in data_files if f.startswith("sensor-")]
    
    if data_files:
        latest_file = os.path.join(receiver.DATA_DIR, data_files[-1])
        print(f"Data saved to: {latest_file}")
        print("\nSample of collected data:")
        with open(latest_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-5:]:  # Show the last 5 entries
                print(f"  {line.strip()}")
    else:
        print("No data files found.")

if __name__ == "__main__":
    main()
