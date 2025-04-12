# server.py
import time
import threading
import pandas as pd
import numpy as np
import serial
from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from flask_cors import CORS
from gait_classifier import GaitClassifier

# --- Configuration ---
SERIAL_PORT = 'COM4'       # Update this port to match your system
BAUD_RATE = 115200
WINDOW_SIZE = 100          # Number of samples per sliding window
OVERLAP = 0.5              # 50% overlap between windows
MODEL_PATH = 'models/gait_classifier.pkl'
SCALER_PATH = 'models/gait_scaler.pkl'

# --- Initialize Flask and SocketIO ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

class SerialDataProcessor:
    def __init__(self, port, baud_rate, window_size):
        self.port = port
        self.baud_rate = baud_rate
        self.window_size = window_size
        self.buffer = []
        self.running = False  # Ensure this attribute is defined!
        self.lock = threading.Lock()
        # Load the classifier model using your existing gait_classifier.py
        self.classifier = GaitClassifier.load_model(MODEL_PATH, SCALER_PATH)
        self.serial_conn = None

    def start(self):
        try:
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)  # Allow the connection to stabilize
            print(f"Connected to serial port {self.port}")
        except Exception as e:
            print(f"Error connecting to serial port: {e}")
            return
        self.running = True
        thread = threading.Thread(target=self.read_serial)
        thread.daemon = True
        thread.start()

    def stop(self):
        self.running = False
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Serial connection closed")

    def read_serial(self):
        while self.running:
            try:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    # Skip header or empty lines
                    if not line or line.startswith('==='):
                        continue
                    with self.lock:
                        self.buffer.append(line)
                    # Process the buffer when enough samples have been collected
                    with self.lock:
                        if len(self.buffer) >= self.window_size:
                            self.process_buffer()
                            # Slide the window (50% overlap)
                            step = int(self.window_size * (1 - OVERLAP))
                            self.buffer = self.buffer[step:]
            except Exception as e:
                print(f"Error reading serial data: {e}")
            time.sleep(0.01)

    def process_buffer(self):
        try:
            # Expecting CSV data with at least these 15 columns:
            columns = ['timestamp', 'label', 'accelX', 'accelY', 'accelZ',
                       'gyroX', 'gyroY', 'gyroZ', 'pitch', 'roll',
                       'bigToe', 'medialFore', 'lateralFore', 'heel', 'gaitPhase']
            # Convert each line to a list (using only the first 15 columns)
            data = [line.split(',')[:15] for line in self.buffer]
            df = pd.DataFrame(data, columns=columns)
            # Convert numeric columns to numbers
            for col in columns[2:14]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # Use the classifier's predict method on the windowed data
            prediction, confidence, class_probs = self.classifier.predict(df, window_size=self.window_size)
            # Emit the classification result to all connected SocketIO clients
            socketio.emit('classification_update', {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': class_probs,
                'data_count': len(df)
            })
            print(f"Emitted prediction: {prediction} with confidence {confidence:.2f}")
        except Exception as e:
            print(f"Error processing buffer: {e}")

# Create a global processor instance
processor = SerialDataProcessor(SERIAL_PORT, BAUD_RATE, WINDOW_SIZE)

# --- REST Endpoints ---
@app.route('/api/start', methods=['GET'])
def start_processing():
    processor.start()
    return jsonify({'success': True, 'message': 'Started serial data processing'})

@app.route('/api/stop', methods=['GET'])
def stop_processing():
    processor.stop()
    return jsonify({'success': True, 'message': 'Stopped serial data processing'})

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'running': processor.running,
        'buffer_length': len(processor.buffer)
    })

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
