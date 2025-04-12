"""
Real-time Gait Analysis System with Tkinter UI
Connects to Arduino and performs continuous gait classification
"""

import time
import pandas as pd
import numpy as np
import threading
import os
import serial.tools.list_ports
import tkinter as tk
from tkinter import ttk
from gait_classifier import GaitClassifier

# Configuration
SERIAL_PORT = 'COM4'  # Change to your Arduino port
BAUD_RATE = 115200
WINDOW_SIZE = 100     # Number of samples to analyze at once
OVERLAP = 0.5         # Window overlap (0.5 = 50%)
MODEL_PATH = 'models/gait_classifier.pkl'
SCALER_PATH = 'models/gait_scaler.pkl'

class RealtimeGaitAnalyzer:
    def __init__(self, port=SERIAL_PORT, baud_rate=BAUD_RATE, 
                 window_size=WINDOW_SIZE, overlap=OVERLAP):
        """Initialize the real-time gait analyzer"""
        self.port = port
        self.baud_rate = baud_rate
        self.window_size = window_size
        self.step_size = int(window_size * (1 - overlap))
        self.buffer = []
        self.arduino = None
        self.running = False
        self.current_prediction = None
        self.current_confidence = 0
        self.current_probs = {}
        self.classifier = None
        self.lock = threading.Lock()  # Thread synchronization
        self.last_predictions = []    # Store recent predictions for smoothing
        self.data_count = 0

    def load_model(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        """Load the trained classifier model"""
        try:
            print(f"Loading model from {model_path}...")
            self.classifier = GaitClassifier.load_model(model_path, scaler_path)
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def connect(self):
        """Connect to the Arduino"""
        try:
            self.arduino = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)  # Allow time for connection to stabilize
            print(f"Connected to Arduino on {self.port}")
            return True
        except Exception as e:
            print(f"Error connecting to Arduino: {e}")
            return False

    def disconnect(self):
        """Disconnect from the Arduino"""
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("Disconnected from Arduino")

    def analyze_data(self, data_frame):
        """Analyze data using the trained model"""
        try:
            prediction, confidence, class_probs = self.classifier.predict(data_frame)
            
            # Add to prediction history for smoothing
            self.last_predictions.append(prediction)
            if len(self.last_predictions) > 5:  # Keep only last 5 predictions
                self.last_predictions.pop(0)
            
            # Apply smoothing
            smoothed_prediction = self.smooth_predictions()
            
            with self.lock:
                self.current_prediction = smoothed_prediction
                self.current_confidence = confidence
                self.current_probs = class_probs
                
            # Log results
            self.log_results(smoothed_prediction, confidence, time.time())
            
            return smoothed_prediction, confidence, class_probs
        except Exception as e:
            print(f"Error during analysis: {e}")
            return None, 0, {}

    def smooth_predictions(self, window_size=5):
        """Apply a rolling window to smooth predictions"""
        if not self.last_predictions:
            return None
            
        # Count occurrences of each prediction
        counts = {}
        for pred in self.last_predictions[-min(window_size, len(self.last_predictions)):]:
            counts[pred] = counts.get(pred, 0) + 1
        
        # Return the most common prediction
        return max(counts.items(), key=lambda x: x[1])[0]

    def log_results(self, prediction, confidence, timestamp):
        """Log classification results to a file"""
        os.makedirs('logs', exist_ok=True)
        log_file = os.path.join('logs', f'gait_log_{time.strftime("%Y%m%d")}.csv')
        
        # Create file with header if it doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write('timestamp,prediction,confidence\n')
        
        # Append result
        with open(log_file, 'a') as f:
            f.write(f"{timestamp},{prediction},{confidence:.4f}\n")

    def process_data_thread(self):
        """Process data in a separate thread"""
        while self.running:
            if self.arduino and self.arduino.in_waiting > 0:
                try:
                    line = self.arduino.readline().decode('utf-8').strip()
                    if line and ',' in line and not line.startswith('==='):
                        values = line.split(',')
                        if len(values) >= 15:  # Ensure we have enough values
                            self.buffer.append(values)
                            self.data_count += 1
                            
                            # When buffer is full, analyze the data
                            if len(self.buffer) >= self.window_size:
                                # Create DataFrame with the right column names
                                columns = [
                                    'timestamp', 'label', 'accelX', 'accelY', 'accelZ', 
                                    'gyroX', 'gyroY', 'gyroZ', 'pitch', 'roll',
                                    'bigToe', 'medialFore', 'lateralFore', 'heel', 'gaitPhase'
                                ]
                                
                                # There might be additional columns for step timing metrics
                                if len(values) > 15:
                                    columns.extend(['stepTime', 'stanceTime', 'swingTime'])
                                
                                # Take only the columns we have data for
                                df = pd.DataFrame(self.buffer, columns=columns[:len(values)])
                                
                                # Convert columns to numeric
                                for col in df.columns[2:]:  # Skip timestamp and label
                                    df[col] = pd.to_numeric(df[col], errors='coerce')
                                
                                # Analyze the data
                                self.analyze_data(df)
                                
                                # Slide the window
                                self.buffer = self.buffer[self.step_size:]
                except Exception as e:
                    print(f"Error processing data: {e}")
            
            time.sleep(0.01)  # Small delay to prevent CPU overuse

    def start(self):
        """Start the real-time analysis"""
        if not self.classifier:
            print("Model not loaded. Please load the model first.")
            return False
        
        if not self.arduino:
            print("Not connected to Arduino. Please connect first.")
            return False
        
        self.running = True
        self.buffer = []
        self.last_predictions = []
        self.data_count = 0
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_data_thread)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        print("Real-time analysis started")
        return True

    def stop(self):
        """Stop the real-time analysis"""
        self.running = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=1.0)
        print("Real-time analysis stopped")

    def get_current_results(self):
        """Get the current classification results"""
        with self.lock:
            return {
                'prediction': self.current_prediction,
                'confidence': self.current_confidence,
                'probabilities': self.current_probs,
                'data_count': self.data_count
            }

def list_serial_ports():
    """List available serial ports"""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found")
        return []
    
    print("Available serial ports:")
    for i, port in enumerate(ports):
        print(f"{i+1}. {port.device} - {port.description}")
    
    return ports

class GaitAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Gait Analyzer")
        self.root.geometry("800x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.analyzer = None
        self.running = False
        self.update_id = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Create frame for port selection
        self.port_frame = ttk.LabelFrame(self.root, text="Arduino Connection")
        self.port_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Port selection
        ttk.Label(self.port_frame, text="Serial Port:").grid(row=0, column=0, padx=5, pady=5)
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(self.port_frame, textvariable=self.port_var)
        self.port_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # Refresh and connect buttons
        ttk.Button(self.port_frame, text="Refresh", command=self.refresh_ports).grid(row=0, column=2, padx=5, pady=5)
        self.connect_button = ttk.Button(self.port_frame, text="Connect", command=self.toggle_connection)
        self.connect_button.grid(row=0, column=3, padx=5, pady=5)
        
        # Status display
        ttk.Label(self.port_frame, text="Status:").grid(row=1, column=0, padx=5, pady=5)
        self.status_var = tk.StringVar(value="Disconnected")
        ttk.Label(self.port_frame, textvariable=self.status_var).grid(row=1, column=1, padx=5, pady=5)
        
        # Data count display
        ttk.Label(self.port_frame, text="Data points:").grid(row=1, column=2, padx=5, pady=5)
        self.data_count_var = tk.StringVar(value="0")
        ttk.Label(self.port_frame, textvariable=self.data_count_var).grid(row=1, column=3, padx=5, pady=5)
        
        # Create frame for classification results
        self.results_frame = ttk.LabelFrame(self.root, text="Classification Results")
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Main prediction display
        self.prediction_var = tk.StringVar(value="No data")
        self.prediction_label = ttk.Label(self.results_frame, textvariable=self.prediction_var, font=("Arial", 20))
        self.prediction_label.pack(pady=20)
        
        # Confidence display
        self.confidence_var = tk.StringVar(value="Confidence: 0.00")
        self.confidence_label = ttk.Label(self.results_frame, textvariable=self.confidence_var, font=("Arial", 16))
        self.confidence_label.pack(pady=10)
        
        # Progress bars for probabilities
        self.prob_frame = ttk.Frame(self.results_frame)
        self.prob_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.gait_types = ['parkinsons', 'cerebral_palsy', 'diabetic_neuropathy', 'ms']
        self.prob_bars = {}
        self.prob_vars = {}
        
        for i, gait_type in enumerate(self.gait_types):
            # Label
            ttk.Label(self.prob_frame, text=gait_type).grid(row=i, column=0, padx=5, pady=5, sticky=tk.W)
            
            # Progress bar
            progress = ttk.Progressbar(self.prob_frame, length=400, mode='determinate')
            progress.grid(row=i, column=1, padx=5, pady=5)
            self.prob_bars[gait_type] = progress
            
            # Value label
            value_var = tk.StringVar(value="0.00")
            ttk.Label(self.prob_frame, textvariable=value_var, width=5).grid(row=i, column=2, padx=5, pady=5)
            self.prob_vars[gait_type] = value_var
        
        # Refresh ports on startup
        self.refresh_ports()
    
    def refresh_ports(self):
        """Refresh the list of available ports"""
        ports = list_serial_ports()
        port_names = [port.device for port in ports]
        self.port_combo['values'] = port_names
        
        if port_names and not self.port_var.get():
            self.port_var.set(port_names[0])
    
    def toggle_connection(self):
        """Connect to or disconnect from Arduino"""
        if not self.analyzer or not self.running:
            # Connect
            port = self.port_var.get()
            if not port:
                self.status_var.set("No port selected")
                return
            
            # Create analyzer
            self.analyzer = RealtimeGaitAnalyzer(port=port)
            
            # Load model
            if not self.analyzer.load_model():
                self.status_var.set("Failed to load model")
                self.analyzer = None
                return
            
            # Connect to Arduino
            if not self.analyzer.connect():
                self.status_var.set(f"Failed to connect to {port}")
                self.analyzer = None
                return
            
            # Start analysis
            if not self.analyzer.start():
                self.status_var.set("Failed to start analysis")
                self.analyzer.disconnect()
                self.analyzer = None
                return
            
            # Start UI updates
            self.running = True
            self.status_var.set(f"Connected to {port}")
            self.connect_button.config(text="Disconnect")
            self.update_ui()
        else:
            # Disconnect
            self.running = False
            self.analyzer.stop()
            self.analyzer.disconnect()
            self.analyzer = None
            self.status_var.set("Disconnected")
            self.connect_button.config(text="Connect")
            
            if self.update_id is not None:
                self.root.after_cancel(self.update_id)
                self.update_id = None
    
    def update_ui(self):
        """Update the UI with current results"""
        if self.analyzer and self.running:
            results = self.analyzer.get_current_results()
            prediction = results['prediction']
            confidence = results['confidence']
            probs = results['probabilities']
            data_count = results['data_count']
            
            # Update data count
            self.data_count_var.set(str(data_count))
            
            if prediction:
                # Update prediction and confidence
                self.prediction_var.set(f"Detected: {prediction}")
                self.confidence_var.set(f"Confidence: {confidence:.2f}")
                
                # Update probability bars
                for gait_type in self.gait_types:
                    prob = probs.get(gait_type, 0)
                    self.prob_bars[gait_type]['value'] = prob * 100  # Convert to percentage
                    self.prob_vars[gait_type].set(f"{prob:.2f}")
                    
                    # Highlight the predicted class
                    if gait_type == prediction:
                        self.prob_bars[gait_type]['style'] = 'green.Horizontal.TProgressbar'
                    else:
                        self.prob_bars[gait_type]['style'] = 'Horizontal.TProgressbar'
            
            # Schedule next update
            self.update_id = self.root.after(70, self.update_ui)
    
    def on_closing(self):
        """Handle window closing"""
        if self.analyzer and self.running:
            self.analyzer.stop()
            self.analyzer.disconnect()
        self.root.destroy()

def main():
    """Main function to run the application"""
    # Create custom styles for progress bars
    root = tk.Tk()
    style = ttk.Style()
    style.configure('green.Horizontal.TProgressbar', background='green')
    
    # Create app
    app = GaitAnalyzerApp(root)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()