"""
Web-based Gait Analysis Dashboard with Enhanced Debugging
Connects to TCP socket server and performs real-time classification
"""

import os
import time
import json
import socket
import threading
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from gait_classifier import GaitClassifier

# Comprehensive Logging Configuration
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('gait_analyzer_debug.log', mode='w'),  # Overwrite log each time
        logging.StreamHandler()  # Also print to console
    ]
)

# Configuration
TCP_SERVER_HOST = "0.0.0.0"  # Listen on all interfaces
TCP_SERVER_PORT = 8080
WINDOW_SIZE = 100     # Number of samples to analyze at once
OVERLAP = 0.5         # Window overlap (0.5 = 50%)
MODEL_PATH = 'models/gait_classifier.pkl'
SCALER_PATH = 'models/gait_scaler.pkl'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

class DetailedGaitDataProcessor:
    def __init__(self, window_size=WINDOW_SIZE, overlap=OVERLAP, exclude_normal=False):
        """Initialize the gait data processor with extensive logging"""
        self.window_size = window_size
        self.step_size = int(window_size * (1 - overlap))
        self.buffer = []
        self.running = False
        self.current_prediction = None
        self.current_confidence = 0
        self.current_probs = {}
        self.classifier = None
        self.lock = threading.Lock()
        self.last_predictions = []
        self.data_count = 0
        self.exclude_normal = exclude_normal
        self.error_count = 0
        self.total_received_lines = 0
        
        # Create logs directory if not exists
        os.makedirs('logs', exist_ok=True)
        os.makedirs('received_data', exist_ok=True)
        
        # Detailed logging file
        self.log_file = os.path.join('logs', f'detailed_gait_log_{time.strftime("%Y%m%d_%H%M%S")}.csv')
        with open(self.log_file, 'w') as f:
            f.write('timestamp,raw_data,processed,prediction,confidence\n')

    def log_detailed_data(self, raw_data, processed=False, prediction=None, confidence=None):
        """Log detailed data processing information"""
        try:
            with open(self.log_file, 'a') as f:
                log_entry = f"{time.time()},{raw_data},{processed},{prediction or 'N/A'},{confidence or 'N/A'}\n"
                f.write(log_entry)
        except Exception as e:
            logging.error(f"Error logging detailed data: {e}")

    def load_model(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        """Load the trained classifier model with extensive error handling"""
        try:
            logging.info(f"Attempting to load model from {model_path}")
            self.classifier = GaitClassifier.load_model(model_path, scaler_path)
            logging.info("Model loaded successfully")
            return True
        except Exception as e:
            logging.error(f"Critical error loading model: {e}")
            return False

    def process_data_stream(self, data_buffer):
        """
        Process a stream of incoming data with robust error handling
        
        Args:
            data_buffer (str): Raw data buffer from socket
        """
        try:
            # Enhanced Debugging
            print(f"RAW DATA BUFFER RECEIVED: {data_buffer}")
            logging.info(f"RAW DATA BUFFER RECEIVED: {data_buffer}")
            
            # Split buffer into lines
            lines = data_buffer.split('\n')
            
            print(f"Number of lines received: {len(lines)}")
            logging.info(f"Number of lines received: {len(lines)}")
            
            processed_data = []
            for line in lines:
                line = line.strip()
                
                # Skip empty or header lines
                if not line or line.startswith('timestamp') or len(line) < 10:
                    print(f"Skipping line: {line}")
                    logging.debug(f"Skipping line: {line}")
                    continue
                    
                # Split the line
                values = line.split(',')
                
                print(f"Processing line: {line}")
                print(f"Number of values: {len(values)}")
                logging.info(f"Processing line: {line}")
                logging.info(f"Number of values: {len(values)}")
                
                # Validate minimum number of columns
                if len(values) < 15:
                    print(f"Insufficient columns in line: {line}")
                    logging.warning(f"Insufficient columns in line: {line}")
                    continue
                
                try:
                    # Convert numeric columns
                    numeric_values = [float(val) for val in values[2:14]]
                    
                    # Reconstruct line
                    processed_line = (
                        [values[0], values[1]] +  # timestamp and label
                        numeric_values +  # numeric sensor data
                        [values[14]]  # gait phase
                    )
                    
                    processed_data.append(processed_line)
                    self.total_received_lines += 1
                    
                    # Log successful processing
                    print(f"Processed line successfully: {processed_line}")
                    logging.info(f"Processed line successfully: {processed_line}")
                
                except ValueError as ve:
                    print(f"Numeric conversion error: {ve} in line {line}")
                    logging.error(f"Numeric conversion error: {ve} in line {line}")
            
            # Extensive Debugging for Processed Data
            print(f"Total processed lines: {len(processed_data)}")
            logging.info(f"Total processed lines: {len(processed_data)}")
            
            # Convert to DataFrame if we have data
            if processed_data:
                columns = [
                    'timestamp', 'label', 'accelX', 'accelY', 'accelZ', 
                    'gyroX', 'gyroY', 'gyroZ', 'pitch', 'roll',
                    'bigToe', 'medialFore', 'lateralFore', 'heel', 'gaitPhase'
                ]
                
                df = pd.DataFrame(processed_data, columns=columns)
                
                # Convert numeric columns
                numeric_cols = columns[2:-1]
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                
                # More Debugging Information
                print(f"DataFrame shape: {df.shape}")
                print(f"DataFrame columns: {df.columns}")
                print(f"DataFrame sample:\n{df.head()}")
                logging.info(f"DataFrame shape: {df.shape}")
                logging.info(f"DataFrame columns: {df.columns}")
                logging.info(f"DataFrame sample:\n{df.head()}")
                
                # Save raw data to file
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join('received_data', f'gait_data_{timestamp}.csv')
                df.to_csv(filename, index=False)
                print(f"Saved raw data to {filename}")
                logging.info(f"Saved raw data to {filename}")
                
                # Analyze if we have enough data
                if len(df) >= self.window_size:
                    print(f"Analyzing window of {len(df)} samples")
                    logging.info(f"Analyzing window of {len(df)} samples")
                    self.analyze_windowed_data(df)
                else:
                    print(f"Insufficient data for window. Only {len(df)} samples.")
                    logging.warning(f"Insufficient data for window. Only {len(df)} samples.")
        
        except Exception as e:
            self.error_count += 1
            print(f"CRITICAL Data stream processing error: {e}")
            logging.error(f"CRITICAL Data stream processing error: {e}")
            # Print full traceback
            import traceback
            print(traceback.format_exc())
            logging.error(traceback.format_exc())

    def analyze_windowed_data(self, data_frame):
        """
        Analyze data using a sliding window approach
        
        Args:
            data_frame (pd.DataFrame): Sensor data DataFrame
        """
        try:
            # Extensive Logging
            logging.info("Starting window analysis")
            logging.info(f"Window size: {len(data_frame)}")
            
            # Use sliding windows
            for i in range(0, len(data_frame) - self.window_size + 1, self.step_size):
                window = data_frame.iloc[i:i+self.window_size]
                
                logging.info(f"Current window index: {i}")
                logging.info(f"Window samples: {len(window)}")
                
                # Predict using window
                try:
                    prediction, confidence, class_probs = self.classifier.predict(window)
                    
                    logging.info(f"Prediction: {prediction}")
                    logging.info(f"Confidence: {confidence}")
                    logging.info(f"Class Probabilities: {class_probs}")
                    
                    # Update prediction history
                    self.last_predictions.append(prediction)
                    if len(self.last_predictions) > 5:
                        self.last_predictions.pop(0)
                    
                    # Smooth predictions
                    smoothed_prediction = self.smooth_predictions()
                    
                    # Thread-safe update of results
                    with self.lock:
                        self.current_prediction = smoothed_prediction
                        self.current_confidence = confidence
                        self.current_probs = class_probs
                        self.data_count += len(window)
                    
                    # Emit results via SocketIO
                    socketio.emit('classification_update', {
                        'prediction': smoothed_prediction,
                        'confidence': float(confidence),
                        'probabilities': {k: float(v) for k, v in class_probs.items()},
                        'data_count': self.data_count,
                        'total_received_lines': self.total_received_lines,
                        'error_count': self.error_count
                    })
                
                except Exception as predict_error:
                    logging.error(f"Prediction error: {predict_error}")
                    import traceback
                    logging.error(traceback.format_exc())
        
        except Exception as e:
            self.error_count += 1
            logging.error(f"Window analysis error: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def smooth_predictions(self, window_size=3):
        """Smooth predictions using most frequent method"""
        if not self.last_predictions:
            return None
            
        # Count occurrences of each prediction
        counts = {}
        for pred in self.last_predictions[-min(window_size, len(self.last_predictions)):]:
            counts[pred] = counts.get(pred, 0) + 1
        
        # Return the most common prediction
        return max(counts.items(), key=lambda x: x[1])[0] if counts else None

    def start_tcp_listener(self, host=TCP_SERVER_HOST, port=TCP_SERVER_PORT):
        """
        Start a TCP listener to receive continuous data stream
        
        Args:
            host (str): Server host
            port (int): Server port
        """
        def tcp_listener():
            try:
                # Create TCP socket
                server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((host, port))
                server.listen(5)
                
                print(f"TCP Listener started on {host}:{port}")
                logging.info(f"TCP Listener started on {host}:{port}")
                print("Waiting for connections...")
                logging.info("Waiting for connections...")
                
                while self.running:
                    # Accept client connection
                    client, address = server.accept()
                    print(f"Client connected from {address}")
                    logging.info(f"Client connected from {address}")
                    
                    # Data buffer
                    data_buffer = ""
                    
                    try:
                        while self.running:
                            # Receive data
                            chunk = client.recv(4096).decode('utf-8', errors='ignore')
                            
                            if not chunk:
                                print("No data received. Connection closed.")
                                logging.warning("No data received. Connection closed.")
                                break
                            
                            # Accumulate data
                            data_buffer += chunk
                            
                            # Split into lines
                            lines = data_buffer.split('\n')
                            
                            # Process complete lines
                            for line in lines[:-1]:
                                print(f"Received line: {line}")
                                logging.debug(f"Received line: {line}")
                                self.process_data_stream(line + '\n')
                            
                            # Keep last potentially incomplete line in buffer
                            data_buffer = lines[-1]
                    
                    except Exception as client_error:
                        print(f"Client connection error: {client_error}")
                        logging.error(f"Client connection error: {client_error}")
                        import traceback
                        print(traceback.format_exc())
                        logging.error(traceback.format_exc())
                    
                    finally:
                        client.close()
                
            except Exception as e:
                print(f"TCP Listener fatal error: {e}")
                logging.critical(f"TCP Listener fatal error: {e}")
                import traceback
                print(traceback.format_exc())
                logging.critical(traceback.format_exc())
            
            finally:
                server.close()
                self.running = False

        # Reset state
        self.running = True
        self.data_count = 0
        self.total_received_lines = 0
        self.error_count = 0
        
        # Start listener in a thread
        listener_thread = threading.Thread(target=tcp_listener, daemon=True)
        listener_thread.start()
        
        return True

    def stop(self):
        """Stop the data processing"""
        self.running = False
        logging.info("Data processing stopped")

    def get_current_results(self):
        """Get current classification results"""
        with self.lock:
            return {
                'prediction': self.current_prediction,
                'confidence': self.current_confidence,
                'probabilities': self.current_probs,
                'data_count': self.data_count,
                'total_received_lines': self.total_received_lines,
                'error_count': self.error_count
            }

# Global processor instance
processor = None

@app.route('/')
def index():
    """Serve main dashboard page"""
    return render_template('index.html')

@app.route('/api/start', methods=['GET'])
def start_processing():
    """Start gait analysis"""
    global processor
    
    # Create processor
    processor = DetailedGaitDataProcessor()
    
    # Load model
    if not processor.load_model():
        return jsonify({'success': False, 'message': 'Failed to load model'})
    
    # Start TCP listener
    if processor.start_tcp_listener():
        return jsonify({'success': True, 'message': 'Started gait analysis'})
    else:
        return jsonify({'success': False, 'message': 'Failed to start TCP listener'})

@app.route('/api/stop', methods=['GET'])
def stop_processing():
    """Stop gait analysis"""
    global processor
    
    if processor:
        processor.stop()
        return jsonify({'success': True, 'message': 'Stopped gait analysis'})
    
    return jsonify({'success': False, 'message': 'Not running'})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current processing status"""
    global processor
    
    if not processor:
        return jsonify({
            'status': 'not_started',
            'data_count': 0,
            'total_received_lines': 0,
            'error_count': 0
        })
    
    results = processor.get_current_results()
    results['status'] = 'running' if processor.running else 'stopped'
    
    return jsonify(results)

@app.route('/api/list_files', methods=['GET'])
def list_files():
    """List available data files"""
    files = []
    
    # Look in received_data directory
    if os.path.exists('received_data'):
        for filename in os.listdir('received_data'):
            if filename.endswith('.csv'):
                filepath = os.path.join('received_data', filename)
                files.append({
                    'name': filename,
                    'path': filepath,
                    'size': os.path.getsize(filepath),
                    'date': datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
                })
    
    return jsonify(files)

@app.route('/api/analyze_file', methods=['GET'])
def analyze_file():
    """Analyze a specific data file"""
    global processor
    
    file_path = request.args.get('file')
    if not file_path:
        return jsonify({'success': False, 'message': 'No file specified'})
    
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'message': f'File not found: {file_path}'})
    
    # Ensure processor exists and is ready
    if not processor:
        processor = DetailedGaitDataProcessor()
        if not processor.load_model():
            return jsonify({'success': False, 'message': 'Failed to load model'})
    
    try:
        # Read the file
        df = pd.read_csv(file_path)
        
        # Process the entire file
        processor.process_data_stream(df.to_csv(index=False))
        
        return jsonify({
            'success': True,
            'message': 'File analyzed successfully',
            'rows': len(df),
            'columns': list(df.columns)
        })
    except Exception as e:
        logging.error(f"Error analyzing file: {e}")
        return jsonify({'success': False, 'message': str(e)})

@socketio.on('connect')
def handle_connect():
    """Handle SocketIO client connection"""
    logging.info('SocketIO client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle SocketIO client disconnection"""
    logging.info('SocketIO client disconnected')

def main():
    """Main application entry point"""
    logging.info("Starting Gait Analysis Web Dashboard")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()