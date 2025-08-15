# Gait Analysis Online - Real-time Neurological Disorder Detection

A wearable sensor system that analyzes walking patterns in real-time to detect early signs of neurological disorders like Parkinson's, diabetic neuropathy, multiple sclerosis, and cerebral palsy.

![The actual hardware setup we built](image1)

## What We Built

This project combines hardware sensing with machine learning to create a comprehensive gait analysis system. Think of it as a "smart insole" that can tell you if someone's walking pattern suggests they might have a neurological condition - all happening in real-time as they walk.

### The Complete System

**Hardware Side:**
- Arduino-based wearable sensor with 4 pressure sensors (big toe, medial forefoot, lateral forefoot, heel)
- 6-axis IMU for motion tracking (accelerometer + gyroscope)
- WiFi connectivity for real-time data streaming
- All packed into a foot-wearable form factor

**Software Side:**
- Real-time data processing and gait phase detection
- Machine learning classifier trained on gait patterns
- Live web dashboard showing predictions and confidence levels
- Multiple communication protocols (WebSocket, MQTT) for flexibility

## The Dashboard in Action

![Live dashboard showing real-time gait analysis](image2)
![Dashboard interface showing prediction results and device status](image3)

The web interface shows everything happening in real-time - connected devices, current predictions with confidence levels, and classification probabilities for each neurological condition. You can see the system continuously analyzing gait patterns and updating predictions as new sensor data streams in from the Arduino. The dashboard provides a comprehensive view of device connectivity status, current gait classification, confidence levels, and probability distributions across different neurological conditions.

## How It Works

### The Science Behind It

Walking is incredibly complex, and neurological disorders affect it in subtle but measurable ways:

- **Parkinson's Disease**: Shuffling gait, reduced arm swing, freezing episodes
- **Diabetic Neuropathy**: Loss of sensation leading to uneven pressure distribution
- **Multiple Sclerosis**: Coordination issues and fatigue-related gait changes
- **Cerebral Palsy**: Spasticity and muscle control problems

Our system captures these differences through:

1. **Pressure Sensing**: How force is distributed across the foot during each step
2. **Motion Analysis**: Pitch, roll, and rotational movements of the foot
3. **Gait Phase Detection**: Identifying heel strike, full contact, toe-off, and swing phases
4. **Temporal Analysis**: Step timing, stance time, and swing time measurements

### The ML Pipeline

```
Raw Sensor Data → Feature Extraction → Classification → Real-time Prediction
     ↓                    ↓                  ↓              ↓
50Hz sampling      Statistical features   Random Forest   Web Dashboard
WiFi streaming     Motion characteristics  Ensemble       Live updates
```

We extract 30+ features from each walking window including:
- Pressure distribution ratios
- Motion smoothness (jerk analysis)
- Gait phase transitions
- Step timing variability
- And more...

## What We Learned (The Real Stuff)

### Technical Challenges That Kept Us Up

**1. Sensor Calibration Was Brutal**
Getting consistent readings from pressure sensors turned out to be way harder than expected. Each sensor had different sensitivity, and foot placement varied between users. We ended up implementing a dynamic calibration system, but honestly, this took forever to get right.

**2. Real-time Processing is Tricky**
Balancing real-time responsiveness with prediction accuracy was like walking a tightrope. Too small a window and predictions were noisy. Too large and the system felt sluggish. We settled on 100-sample windows with 50% overlap after lots of trial and error.

**3. WiFi Connectivity Issues**
Arduino's WiFi kept dropping connections, especially when moving around. We learned to build robust reconnection logic and buffer data locally when the connection was spotty. Also discovered that WebSocket implementation on Arduino is... quirky.

**4. Machine Learning Model Selection**
Started with deep learning (because, you know, it's cool), but Random Forest actually performed better with our feature set. Sometimes simpler is better, and interpretability matters when you're dealing with medical data.

### Unexpected Discoveries

**Gait is Personal**: Even "normal" walking varies dramatically between individuals. What we thought would be clear patterns turned out to be much more nuanced.

**Data Quality Matters More Than Quantity**: We spent way too much time collecting data initially, when we should have focused on getting clean, well-labeled samples first.

**Real-time Constraints Change Everything**: Features that worked great in offline analysis sometimes couldn't be computed fast enough for real-time use.

**User Experience is Critical**: The most accurate system in the world is useless if people won't wear it. We learned to prioritize comfort and ease of use.

## System Architecture

### Communication Flows

We implemented multiple communication methods because we kept running into limitations:

1. **WebSocket Server** (`websocket_server.py`): Direct Arduino-to-dashboard communication
2. **Flask-SocketIO Server** (`app.py`): More robust web interface with database integration  
3. **MQTT Implementation** (`mqqt.py`): For scenarios requiring message queuing

Each has its place - WebSocket for simple real-time demos, Flask-SocketIO for production use, and MQTT for distributed deployments.

## Getting Started

### Hardware Setup

1. **Arduino Requirements:**
   - Arduino Nano 33 IoT (or similar with WiFi)
   - 4x Force Sensitive Resistors (FSRs)
   - Pull-down resistors (10kΩ)
   - Wiring to analog pins A0-A3

2. **Sensor Placement:**
   - Big toe, medial forefoot, lateral forefoot, heel
   - Secure mounting (we used adhesive + fabric)

### Software Setup

```bash
# Install Python dependencies
pip install flask flask-socketio pandas scikit-learn tornado paho-mqtt

# Start the main server
python app.py

# Or try the WebSocket server
python websocket_server.py

# Or the MQTT version
python mqqt.py
```

### Arduino Code

Flash `arduino/arduino.ino` to your device. Update WiFi credentials and server IP address in the code.

### Web Dashboard

Navigate to `http://localhost:8000` to see the real-time dashboard.

## File Structure

```
├── app.py                      # Main Flask-SocketIO server
├── websocket_server.py         # Lightweight WebSocket server  
├── mqqt.py                     # MQTT-based server
├── gait_classifier.py          # Machine learning classifier
├── arduino/
│   └── arduino.ino            # Arduino sensor code
├── models/
│   ├── gait_classifier.pkl    # Trained ML model
│   ├── gait_scaler.pkl        # Feature scaler
│   └── feature_importance.png # Model interpretation
├── templates/
│   └── index.html             # Web dashboard
└── gait_analysis.db           # SQLite database
```

## Performance & Results

Our current model achieves:
- **85% accuracy** on test data
- **~200ms latency** for real-time predictions
- **Successful detection** of major gait pattern differences
- **Robust operation** over WiFi connections

## Future Improvements

Things we'd love to tackle next:

1. **Better Sensor Fusion**: Incorporate more IMU data and maybe add magnetometer
2. **Personalized Baselines**: Adapt the model to individual walking patterns
3. **Edge Computing**: Run inference directly on Arduino for offline operation
4. **Clinical Validation**: Work with medical professionals for proper validation
5. **Mobile App**: Because who doesn't want gait analysis on their phone?

## Contributing

This is a research project, but we're open to collaboration! If you're interested in:
- Medical applications of wearable sensing
- Real-time machine learning systems
- Gait analysis research
- Arduino/IoT development

Feel free to reach out or submit issues/PRs.

## Acknowledgments

This project taught us that building real-world ML systems is 10% algorithms and 90% dealing with messy data, hardware quirks, and user needs. It's been frustrating, exciting, and incredibly educational.

Special thanks to the open-source community for the libraries that made this possible, and to everyone who helped test our prototype (and put up with us strapping sensors to their feet).

*"The best way to predict the future is to build it... even if it involves a lot of debugging Arduino WiFi code."* 😅

