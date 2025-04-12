"""
Gait Disorder Classification System
Uses existing gait data to detect neurological disorders
"""

import numpy as np
import pandas as pd
import os
import glob
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class GaitClassifier:
    def __init__(self):
        """Initialize the gait classifier"""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_data(self, data_directory):
        """
        Load data from organized directory structure
        
        Parameters:
        -----------
        data_directory : str
            Path to the main data directory containing subdirectories for each gait type
            
        Returns:
        --------
        pd.DataFrame
            Combined DataFrame with all data and labels
        """
        all_data = []
        
        # Scan through each subdirectory (gait type)
        for gait_type in os.listdir(data_directory):
            gait_path = os.path.join(data_directory, gait_type)
            
            # Skip if not a directory
            if not os.path.isdir(gait_path):
                continue
                
            # Find all CSV files in this gait type directory
            csv_files = glob.glob(os.path.join(gait_path, "*.csv"))
            
            for file_path in csv_files:
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    
                    # Ensure it has the label column, or add it
                    if 'label' not in df.columns:
                        df['label'] = gait_type
                        
                    # Append to our collection
                    all_data.append(df)
                    print(f"Loaded {file_path} with {len(df)} samples")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        # Combine all data
        if not all_data:
            raise ValueError("No data files found or loaded successfully")
            
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Combined dataset has {len(combined_data)} samples with labels: {combined_data['label'].unique()}")
        
        return combined_data
    
    def preprocess_data(self, data):
        """
        Preprocess raw sensor data for feature extraction
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw sensor data with columns for timestamp, label, and sensor readings
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed data ready for feature extraction
        """
        # Handle missing values
        if data.isnull().sum().sum() > 0:
            # Fill missing values with forward fill, then backward fill
            data = data.fillna(method='ffill').fillna(method='bfill')
            
        # Remove step timing metrics if they exist and are all zeros
        timing_cols = ['stepTime', 'stanceTime', 'swingTime']
        if all(col in data.columns for col in timing_cols):
            if (data[timing_cols] == 0).all().all():
                data = data.drop(columns=timing_cols)
                print("Removed timing columns (all zeros)")
        
        # Ensure datatypes are correct
        numeric_cols = ['accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ', 
                      'pitch', 'roll', 'bigToe', 'medialFore', 'lateralFore', 'heel']
                       
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data
    
    def extract_features(self, data, window_size=100, overlap=0.5, min_samples=20):
        """
        Extract features from preprocessed data using sliding windows
        
        Parameters:
        -----------
        data : pd.DataFrame
            Preprocessed sensor data
        window_size : int
            Number of samples per window
        overlap : float
            Fraction of overlap between consecutive windows
        min_samples : int
            Minimum number of samples required for a valid window
            
        Returns:
        --------
        features_df : pd.DataFrame
            Extracted features for each window
        labels : np.array
            Label for each window
        """
        step = int(window_size * (1 - overlap))
        
        # Create windows
        windows = []
        labels = []
        
        # If data is very small, use it as a single window
        if len(data) < min_samples:
            print(f"Warning: Only {len(data)} samples available, which is less than min_samples={min_samples}")
            windows.append(data)
            labels.append(data['label'].iloc[0] if data['label'].nunique() == 1 else "unknown")
        else:
            # Create overlapping windows
            for i in range(0, len(data) - window_size + 1, step):
                window = data.iloc[i:i+window_size]
                
                # Only use the window if it has a single label type
                if window['label'].nunique() == 1:
                    windows.append(window)
                    labels.append(window['label'].iloc[0])
        
        if not windows:
            raise ValueError("No valid windows found. Check your data or window parameters.")
        
        # Extract features from each window
        features = []
        
        for window in windows:
            window_features = self._extract_window_features(window)
            features.append(window_features)
            
        # Create feature dataframe
        features_df = pd.DataFrame(features)
        self.feature_names = features_df.columns.tolist()
        
        return features_df, np.array(labels)
    
    def _extract_window_features(self, window):
        """Extract features from a single window of sensor data"""
        features = {}
        
        # Pressure distribution features
        features['big_toe_mean'] = window['bigToe'].mean()
        features['medial_fore_mean'] = window['medialFore'].mean()
        features['lateral_fore_mean'] = window['lateralFore'].mean()
        features['heel_mean'] = window['heel'].mean()
        
        features['big_toe_max'] = window['bigToe'].max()
        features['heel_max'] = window['heel'].max()
        
        # Calculate toe-to-heel ratio
        features['big_toe_to_heel_ratio'] = features['big_toe_mean'] / (features['heel_mean'] if features['heel_mean'] > 0 else 1)
        
        # Calculate lateral-to-medial ratio
        features['lateral_to_medial_ratio'] = features['lateral_fore_mean'] / (features['medial_fore_mean'] if features['medial_fore_mean'] > 0 else 1)
        
        # Pressure variability
        features['big_toe_std'] = window['bigToe'].std()
        features['heel_std'] = window['heel'].std()
        features['big_toe_cv'] = features['big_toe_std'] / (features['big_toe_mean'] if features['big_toe_mean'] > 0 else 1)
        features['heel_cv'] = features['heel_std'] / (features['heel_mean'] if features['heel_mean'] > 0 else 1)
        
        # Total pressure and asymmetry
        features['total_pressure_mean'] = window[['bigToe', 'medialFore', 'lateralFore', 'heel']].sum(axis=1).mean()
        features['pressure_asymmetry'] = abs(features['lateral_fore_mean'] - features['medial_fore_mean']) / (features['lateral_fore_mean'] + features['medial_fore_mean'] if (features['lateral_fore_mean'] + features['medial_fore_mean']) > 0 else 1)
        
        # Motion features
        features['pitch_range'] = window['pitch'].max() - window['pitch'].min()
        features['roll_range'] = window['roll'].max() - window['roll'].min()
        features['pitch_std'] = window['pitch'].std()
        features['roll_std'] = window['roll'].std()
        
        # Gyroscope features
        window['gyro_magnitude'] = np.sqrt(window['gyroX']**2 + window['gyroY']**2 + window['gyroZ']**2)
        features['gyro_magnitude_mean'] = window['gyro_magnitude'].mean()
        features['gyro_magnitude_max'] = window['gyro_magnitude'].max()
        features['gyro_magnitude_std'] = window['gyro_magnitude'].std()
        
        # Acceleration features
        window['accel_magnitude'] = np.sqrt(window['accelX']**2 + window['accelY']**2 + window['accelZ']**2)
        features['accel_magnitude_mean'] = window['accel_magnitude'].mean()
        features['accel_magnitude_std'] = window['accel_magnitude'].std()
        
        # Calculate jerk (derivative of acceleration) for smoothness
        window['accel_magnitude_diff'] = window['accel_magnitude'].diff().abs()
        features['jerk_mean'] = window['accel_magnitude_diff'].mean()
        features['jerk_max'] = window['accel_magnitude_diff'].max()
        
        # Gait phase distribution
        phase_counts = window['gaitPhase'].value_counts(normalize=True)
        for phase in ['HEEL_STRIKE', 'FULL_CONTACT', 'TOE_OFF', 'SWING', 'PARTIAL_CONTACT']:
            features[f'{phase.lower()}_percent'] = phase_counts.get(phase, 0)
        
        # Phase transitions
        transitions = (window['gaitPhase'] != window['gaitPhase'].shift()).sum()
        features['transition_rate'] = transitions / len(window)
        
        # Double support estimation (both heel and toe pressure simultaneously)
        double_support = ((window['heel'] > 100) & ((window['bigToe'] > 100) | (window['medialFore'] > 100) | (window['lateralFore'] > 100))).mean()
        features['double_support_percent'] = double_support
        
        # Heel strike presence
        features['heel_strike_detected'] = 1 if (window['gaitPhase'] == 'HEEL_STRIKE').any() and features['heel_max'] > 150 else 0
        
        return features
    
    def train(self, features, labels):
        """
        Train the classifier model
        
        Parameters:
        -----------
        features : pd.DataFrame
            Extracted features
        labels : np.array
            Target labels
            
        Returns:
        --------
        self : GaitClassifier
            Trained classifier
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            class_weight='balanced',
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_predictions = self.model.predict(X_val)
        val_accuracy = np.mean(val_predictions == y_val)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, val_predictions))
        
        # Print confusion matrix
        cm = confusion_matrix(y_val, val_predictions)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, labels, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.4f}")
        
        return self
    
    def predict(self, data, window_size=100):
        """
        Predict gait patterns from new sensor data
        
        Parameters:
        -----------
        data : pd.DataFrame
            New sensor data
        window_size : int
            Size of the window to use for prediction
            
        Returns:
        --------
        prediction : str
            Predicted gait pattern
        confidence : float
            Confidence score (probability of the prediction)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preprocess data
        data = self.preprocess_data(data)
        
        # Use the last window_size samples
        if len(data) >= window_size:
            window = data.iloc[-window_size:]
        else:
            # Pad with the first row if not enough data
            padding = pd.concat([data.iloc[:1]] * (window_size - len(data)))
            window = pd.concat([padding, data])
        
        # Extract features
        window_features = self._extract_window_features(window)
        features_df = pd.DataFrame([window_features])
        
        # Ensure all expected features are present
        for feature in self.feature_names:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        # Reorder columns to match training data
        features_df = features_df[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(features_df)
        
        # Get prediction and probabilities
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Get confidence (probability of predicted class)
        confidence = probabilities[self.model.classes_.tolist().index(prediction)]
        
        # Get all class probabilities
        class_probs = {cls: prob for cls, prob in zip(self.model.classes_, probabilities)}
        
        return prediction, confidence, class_probs
    
    def save_model(self, model_path, scaler_path):
        """Save the trained model and scaler to disk"""
        if self.model is None:
            raise ValueError("No trained model to save.")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        feature_info = {
            'feature_names': self.feature_names
        }
        joblib.dump(feature_info, model_path.replace('.pkl', '_features.pkl'))
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    @classmethod
    def load_model(cls, model_path, scaler_path):
        """Load a trained model from disk"""
        classifier = cls()
        classifier.model = joblib.load(model_path)
        classifier.scaler = joblib.load(scaler_path)
        
        feature_info = joblib.load(model_path.replace('.pkl', '_features.pkl'))
        classifier.feature_names = feature_info['feature_names']
        
        return classifier
        
    def visualize_feature_importance(self, top_n=15):
        """Visualize the most important features"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Get feature importance
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.title('Feature Importance')
        plt.barh(range(len(indices)), importance[indices], align='center')
        plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        
        return plt
        
    def visualize_confusion_matrix(self, X_test, y_test):
        """Visualize the confusion matrix"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Get predictions
        y_pred = self.model.predict(X_test)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.model.classes_, yticklabels=self.model.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Confusion Matrix')
        plt.tight_layout()
        
        return plt