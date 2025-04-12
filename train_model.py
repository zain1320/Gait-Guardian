"""
Train the gait classification model on existing data
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from gait_classifier import GaitClassifier

def main():
    # Set paths
    data_dir = "data"
    model_output_dir = "models"
    
    # Create model output directory if it doesn't exist
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Initialize classifier
    classifier = GaitClassifier()
    
    # Load all data
    print("Loading data...")
    try:
        # If data is organized by gait type in subdirectories
        all_data = classifier.load_data(data_dir)
    except Exception as e:
        print(f"Error loading data from subdirectories: {e}")
        print("Trying to load individual CSV files...")
        
        # If data is in individual CSV files
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        all_data = []
        
        for csv_file in csv_files:
            file_path = os.path.join(data_dir, csv_file)
            try:
                df = pd.read_csv(file_path)
                
                # Ensure data has labels
                if 'label' not in df.columns:
                    # Try to infer label from filename
                    label = None
                    for gait_type in ['normal', 'parkinsons', 'cerebral_palsy', 'diabetic_neuropathy']:
                        if gait_type in csv_file.lower():
                            label = gait_type
                            break
                    
                    if label is None:
                        print(f"Warning: Couldn't determine label for {csv_file}, skipping...")
                        continue
                    
                    df['label'] = label
                
                all_data.append(df)
                print(f"Loaded {file_path} with {len(df)} samples")
            except Exception as file_error:
                print(f"Error loading {file_path}: {file_error}")
        
        if not all_data:
            print("No data could be loaded. Exiting.")
            return
            
        all_data = pd.concat(all_data, ignore_index=True)
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessed_data = classifier.preprocess_data(all_data)
    
    # Extract features
    print("Extracting features...")
    features, labels = classifier.extract_features(preprocessed_data)
    
    # Print feature summary
    print(f"Extracted {len(features)} feature sets with {features.shape[1]} features each")
    print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    # Train the model
    print("\nTraining model...")
    classifier.train(features, labels)
    
    # Visualize feature importance
    print("\nGenerating feature importance plot...")
    importance_plot = classifier.visualize_feature_importance()
    importance_plot.savefig(os.path.join(model_output_dir, "feature_importance.png"))
    print(f"Feature importance plot saved to {os.path.join(model_output_dir, 'feature_importance.png')}")
    
    # Save the model
    print("\nSaving model...")
    classifier.save_model(
        os.path.join(model_output_dir, "gait_classifier.pkl"),
        os.path.join(model_output_dir, "gait_scaler.pkl")
    )
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()