#!/usr/bin/env python3
"""
Test script for CBC Claim Autoencoder (Model 1)
Verifies model loading and anomaly detection on synthetic data
"""

import os
import sys
import numpy as np
import tensorflow as tf
import joblib
import json

# Paths
BASE_DIR = "/home/comphortinoe/comphortinoe/dev/resultshield/rsl-engine"
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Model files
MODEL_PATH = os.path.join(MODEL_DIR, "cbc_model1_claim_autoencoder.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "cbc_model1_scaler.pkl")
META_PATH = os.path.join(MODEL_DIR, "cbc_model1_claim_autoencoder_meta.json")

# Feature names (must match training)
LAB_FEATURES = ['HGB', 'HCT', 'MCV', 'MCHC', 'NEU', 'LYM', 'EOS', 'BAS', 'MON', 'PLT']
ALL_FEATURES = ['age', 'sex_encoded'] + LAB_FEATURES
N_FEATURES = len(ALL_FEATURES)

def create_synthetic_test_data(n_samples=5):
    """Create synthetic test samples with normal and anomalous patterns"""
    np.random.seed(42)
    
    # Normal ranges
    age_range = (20, 80)
    sex_values = [0, 1]  # 0=Female, 1=Male
    
    normal_ranges = {
        'HGB': (12.0, 16.0),
        'HCT': (36.0, 48.0),
        'MCV': (80.0, 100.0),
        'MCHC': (31.0, 36.0),
        'NEU': (40.0, 70.0),
        'LYM': (20.0, 45.0),
        'EOS': (1.0, 5.0),
        'BAS': (0.5, 1.5),
        'MON': (2.0, 10.0),
        'PLT': (150.0, 400.0)
    }
    
    # Anomalous patterns (extreme values)
    anomalous_ranges = {
        'HGB': (4.0, 8.0),      # Severe anemia
        'PLT': (500.0, 800.0),   # Thrombocytosis
        'NEU': (20.0, 30.0)      # High neutrophils (infection)
    }
    
    test_samples = []
    
    for i in range(n_samples):
        sample = []
        
        # Age
        age = np.random.uniform(age_range[0], age_range[1])
        sample.append(age)
        
        # Sex
        sex = np.random.choice(sex_values)
        sample.append(sex)
        
        # Lab values
        if i == 0:
            # Normal sample
            for feature in LAB_FEATURES:
                low, high = normal_ranges[feature]
                value = np.random.uniform(low, high)
                sample.append(value)
        elif i == 1:
            # Anomalous: Very low HGB (anemia)
            for feature in LAB_FEATURES:
                if feature == 'HGB':
                    value = np.random.uniform(anomalous_ranges['HGB'][0], anomalous_ranges['HGB'][1])
                else:
                    low, high = normal_ranges[feature]
                    value = np.random.uniform(low, high)
                sample.append(value)
        elif i == 2:
            # Anomalous: Very high PLT
            for feature in LAB_FEATURES:
                if feature == 'PLT':
                    value = np.random.uniform(anomalous_ranges['PLT'][0], anomalous_ranges['PLT'][1])
                else:
                    low, high = normal_ranges[feature]
                    value = np.random.uniform(low, high)
                sample.append(value)
        elif i == 3:
            # Anomalous: Very high NEU (infection pattern)
            for feature in LAB_FEATURES:
                if feature == 'NEU':
                    value = np.random.uniform(anomalous_ranges['NEU'][0], anomalous_ranges['NEU'][1])
                else:
                    low, high = normal_ranges[feature]
                    value = np.random.uniform(low, high)
                sample.append(value)
        else:
            # Combined anomalies (low HGB + high PLT)
            for feature in LAB_FEATURES:
                if feature == 'HGB':
                    value = np.random.uniform(anomalous_ranges['HGB'][0], anomalous_ranges['HGB'][1])
                elif feature == 'PLT':
                    value = np.random.uniform(anomalous_ranges['PLT'][0], anomalous_ranges['PLT'][1])
                else:
                    low, high = normal_ranges[feature]
                    value = np.random.uniform(low, high)
                sample.append(value)
        
        test_samples.append(sample)
    
    return np.array(test_samples)

def run_test():
    """Load autoencoder and run anomaly detection on synthetic data"""
    
    print("="*60)
    print("CBC Claim Autoencoder - Test Script")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return False
    
    try:
        # 1. Load model
        print("\n📦 Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
        
        # 2. Load scaler
        print("\n📦 Loading scaler...")
        scaler = joblib.load(SCALER_PATH)
        print("✅ Scaler loaded successfully")
        
        # 3. Load metadata
        print("\n📦 Loading metadata...")
        with open(META_PATH, 'r') as f:
            metadata = json.load(f)
        
        threshold = metadata.get('threshold', 0.1)
        print(f"✅ Metadata loaded")
        print(f"   Input dimension: {metadata.get('input_dim', N_FEATURES)}")
        print(f"   Anomaly threshold: {threshold:.6f}")
        print(f"   Best validation loss: {metadata.get('best_val_loss', 'N/A')}")
        
        # 4. Create synthetic test data
        print("\n🔧 Creating synthetic test data...")
        test_data = create_synthetic_test_data(n_samples=5)
        print(f"✅ Test data shape: {test_data.shape}")
        
        # Display raw test data
        print("\n📊 Raw Test Data:")
        for i in range(len(test_data)):
            age, sex = test_data[i][0], test_data[i][1]
            sex_label = "Male" if sex == 1 else "Female"
            hgb = test_data[i][2]
            plt = test_data[i][-1]
            neu = test_data[i][5]  # NEU is index 5 (after age, sex, HGB, HCT, MCV, MCHC)
            print(f"  Sample {i+1}: Age={age:.0f}, Sex={sex_label}, HGB={hgb:.1f}, NEU={neu:.1f}, PLT={plt:.1f}")
        
        # 5. Scale test data
        test_data_scaled = scaler.transform(test_data)
        
        # 6. Run inference
        print("\n🚀 Running anomaly detection...")
        reconstructed = model.predict(test_data_scaled, verbose=0)
        
        # 7. Calculate reconstruction errors
        mse_per_sample = np.mean(np.square(test_data_scaled - reconstructed), axis=1)
        
        # 8. Detect anomalies
        is_anomaly = mse_per_sample > threshold
        
        # 9. Calculate anomaly scores (normalized)
        anomaly_scores = np.minimum(1.0, mse_per_sample / threshold)
        
        # 10. Calculate per-feature errors for each sample
        per_sample_feature_errors = np.mean(np.square(test_data_scaled - reconstructed), axis=1)  # This is per sample, not per feature
        # Actually compute per-feature errors correctly:
        feature_errors_per_sample = np.square(test_data_scaled - reconstructed)  # Shape: (n_samples, n_features)
        
        # 11. Display results
        print("\n" + "="*60)
        print("📊 ANOMALY DETECTION RESULTS")
        print("="*60)
        
        for i in range(len(test_data)):
            severity = "HIGH" if anomaly_scores[i] > 0.8 else "MEDIUM" if anomaly_scores[i] > 0.5 else "LOW"
            
            print(f"\nSample {i+1}:")
            print(f"  Reconstruction Error: {mse_per_sample[i]:.6f}")
            print(f"  Anomaly Score: {anomaly_scores[i]:.3f}")
            print(f"  Anomaly Flag: {'⚠️  ANOMALY' if is_anomaly[i] else '✅ NORMAL'}")
            print(f"  Severity: {severity}")
            
            if is_anomaly[i]:
                # Show feature-wise contributions
                sample_feature_errors = feature_errors_per_sample[i]
                top_features_idx = np.argsort(sample_feature_errors)[-5:][::-1]  # Top 5 features
                print(f"  Top contributing features:")
                for idx in top_features_idx:
                    print(f"    - {ALL_FEATURES[idx]}: {sample_feature_errors[idx]:.6f}")
        
        print("\n" + "="*60)
        print(f"Summary:")
        print(f"  Total samples: {len(test_data)}")
        print(f"  Anomalies detected: {np.sum(is_anomaly)}")
        print(f"  Anomaly rate: {np.mean(is_anomaly):.2%}")
        print("="*60)
        print("✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)