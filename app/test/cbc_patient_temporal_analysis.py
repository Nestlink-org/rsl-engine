#!/usr/bin/env python3
"""
Test script for CBC Patient Temporal LSTM Autoencoder (Model 3)
Verifies model loading and patient trajectory anomaly detection on synthetic data
"""

import os
import sys
import numpy as np
import tensorflow as tf
import joblib
import json
import pandas as pd
from datetime import datetime, timedelta

# Paths
BASE_DIR = "/home/comphortinoe/comphortinoe/dev/resultshield/rsl-engine"
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Model files
MODEL_PATH = os.path.join(MODEL_DIR, "cbc_model3_patient_temporal_ae.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "cbc_model3_patient_scaler.pkl")
META_PATH = os.path.join(MODEL_DIR, "cbc_model3_patient_temporal_meta.json")

# Feature names (must match training)
LAB_FEATURES = ['HGB', 'HCT', 'MCV', 'MCHC', 'NEU', 'LYM', 'EOS', 'BAS', 'MON', 'PLT']
DEMO_FEATURES = ['age', 'sex_encoded']
TEMPORAL_FEATURES = ['length_of_stay']
PATIENT_FEATURES = DEMO_FEATURES + LAB_FEATURES + TEMPORAL_FEATURES
N_PATIENT_FEATURES = len(PATIENT_FEATURES)

def create_synthetic_patient_sequence(seq_len=5):
    """Create a synthetic patient visit sequence with normal and anomalous patterns"""
    np.random.seed(42)
    
    # Normal ranges
    normal_ranges = {
        'age': (20, 80),
        'sex_encoded': [0, 1],
        'length_of_stay': (1, 7),
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
    
    def generate_normal_visit(age=None, sex=None):
        visit = []
        # Age (consistent across visits for same patient)
        if age is None:
            age = np.random.uniform(normal_ranges['age'][0], normal_ranges['age'][1])
        visit.append(age)
        
        # Sex (consistent)
        if sex is None:
            sex = np.random.choice(normal_ranges['sex_encoded'])
        visit.append(sex)
        
        # Lab values
        for feature in LAB_FEATURES:
            low, high = normal_ranges[feature]
            value = np.random.uniform(low, high)
            visit.append(value)
        
        # Length of stay
        los_low, los_high = normal_ranges['length_of_stay']
        los = np.random.uniform(los_low, los_high)
        visit.append(los)
        
        return np.array(visit)
    
    def generate_anomalous_visit(age, sex, anomaly_type):
        visit = []
        visit.append(age)
        visit.append(sex)
        
        for feature in LAB_FEATURES:
            if anomaly_type == 'low_hgb' and feature == 'HGB':
                value = np.random.uniform(4.0, 8.0)  # Severe anemia
            elif anomaly_type == 'high_plt' and feature == 'PLT':
                value = np.random.uniform(500.0, 800.0)  # Thrombocytosis
            elif anomaly_type == 'high_neu' and feature == 'NEU':
                value = np.random.uniform(20.0, 30.0)  # High neutrophils
            elif anomaly_type == 'abnormal_los' and feature == 'length_of_stay':
                # This is handled separately
                pass
            else:
                low, high = normal_ranges[feature]
                value = np.random.uniform(low, high)
            visit.append(value)
        
        # Length of stay
        if anomaly_type == 'abnormal_los':
            los = np.random.uniform(15.0, 30.0)  # Extended stay
        else:
            los = np.random.uniform(1.0, 7.0)
        visit.append(los)
        
        return np.array(visit)
    
    # Create sequences
    sequences = []
    
    # Sequence 1: Normal trajectory (5 normal visits)
    age = np.random.uniform(20, 80)
    sex = np.random.choice([0, 1])
    normal_seq = []
    for _ in range(seq_len):
        normal_seq.append(generate_normal_visit(age, sex))
    sequences.append(np.array(normal_seq))
    
    # Sequence 2: Anomalous - Sudden drop in HGB at last visit
    age = np.random.uniform(20, 80)
    sex = np.random.choice([0, 1])
    anemia_seq = []
    for i in range(seq_len):
        if i < seq_len - 1:
            anemia_seq.append(generate_normal_visit(age, sex))
        else:
            anemia_seq.append(generate_anomalous_visit(age, sex, 'low_hgb'))
    sequences.append(np.array(anemia_seq))
    
    # Sequence 3: Anomalous - Sudden spike in PLT at last visit
    age = np.random.uniform(20, 80)
    sex = np.random.choice([0, 1])
    plt_seq = []
    for i in range(seq_len):
        if i < seq_len - 1:
            plt_seq.append(generate_normal_visit(age, sex))
        else:
            plt_seq.append(generate_anomalous_visit(age, sex, 'high_plt'))
    sequences.append(np.array(plt_seq))
    
    # Sequence 4: Anomalous - Extended LOS throughout
    age = np.random.uniform(20, 80)
    sex = np.random.choice([0, 1])
    los_seq = []
    for _ in range(seq_len):
        los_seq.append(generate_anomalous_visit(age, sex, 'abnormal_los'))
    sequences.append(np.array(los_seq))
    
    # Sequence 5: Anomalous - Multiple anomalies (low HGB + high PLT)
    age = np.random.uniform(20, 80)
    sex = np.random.choice([0, 1])
    combined_seq = []
    for i in range(seq_len):
        if i < seq_len - 2:
            combined_seq.append(generate_normal_visit(age, sex))
        elif i == seq_len - 2:
            combined_seq.append(generate_anomalous_visit(age, sex, 'low_hgb'))
        else:
            combined_seq.append(generate_anomalous_visit(age, sex, 'high_plt'))
    sequences.append(np.array(combined_seq))
    
    return np.array(sequences)

def run_test():
    """Load patient temporal model and run anomaly detection on synthetic sequences"""
    
    print("="*60)
    print("CBC Patient Temporal LSTM Autoencoder - Test Script")
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
        seq_len = metadata.get('sequence_length', 5)
        n_features = metadata.get('n_features', N_PATIENT_FEATURES)
        
        print(f"✅ Metadata loaded")
        print(f"   Sequence length: {seq_len}")
        print(f"   Features per visit: {n_features}")
        print(f"   Anomaly threshold: {threshold:.6f}")
        print(f"   Best validation loss: {metadata.get('best_val_loss', 'N/A')}")
        
        # 4. Create synthetic test sequences
        print("\n🔧 Creating synthetic patient sequences...")
        test_sequences = create_synthetic_patient_sequence(seq_len=seq_len)
        print(f"✅ Test sequences shape: {test_sequences.shape}")
        
        # Display sequence summaries
        print("\n📊 Test Sequence Summary:")
        for i in range(len(test_sequences)):
            seq = test_sequences[i]
            first_visit = seq[0]
            last_visit = seq[-1]
            print(f"\n  Sequence {i+1}:")
            print(f"    Patient Age: {first_visit[0]:.0f}")
            print(f"    Sex: {'Male' if first_visit[1] == 1 else 'Female'}")
            print(f"    First Visit - HGB: {first_visit[2]:.1f}, PLT: {first_visit[-1]:.1f}, LOS: {first_visit[-2]:.1f}")
            print(f"    Last Visit - HGB: {last_visit[2]:.1f}, PLT: {last_visit[-1]:.1f}, LOS: {last_visit[-2]:.1f}")
        
        # 5. Flatten sequences for scaling (need to scale each feature independently)
        # Reshape to (n_sequences * seq_len, n_features) for scaling
        n_sequences = len(test_sequences)
        flattened = test_sequences.reshape(-1, n_features)
        
        # Scale
        flattened_scaled = scaler.transform(flattened)
        
        # Reshape back to sequences
        test_scaled = flattened_scaled.reshape(n_sequences, seq_len, n_features)
        
        # 6. Run inference
        print("\n🚀 Running anomaly detection...")
        reconstructed = model.predict(test_scaled, verbose=0)
        
        # 7. Calculate reconstruction errors
        mse_per_sequence = np.mean(np.square(test_scaled - reconstructed), axis=(1, 2))
        
        # 8. Detect anomalies
        is_anomaly = mse_per_sequence > threshold
        
        # 9. Calculate anomaly scores
        anomaly_scores = np.minimum(1.0, mse_per_sequence / threshold)
        
        # 10. Calculate per-visit errors
        per_visit_errors = np.mean(np.square(test_scaled - reconstructed), axis=2)  # Shape: (n_sequences, seq_len)
        
        # 11. Display results
        print("\n" + "="*60)
        print("📊 PATIENT TRAJECTORY ANOMALY DETECTION RESULTS")
        print("="*60)
        
        for i in range(n_sequences):
            severity = "HIGH" if anomaly_scores[i] > 0.8 else "MEDIUM" if anomaly_scores[i] > 0.5 else "LOW"
            
            print(f"\nSequence {i+1}:")
            print(f"  Reconstruction Error: {mse_per_sequence[i]:.6f}")
            print(f"  Anomaly Score: {anomaly_scores[i]:.3f}")
            print(f"  Anomaly Flag: {'⚠️  ANOMALOUS TRAJECTORY' if is_anomaly[i] else '✅ NORMAL TRAJECTORY'}")
            print(f"  Severity: {severity}")
            
            if is_anomaly[i]:
                # Show which visits are most anomalous
                visit_errors = per_visit_errors[i]
                most_anomalous_visit = np.argmax(visit_errors)
                print(f"  Most anomalous visit: Visit {most_anomalous_visit + 1} (error: {visit_errors[most_anomalous_visit]:.6f})")
                
                # Show feature contributions for the most anomalous visit
                if most_anomalous_visit >= 0:
                    visit_idx = most_anomalous_visit
                    feature_errors = np.square(test_scaled[i, visit_idx] - reconstructed[i, visit_idx])
                    top_features_idx = np.argsort(feature_errors)[-5:][::-1]
                    print(f"  Top contributing features in anomalous visit:")
                    for idx in top_features_idx:
                        print(f"    - {PATIENT_FEATURES[idx]}: {feature_errors[idx]:.6f}")
        
        print("\n" + "="*60)
        print(f"Summary:")
        print(f"  Total sequences: {n_sequences}")
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