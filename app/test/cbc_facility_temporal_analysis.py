#!/usr/bin/env python3
"""
Test script for CBC Facility Temporal LSTM Autoencoder (Model 4)
Verifies model loading and facility behavior anomaly detection on synthetic data
"""

import os
import sys
import numpy as np
import tensorflow as tf
import joblib
import json
import pandas as pd

# Paths
BASE_DIR = "/home/comphortinoe/comphortinoe/dev/resultshield/rsl-engine"
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Model files
MODEL_PATH = os.path.join(MODEL_DIR, "model4_facility_temporal_ae.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "model4_facility_scaler.pkl")
META_PATH = os.path.join(MODEL_DIR, "model4_facility_temporal_meta.json")

# Feature names (must match training - these are weekly aggregates)
# Matches model4_facility_temporal_meta.json feature_names exactly
FACILITY_FEATURE_NAMES = [
    "claim_volume", "avg_age", "age_std", "pct_male",
    "HGB_mean", "HGB_std", "HCT_mean", "HCT_std",
    "MCV_mean", "MCV_std", "MCHC_mean", "MCHC_std",
    "NEU_mean", "LYM_mean", "EOS_mean", "BAS_mean", "MON_mean", "PLT_mean",
    "avg_los",
]
N_FACILITY_FEATURES = len(FACILITY_FEATURE_NAMES)

def create_synthetic_facility_sequence(seq_len=8):
    """
    Create synthetic facility weekly sequences with normal and anomalous patterns
    
    Returns:
        sequences: Array of shape (n_sequences, seq_len, n_features)
        descriptions: List of sequence descriptions
    """
    np.random.seed(42)
    
    # Normal ranges for weekly aggregates
    normal_ranges = {
        'claim_volume': (20, 50),
        'avg_age': (35, 55),
        'age_std': (10, 20),
        'pct_male': (0.4, 0.6),
        'HGB_mean': (12.5, 15.5),
        'HGB_std': (0.5, 1.5),
        'HCT_mean': (38, 45),
        'HCT_std': (2, 4),
        'MCV_mean': (85, 95),
        'MCV_std': (3, 6),
        'MCHC_mean': (32, 35),
        'MCHC_std': (0.5, 1.5),
        'NEU_mean': (45, 65),
        'LYM_mean': (25, 40),
        'EOS_mean': (2, 4),
        'BAS_mean': (0.6, 1.2),
        'MON_mean': (5, 8),
        'PLT_mean': (200, 350),
        'avg_los': (3, 7),
    }
    
    def generate_normal_week():
        """Generate a normal weekly aggregate"""
        week = []
        for feature in FACILITY_FEATURE_NAMES:
            low, high = normal_ranges[feature]
            value = np.random.uniform(low, high)
            week.append(value)
        return np.array(week)
    
    def generate_anomalous_week(anomaly_type):
        """Generate an anomalous weekly aggregate"""
        week = []
        for feature in FACILITY_FEATURE_NAMES:
            if anomaly_type == 'high_volume' and feature == 'claim_volume':
                value = np.random.uniform(150, 300)  # Spike in claims
            elif anomaly_type == 'low_hgb' and feature == 'HGB_mean':
                value = np.random.uniform(8, 10)  # Low hemoglobin (anemia cases)
            elif anomaly_type == 'high_plt' and feature == 'PLT_mean':
                value = np.random.uniform(500, 700)  # High platelets
            elif anomaly_type == 'high_variance' and feature.endswith('_std'):
                value = np.random.uniform(8, 15)  # High variability
            elif anomaly_type == 'abnormal_los' and feature == 'avg_los':
                value = np.random.uniform(15, 25)  # Extended stays
            else:
                low, high = normal_ranges.get(feature, (0, 1))
                value = np.random.uniform(low, high)
            week.append(value)
        return np.array(week)
    
    sequences = []
    descriptions = []
    
    # Sequence 1: Normal facility behavior (consistent weeks)
    normal_seq = []
    for _ in range(seq_len):
        normal_seq.append(generate_normal_week())
    sequences.append(np.array(normal_seq))
    descriptions.append("Normal facility behavior")
    
    # Sequence 2: Sudden volume spike at week 6
    volume_spike_seq = []
    for i in range(seq_len):
        if i < 5:
            volume_spike_seq.append(generate_normal_week())
        else:
            volume_spike_seq.append(generate_anomalous_week('high_volume'))
    sequences.append(np.array(volume_spike_seq))
    descriptions.append("Sudden claim volume spike (week 6-8)")
    
    # Sequence 3: Low HGB trend (anemia outbreak)
    low_hgb_seq = []
    for i in range(seq_len):
        if i < 4:
            low_hgb_seq.append(generate_normal_week())
        else:
            low_hgb_seq.append(generate_anomalous_week('low_hgb'))
    sequences.append(np.array(low_hgb_seq))
    descriptions.append("Low HGB trend (possible anemia cases)")
    
    # Sequence 4: High platelet counts
    high_plt_seq = []
    for i in range(seq_len):
        if i < 5:
            high_plt_seq.append(generate_normal_week())
        else:
            high_plt_seq.append(generate_anomalous_week('high_plt'))
    sequences.append(np.array(high_plt_seq))
    descriptions.append("High platelet counts (possible inflammation/thrombocytosis)")
    
    # Sequence 5: High variance (unstable lab values)
    high_var_seq = []
    for i in range(seq_len):
        if i < 4:
            high_var_seq.append(generate_normal_week())
        else:
            high_var_seq.append(generate_anomalous_week('high_variance'))
    sequences.append(np.array(high_var_seq))
    descriptions.append("High variance in lab values (possible data quality issues)")
    
    # Sequence 6: Extended length of stay
    extended_los_seq = []
    for i in range(seq_len):
        if i < 5:
            extended_los_seq.append(generate_normal_week())
        else:
            extended_los_seq.append(generate_anomalous_week('abnormal_los'))
    sequences.append(np.array(extended_los_seq))
    descriptions.append("Extended length of stay (unusual patient acuity)")
    
    # Sequence 7: Combined anomalies (volume spike + low HGB)
    combined_seq = []
    for i in range(seq_len):
        if i < 3:
            combined_seq.append(generate_normal_week())
        elif i < 5:
            combined_seq.append(generate_anomalous_week('low_hgb'))
        else:
            combined_seq.append(generate_anomalous_week('high_volume'))
    sequences.append(np.array(combined_seq))
    descriptions.append("Combined: low HGB then volume spike")
    
    return np.array(sequences), descriptions

def run_test():
    """Load facility temporal model and run anomaly detection on synthetic sequences"""
    
    print("="*60)
    print("CBC Facility Temporal LSTM Autoencoder - Test Script")
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
        seq_len = metadata.get('sequence_length', 8)
        n_features = metadata.get('n_features', N_FACILITY_FEATURES)
        
        print(f"✅ Metadata loaded")
        print(f"   Sequence length (weeks): {seq_len}")
        print(f"   Features per week: {n_features}")
        print(f"   Anomaly threshold: {threshold:.6f}")
        print(f"   Best validation loss: {metadata.get('best_val_loss', 'N/A')}")
        
        # 4. Create synthetic test sequences
        print("\n🔧 Creating synthetic facility sequences...")
        test_sequences, descriptions = create_synthetic_facility_sequence(seq_len=seq_len)
        print(f"✅ Test sequences shape: {test_sequences.shape}")
        
        # Display sequence summaries
        print("\n📊 Test Sequence Summary:")
        for i, desc in enumerate(descriptions):
            seq = test_sequences[i]
            print(f"\n  Sequence {i+1}: {desc}")
            print(f"    Weeks 1-3 avg claim volume: {np.mean(seq[:3, 0]):.1f}")
            print(f"    Weeks 6-8 avg claim volume: {np.mean(seq[5:, 0]):.1f}")
            print(f"    Avg HGB (weeks 1-3): {np.mean(seq[:3, 4]):.1f}")   # HGB_mean is index 4
            print(f"    Avg HGB (weeks 6-8): {np.mean(seq[5:, 4]):.1f}")
            print(f"    Avg PLT (weeks 1-3): {np.mean(seq[:3, 17]):.1f}")  # PLT_mean is index 17
            print(f"    Avg PLT (weeks 6-8): {np.mean(seq[5:, 17]):.1f}")
        
        # 5. Flatten sequences for scaling
        n_sequences = len(test_sequences)
        flattened = test_sequences.reshape(-1, n_features)

        # Scale using DataFrame to match feature names the scaler was fitted with
        import pandas as pd
        flattened_df = pd.DataFrame(flattened, columns=FACILITY_FEATURE_NAMES)
        flattened_scaled = scaler.transform(flattened_df)
        # Replace NaN from zero-variance features with 0.0
        flattened_scaled = np.nan_to_num(flattened_scaled, nan=0.0)
        
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
        
        # 10. Calculate per-week errors
        per_week_errors = np.mean(np.square(test_scaled - reconstructed), axis=2)  # Shape: (n_sequences, seq_len)
        
        # 11. Display results
        print("\n" + "="*60)
        print("📊 FACILITY BEHAVIOR ANOMALY DETECTION RESULTS")
        print("="*60)
        
        for i in range(n_sequences):
            severity = "HIGH" if anomaly_scores[i] > 0.8 else "MEDIUM" if anomaly_scores[i] > 0.5 else "LOW"
            
            print(f"\nSequence {i+1}: {descriptions[i]}")
            print(f"  Reconstruction Error: {mse_per_sequence[i]:.6f}")
            print(f"  Anomaly Score: {anomaly_scores[i]:.3f}")
            print(f"  Anomaly Flag: {'⚠️  ANOMALOUS FACILITY' if is_anomaly[i] else '✅ NORMAL FACILITY'}")
            print(f"  Severity: {severity}")
            
            if is_anomaly[i]:
                # Show which weeks are most anomalous
                week_errors = per_week_errors[i]
                most_anomalous_weeks = np.argsort(week_errors)[-3:][::-1]
                print(f"  Most anomalous weeks: {most_anomalous_weeks + 1}")
                
                # Show feature contributions for the most anomalous week
                worst_week = most_anomalous_weeks[0]
                feature_errors = np.square(test_scaled[i, worst_week] - reconstructed[i, worst_week])
                top_features_idx = np.argsort(feature_errors)[-5:][::-1]
                print(f"\n  Top contributing features (Week {worst_week + 1}):")
                for idx in top_features_idx:
                    feature_name = FACILITY_FEATURE_NAMES[idx]
                    orig_val = test_scaled[i, worst_week, idx]
                    recon_val = reconstructed[i, worst_week, idx]
                    print(f"    - {feature_name}: error={feature_errors[idx]:.6f} (orig={orig_val:.3f}, recon={recon_val:.3f})")
        
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