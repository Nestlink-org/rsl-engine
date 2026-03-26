#!/usr/bin/env python3
"""
Test script for CBC Hierarchical Disease Classifier (Model 2)
Verifies model loading and inference with synthetic test data
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
MODEL_PATH = os.path.join(MODEL_DIR, "cbc_model2_hierarchical_classifier.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "cbc_model2_scaler.pkl")
META_PATH = os.path.join(MODEL_DIR, "cbc_model2_hierarchical_classifier_meta.json")
CATEGORY_ENCODER_PATH = os.path.join(MODEL_DIR, "cbc_model2_category_encoder.pkl")
DIAGNOSIS_ENCODER_PATH = os.path.join(MODEL_DIR, "cbc_model2_diagnosis_encoder.pkl")

# Feature names (must match training)
LAB_FEATURES = ['HGB', 'HCT', 'MCV', 'MCHC', 'NEU', 'LYM', 'EOS', 'BAS', 'MON', 'PLT']
N_FEATURES = len(LAB_FEATURES)

def create_synthetic_test_data(n_samples=5):
    """Create synthetic FBC test samples within normal ranges"""
    np.random.seed(42)
    
    # Normal ranges for each lab value
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
    
    test_samples = []
    for _ in range(n_samples):
        sample = []
        for feature in LAB_FEATURES:
            low, high = normal_ranges[feature]
            value = np.random.uniform(low, high)
            sample.append(value)
        test_samples.append(sample)
    
    return np.array(test_samples)

def run_test():
    """Load model and run inference on synthetic data"""
    
    print("="*60)
    print("CBC Hierarchical Disease Classifier - Test Script")
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
        
        # 3. Load encoders
        print("\n📦 Loading encoders...")
        category_encoder = joblib.load(CATEGORY_ENCODER_PATH)
        diagnosis_encoder = joblib.load(DIAGNOSIS_ENCODER_PATH)
        print(f"✅ Category encoder loaded ({len(category_encoder.classes_)} classes)")
        print(f"✅ Diagnosis encoder loaded ({len(diagnosis_encoder.classes_)} classes)")
        
        # 4. Load metadata
        print("\n📦 Loading metadata...")
        with open(META_PATH, 'r') as f:
            metadata = json.load(f)
        print(f"✅ Metadata loaded (input_dim: {metadata.get('input_dim', 'unknown')})")
        
        # 5. Create synthetic test data
        print("\n🔧 Creating synthetic test data...")
        test_data = create_synthetic_test_data(n_samples=5)
        print(f"✅ Test data shape: {test_data.shape}")
        
        # 6. Scale test data
        test_data_scaled = scaler.transform(test_data)
        
        # 7. Run inference
        print("\n🚀 Running inference...")
        cat_proba, diag_proba = model.predict(test_data_scaled, verbose=0)
        
        # 8. Display results
        print("\n" + "="*60)
        print("📊 PREDICTION RESULTS")
        print("="*60)
        
        for i in range(len(test_data)):
            # Category prediction
            pred_cat_idx = np.argmax(cat_proba[i])
            pred_cat = category_encoder.inverse_transform([pred_cat_idx])[0]
            cat_conf = cat_proba[i][pred_cat_idx]
            
            # Diagnosis prediction
            pred_diag_idx = np.argmax(diag_proba[i])
            pred_diag = diagnosis_encoder.inverse_transform([pred_diag_idx])[0]
            diag_conf = diag_proba[i][pred_diag_idx]
            
            print(f"\nSample {i+1}:")
            print(f"  Input (HGB, HCT, MCV, MCHC, NEU, LYM, EOS, BAS, MON, PLT):")
            print(f"    {', '.join([f'{v:.1f}' for v in test_data[i]])}")
            print(f"  Predicted Category: {pred_cat} (confidence: {cat_conf:.3f})")
            print(f"  Predicted Diagnosis: {pred_diag} (confidence: {diag_conf:.3f})")
        
        print("\n" + "="*60)
        print("✅ Test completed successfully!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)