┌──(.venv)─(comphortinoe㉿kali)-[~/comphortinoe/dev/resultshield/rsl-engine]
└─$ python app/test/disease_classifier.py
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1774006230.039802 2368508 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
I0000 00:00:1774006230.932157 2368508 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
/home/comphortinoe/comphortinoe/dev/resultshield/rsl-engine/.venv/lib/python3.12/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.2.0)/charset_normalizer (3.4.6) doesn't match a supported version!
  warnings.warn(
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1774006239.952906 2368508 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
============================================================
CBC Hierarchical Disease Classifier - Test Script
============================================================

📦 Loading model...
E0000 00:00:1774006245.737252 2368508 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
✅ Model loaded successfully

📦 Loading scaler...
/home/comphortinoe/comphortinoe/dev/resultshield/rsl-engine/.venv/lib/python3.12/site-packages/sklearn/base.py:463: InconsistentVersionWarning: Trying to unpickle estimator StandardScaler from version 1.5.1 when using version 1.8.0. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
  warnings.warn(
✅ Scaler loaded successfully

📦 Loading encoders...
/home/comphortinoe/comphortinoe/dev/resultshield/rsl-engine/.venv/lib/python3.12/site-packages/sklearn/base.py:463: InconsistentVersionWarning: Trying to unpickle estimator LabelEncoder from version 1.5.1 when using version 1.8.0. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
  warnings.warn(
✅ Category encoder loaded (3 classes)
✅ Diagnosis encoder loaded (5 classes)

📦 Loading metadata...
✅ Metadata loaded (input_dim: 10)

🔧 Creating synthetic test data...
✅ Test data shape: (5, 10)

🚀 Running inference...

============================================================
📊 PREDICTION RESULTS
============================================================

Sample 1:
  Input (HGB, HCT, MCV, MCHC, NEU, LYM, EOS, BAS, MON, PLT):
    13.5, 47.4, 94.6, 34.0, 44.7, 23.9, 1.2, 1.4, 6.8, 327.0
  Predicted Category: respiratory (confidence: 0.998)
  Predicted Diagnosis: ASTHMA (confidence: 1.000)

Sample 2:
  Input (HGB, HCT, MCV, MCHC, NEU, LYM, EOS, BAS, MON, PLT):
    12.1, 47.6, 96.6, 32.1, 45.5, 24.6, 2.2, 1.0, 5.5, 222.8
  Predicted Category: obstetric (confidence: 0.505)
  Predicted Diagnosis: PUERPERAL SEPSIS (confidence: 0.344)

Sample 3:
  Input (HGB, HCT, MCV, MCHC, NEU, LYM, EOS, BAS, MON, PLT):
    14.4, 37.7, 85.8, 32.8, 53.7, 39.6, 1.8, 1.0, 6.7, 161.6
  Predicted Category: obstetric (confidence: 0.538)
  Predicted Diagnosis: PUERPERAL SEPSIS (confidence: 0.410)

Sample 4:
  Input (HGB, HCT, MCV, MCHC, NEU, LYM, EOS, BAS, MON, PLT):
    14.4, 38.0, 81.3, 35.7, 69.0, 40.2, 2.2, 0.6, 7.5, 260.0
  Predicted Category: obstetric (confidence: 0.510)
  Predicted Diagnosis: PUERPERAL SEPSIS (confidence: 0.402)

Sample 5:
  Input (HGB, HCT, MCV, MCHC, NEU, LYM, EOS, BAS, MON, PLT):
    12.5, 41.9, 80.7, 35.5, 47.8, 36.6, 2.2, 1.0, 6.4, 196.2
  Predicted Category: obstetric (confidence: 0.554)
  Predicted Diagnosis: PUERPERAL SEPSIS (confidence: 0.413)

============================================================
✅ Test completed successfully!
============================================================
                                                                                                                            
┌──(.venv)─(comphortinoe㉿kali)-[~/comphortinoe/dev/resultshield/rsl-engine]
└─$ 


                                                                                                                            
┌──(.venv)─(comphortinoe㉿kali)-[~/comphortinoe/dev/resultshield/rsl-engine]
└─$ python app/test/claim_autoencoder.py
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1774007336.184323 2390302 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
I0000 00:00:1774007336.275565 2390302 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
/home/comphortinoe/comphortinoe/dev/resultshield/rsl-engine/.venv/lib/python3.12/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.2.0)/charset_normalizer (3.4.6) doesn't match a supported version!
  warnings.warn(
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1774007338.436910 2390302 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
============================================================
CBC Claim Autoencoder - Test Script
============================================================

📦 Loading model...
E0000 00:00:1774007340.829239 2390302 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
✅ Model loaded successfully

📦 Loading scaler...
/home/comphortinoe/comphortinoe/dev/resultshield/rsl-engine/.venv/lib/python3.12/site-packages/sklearn/base.py:463: InconsistentVersionWarning: Trying to unpickle estimator MinMaxScaler from version 1.5.1 when using version 1.8.0. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
  warnings.warn(
✅ Scaler loaded successfully

📦 Loading metadata...
✅ Metadata loaded
   Input dimension: 12
   Anomaly threshold: 0.000050
   Best validation loss: 1.0407140507595614e-05

🔧 Creating synthetic test data...
✅ Test data shape: (5, 12)

📊 Raw Test Data:
  Sample 1: Age=42, Sex=Female, HGB=12.7, NEU=33.2, PLT=164.1
  Sample 2: Age=63, Sex=Male, HGB=4.8, NEU=32.5, PLT=223.0
  Sample 3: Age=42, Sex=Male, HGB=12.4, NEU=35.9, PLT=782.7
  Sample 4: Age=54, Sex=Male, HGB=13.2, NEU=33.2, PLT=315.6
  Sample 5: Age=39, Sex=Male, HGB=4.8, NEU=35.2, PLT=671.1

🚀 Running anomaly detection...

============================================================
📊 ANOMALY DETECTION RESULTS
============================================================

Sample 1:
  Reconstruction Error: 0.000002
  Anomaly Score: 0.049
  Anomaly Flag: ✅ NORMAL
  Severity: LOW

Sample 2:
  Reconstruction Error: 0.000061
  Anomaly Score: 1.000
  Anomaly Flag: ⚠️  ANOMALY
  Severity: HIGH
  Top contributing features:
    - BAS: 0.000167
    - MON: 0.000151
    - age: 0.000138
    - HGB: 0.000120
    - EOS: 0.000048

Sample 3:
  Reconstruction Error: 0.000087
  Anomaly Score: 1.000
  Anomaly Flag: ⚠️  ANOMALY
  Severity: HIGH
  Top contributing features:
    - PLT: 0.000410
    - EOS: 0.000135
    - HCT: 0.000133
    - MCV: 0.000112
    - HGB: 0.000076

Sample 4:
  Reconstruction Error: 0.000115
  Anomaly Score: 1.000
  Anomaly Flag: ⚠️  ANOMALY
  Severity: HIGH
  Top contributing features:
    - NEU: 0.000744
    - BAS: 0.000195
    - MON: 0.000191
    - LYM: 0.000068
    - PLT: 0.000058

Sample 5:
  Reconstruction Error: 0.000187
  Anomaly Score: 1.000
  Anomaly Flag: ⚠️  ANOMALY
  Severity: HIGH
  Top contributing features:
    - MON: 0.000739
    - EOS: 0.000694
    - MCV: 0.000165
    - HGB: 0.000151
    - PLT: 0.000137

============================================================
Summary:
  Total samples: 5
  Anomalies detected: 4
  Anomaly rate: 80.00%
============================================================
✅ Test completed successfully!
                                                                                                                            
┌──(.venv)─(comphortinoe㉿kali)-[~/comphortinoe/dev/resultshield/rsl-engine]
└─$ 


┌──(.venv)─(comphortinoe㉿kali)-[~/comphortinoe/dev/resultshield/rsl-engine]
└─$ python app/test/patient_temporal_analysis.py
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1774008254.810761 2408177 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
I0000 00:00:1774008254.926155 2408177 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
/home/comphortinoe/comphortinoe/dev/resultshield/rsl-engine/.venv/lib/python3.12/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.2.0)/charset_normalizer (3.4.6) doesn't match a supported version!
  warnings.warn(
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1774008256.986195 2408177 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
============================================================
CBC Patient Temporal LSTM Autoencoder - Test Script
============================================================

📦 Loading model...
E0000 00:00:1774008259.326975 2408177 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
✅ Model loaded successfully

📦 Loading scaler...
/home/comphortinoe/comphortinoe/dev/resultshield/rsl-engine/.venv/lib/python3.12/site-packages/sklearn/base.py:463: InconsistentVersionWarning: Trying to unpickle estimator StandardScaler from version 1.5.1 when using version 1.8.0. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
  warnings.warn(
✅ Scaler loaded successfully

📦 Loading metadata...
✅ Metadata loaded
   Sequence length: 5
   Features per visit: 13
   Anomaly threshold: 0.295183
   Best validation loss: 0.139494389295578

🔧 Creating synthetic patient sequences...
✅ Test sequences shape: (5, 5, 13)

📊 Test Sequence Summary:

  Sequence 1:
    Patient Age: 42
    Sex: Female
    First Visit - HGB: 12.7, PLT: 5.3, LOS: 164.1
    Last Visit - HGB: 15.0, PLT: 3.0, LOS: 331.8

  Sequence 2:
    Patient Age: 54
    Sex: Male
    First Visit - HGB: 12.2, PLT: 1.4, LOS: 350.5
    Last Visit - HGB: 5.3, PLT: 6.6, LOS: 190.3

  Sequence 3:
    Patient Age: 68
    Sex: Female
    First Visit - HGB: 12.4, PLT: 4.9, LOS: 218.0
    Last Visit - HGB: 13.0, PLT: 6.2, LOS: 708.7

  Sequence 4:
    Patient Age: 40
    Sex: Male
    First Visit - HGB: 12.1, PLT: 16.7, LOS: 235.3
    Last Visit - HGB: 12.9, PLT: 18.7, LOS: 216.3

  Sequence 5:
    Patient Age: 78
    Sex: Female
    First Visit - HGB: 13.9, PLT: 5.6, LOS: 293.0
    Last Visit - HGB: 13.0, PLT: 1.2, LOS: 766.0
/home/comphortinoe/comphortinoe/dev/resultshield/rsl-engine/.venv/lib/python3.12/site-packages/sklearn/utils/validation.py:2691: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
  warnings.warn(

🚀 Running anomaly detection...

============================================================
📊 PATIENT TRAJECTORY ANOMALY DETECTION RESULTS
============================================================

Sequence 1:
  Reconstruction Error: 0.064544
  Anomaly Score: 0.219
  Anomaly Flag: ✅ NORMAL TRAJECTORY
  Severity: LOW

Sequence 2:
  Reconstruction Error: 0.100089
  Anomaly Score: 0.339
  Anomaly Flag: ✅ NORMAL TRAJECTORY
  Severity: LOW

Sequence 3:
  Reconstruction Error: 0.076748
  Anomaly Score: 0.260
  Anomaly Flag: ✅ NORMAL TRAJECTORY
  Severity: LOW

Sequence 4:
  Reconstruction Error: 0.280771
  Anomaly Score: 0.951
  Anomaly Flag: ✅ NORMAL TRAJECTORY
  Severity: HIGH

Sequence 5:
  Reconstruction Error: 0.131138
  Anomaly Score: 0.444
  Anomaly Flag: ✅ NORMAL TRAJECTORY
  Severity: LOW

============================================================
Summary:
  Total sequences: 5
  Anomalies detected: 0
  Anomaly rate: 0.00%
============================================================
✅ Test completed successfully!
                                                                                                                            
┌──(.venv)─(comphortinoe㉿kali)-[~/comphortinoe/dev/resultshield/rsl-engine]
└─$ 

# ==================== TEST MODEL 3 LOADING ====================

print("🔍 Testing Model 3 (Patient Temporal) loading and inference...")

# Load model
loaded_model3 = tf.keras.models.load_model(model3_path)
print("✅ Model 3 loaded successfully")

# Load metadata
with open(meta3_path, 'r') as f:
    loaded_meta3 = json.load(f)
print("✅ Metadata loaded successfully")

# Load scaler
loaded_scaler3 = joblib.load(scaler_patient_path)
print("✅ Scaler loaded successfully")

# Test prediction
test_samples = X_patient_test[:5]
predictions = loaded_model3.predict(test_samples, verbose=0)

print(f"\n📊 Test inference:")
print(f"  Input shape: {test_samples.shape}")
print(f"  Output shape: {predictions.shape}")

# Calculate sequence errors
seq_errors = np.mean(np.square(test_samples - predictions), axis=(1, 2))
print(f"  Sequence errors: {seq_errors}")
print(f"  Anomaly flags: {seq_errors > loaded_meta3['threshold']}")

print("\n🔍 Testing Model 4 (Facility Temporal) loading and inference...")

# Load model
loaded_model4 = tf.keras.models.load_model(model4_path)
print("✅ Model 4 loaded successfully")

# Load metadata
with open(meta4_path, "r") as f:
    loaded_meta4 = json.load(f)
print("✅ Metadata loaded successfully")

# Load scaler
loaded_scaler4 = joblib.load(scaler_facility_path)
print("✅ Scaler loaded successfully")

# Test prediction
test_samples = X_facility_test[:5]
predictions = loaded_model4.predict(test_samples, verbose=0)

print("\n📊 Test inference:")
print("Input shape:", test_samples.shape)
print("Output shape:", predictions.shape)

# Sequence reconstruction error
seq_errors = np.mean(np.square(test_samples - predictions), axis=(1,2))

print("Sequence errors:", seq_errors)
print("Anomaly flags:", seq_errors > loaded_meta4["threshold"])

# ==================== COMBINED TEMPORAL ANALYSIS EXAMPLE ====================

print("\n🔍 Combined Temporal Analysis Example:")
print("=" * 50)

# Example: Analyze a specific patient's trajectory
if len(patient_ids) > 0:
    sample_patient = patient_ids[0]
    print(f"\nAnalyzing patient: {sample_patient}")
    
    # Get all sequences for this patient
    patient_seq_indices = [i for i, pid in enumerate(patient_ids) if pid == sample_patient]
    
    if patient_seq_indices:
        patient_seqs = X_patient_seq[patient_seq_indices]
        patient_preds = loaded_model3.predict(patient_seqs, verbose=0)
        patient_errors = np.mean(np.square(patient_seqs - patient_preds), axis=(1, 2))
        
        print(f"  Number of sequences: {len(patient_seqs)}")
        print(f"  Reconstruction errors: {patient_errors}")
        print(f"  Anomaly flags: {patient_errors > loaded_meta3['threshold']}")
        
        if np.any(patient_errors > loaded_meta3['threshold']):
            print("  ⚠️  Anomalous patient trajectory detected!")

# Example: Analyze a specific facility's recent weeks
if len(facility_ids) > 0:
    sample_facility = facility_ids[0]
    print(f"\nAnalyzing facility: {sample_facility}")
    
    # Get all sequences for this facility
    facility_seq_indices = [i for i, fid in enumerate(facility_ids) if fid == sample_facility]
    
    if facility_seq_indices:
        facility_seqs = X_facility_seq[facility_seq_indices]
        facility_preds = loaded_model4.predict(facility_seqs, verbose=0)
        facility_errors = np.mean(np.square(facility_seqs - facility_preds), axis=(1, 2))
        
        print(f"  Number of sequences: {len(facility_seqs)}")
        print(f"  Reconstruction errors: {facility_errors}")
        print(f"  Anomaly flags: {facility_errors > loaded_meta4['threshold']}")
        
        if np.any(facility_errors > loaded_meta4['threshold']):
            print("  ⚠️  Anomalous facility behavior detected!")

# ==================== SUMMARY ====================

print("\n" + "="*60)
print("✅✅✅ ALL TEMPORAL MODELS TRAINED AND TESTED SUCCESSFULLY ✅✅✅")
print("="*60)
print(f"""
Model 3: Patient Temporal LSTM Autoencoder
  - Input shape: (None, {PATIENT_SEQ_LEN}, {N_PATIENT_FEATURES})
  - Output shape: (None, {PATIENT_SEQ_LEN}, {N_PATIENT_FEATURES})
  - Latent dimension: {PATIENT_LATENT_DIM}
  - Sequences: {len(X_patient_seq)} from {len(np.unique(patient_ids))} patients
  - Anomaly threshold: {patient_threshold:.6f}
  - Best validation loss: {min(history3.history['val_loss']):.6f}

Model 4: Facility Temporal LSTM Autoencoder
  - Input shape: (None, {FACILITY_SEQ_LEN}, {N_FACILITY_FEATURES})
  - Output shape: (None, {FACILITY_SEQ_LEN}, {N_FACILITY_FEATURES})
  - Latent dimension: {FACILITY_LATENT_DIM}
  - Sequences: {len(X_facility_seq)} from {len(np.unique(facility_ids))} facilities
  - Features per week: {N_FACILITY_FEATURES}
  - Anomaly threshold: {facility_threshold:.6f}
  - Best validation loss: {min(history4.history['val_loss']):.6f}

Files saved in {MODEL_DIR}:
  - model3_patient_temporal_ae.keras
  - model3_patient_temporal_meta.json
  - model3_patient_scaler.pkl
  - model4_facility_temporal_ae.keras
  - model4_facility_temporal_meta.json
  - model4_facility_scaler.pkl
""")