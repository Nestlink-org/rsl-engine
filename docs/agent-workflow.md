# RSL Agent Workflow — Complete Technical Reference

ResultShield Lite multi-agent pipeline: from raw file upload to fraud prediction scores stored in the database.

---

## System Overview

```
Client (HTTP)
     │
     ▼
FastAPI Routes (/api/v1/agent/*)
     │
     ├── POST /agent/structure/preview  ── dry-run, no DB write
     ├── POST /agent/process            ── async job (202)
     ├── POST /agent/process/sync       ── synchronous, returns results inline
     ├── GET  /agent/queue/{job_id}     ── Redis queue depth
     └── POST /agent/queue/{job_id}/next── dequeue + process next batch
          │
          ▼
     job_worker.py  (background task)
          │
          ▼
     graph.py  (LangGraph pipeline)
          │
     ┌────┴────────────────────────────────────┐
     │                                         │
  ocr_node                              structuring_node (CSV/Excel)
  (PDF/image)                           pandas column normalisation
     │                                         │
     └────────────┬────────────────────────────┘
                  ▼
          structuring_node
          (gpt-5.4-nano for OCR text)
                  │
                  ▼
          validation_node
          (field checks, ranges, dates)
                  │
                  ▼
          inference_node (placeholder)
                  │
                  ▼
          orchestrator.py
          (model dispatch + DB write)
                  │
          ┌───────┼───────┬───────┐
          ▼       ▼       ▼       ▼
       Model1  Model2  Model3  Model4
          │       │       │       │
          └───────┴───────┴───────┘
                  │
          persistence.py
          (Claim, CBCData, FraudFlag,
           PatientTrajectory,
           FacilityWeeklyMetric)
                  │
                  ▼
            PostgreSQL DB
```

---

## Input Sources

Three file types enter the pipeline through the same endpoints:

| File Type | Extensions | Processing Path |
|---|---|---|
| Structured | `.csv`, `.xlsx`, `.xls` | pandas → column normalisation → structuring_node |
| Scanned document | `.pdf` | PaddleOCR (multi-page) → gpt-5.4-nano → structuring_node |
| Scanned image | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff` | PaddleOCR (single-page) → gpt-5.4-nano → structuring_node |

---

## Phase 1 — Structuring Pipeline

### Step 1: File Upload

`POST /api/v1/agent/process` or `POST /api/v1/upload`

- File saved to `uploads/{job_id}.{ext}`
- Job record created in DB (`status=pending`)
- `process_job()` fired as background task (asyncio)
- HTTP 202 returned immediately with `job_id`

### Step 2: OCR Node (`app/agent/nodes.py → ocr_node`)

Only runs for PDF and image files. CSV/Excel skips directly to structuring.

```
PDF input
  └── ocr_service.process_pdf(file_path)
        └── pdf_handler.pdf_to_images()     ← PyMuPDF: PDF → numpy arrays
              └── PaddleOCR.ocr(image)      ← per page
                    └── TextBlock list
                          └── full_text = "\n".join(block.text for block in blocks)

Image input (.jpg/.png/etc)
  └── ocr_service.process_image(file_path)
        └── PaddleOCR.ocr(image_path)
              └── TextBlock list
                    └── full_text
```

Redis cache check happens before OCR runs:
- Key: `ocr:result:{sha256_of_file}:p1` (image) or `:pdf` (PDF)
- TTL: 24 hours
- Cache hit → skip OCR entirely, return cached blocks

### Step 3: Structuring Node (`app/agent/nodes.py → structuring_node`)

Routes to one of two sub-paths:

**Path A — CSV/Excel (no LLM)**

```
structuring_agent.structure_from_file(file_path)
  └── pandas.read_csv() / read_excel()
        └── _normalise_columns()     ← maps aliases to canonical names
              │  e.g. "haemoglobin" → "HGB", "los" → "length_of_stay"
              └── _row_to_claim()    ← per row → claim dict
                    └── _encode_sex()     ← "Male"/"Female" → sex_encoded 0/1
                          └── _calc_los() ← discharge - admission if missing
```

**Path B — OCR text (gpt-5.4-nano)**

```
structuring_agent.structure_from_ocr(full_text)
  └── AzureChatOpenAI(deployment="gpt-5.4-nano")
        └── OCR_EXTRACTION_PROMPT
              │  "Extract ALL medical claims from OCR text.
              │   Return JSON array with CBC or HBA1C fields.
              │   Missing fields → null."
              └── JSON array of claim dicts
                    └── _encode_sex() + _calc_los()
```

Both paths then attach model validation to every claim:

```
detect_claim_type(claim)          ← "cbc" | "hba1c" | "unknown"
validate_model_inputs(claim, type) ← checks which models have all required features
summarise_validation(result)       ← human-readable eligibility summary
```

**Batch limit enforcement:**

```
all_claims = [60 rows from CSV]
batch = all_claims[:5]            ← AGENT_BATCH_SIZE = 5
overflow = all_claims[5:]         ← 55 claims
queue_overflow_claims(job_id, overflow)
  └── Redis RPUSH rsl:batch_queue:{job_id}
        └── TTL 24h
```

### Step 4: Validation Node (`app/agent/nodes.py → validation_node`)

Per claim:

```
validate_claim(claim, seen_ids)
  ├── required fields present?
  ├── duplicate claim_id?
  ├── sex value valid? ("Male"/"Female"/"0"/"1")
  ├── numeric ranges:
  │     HGB: 1–25, HCT: 5–70, MCV: 50–150, MCHC: 20–40
  │     NEU/LYM: 0–100, EOS: 0–50, BAS: 0–10, MON: 0–30
  │     PLT: 10–1500, age: 0–120
  └── admission_date <= discharge_date?

Pass → validated_claims[]
Fail → failed_claims[{claim_id, reason}]
```

Claims with no eligible models still pass through with `_validation_warning` — the orchestrator decides what to do with them.

---

## Phase 2 — Inference & Persistence

### Step 5: Orchestrator (`app/agent/orchestrator.py`)

Iterates over `validated_claims`. For each claim:

1. Checks `_model_validation.models` to see which models are eligible
2. Fetches patient history from `PatientTrajectory` table (for Model 3)
3. Builds facility weekly sequence from DB + current batch (for Model 4)
4. Dispatches eligible tools via `loop.run_in_executor` (non-blocking)
5. Calls `persist_claim_results()` after all tools complete

```
for claim in validated_claims:
    if claim._model_validation.models.model1.eligible:
        m1 = run_in_executor(run_cbc_model1, claim, registry)
    if claim._model_validation.models.model2.eligible:
        m2 = run_in_executor(run_cbc_model2, claim, registry)
    if claim._model_validation.models.model3.eligible:
        history = await _get_patient_history(session, patient_id)
        m3 = run_in_executor(run_cbc_model3, claim, history, registry)
    if registry.model4_available:
        fseq = db_weekly_history + batch_weekly_aggregates
        m4 = run_in_executor(run_cbc_model4, fseq, facility_id, registry)

    persist_claim_results(session, claim, {m1, m2, m3, m4})
```

---

## The Four CBC Models

### Model 1 — Claim Autoencoder (`cbc_model1_claim_autoencoder.keras`)

**Purpose:** detect anomalous CBC patterns at the individual claim level

```
Input (12 features):
  [age, sex_encoded, HGB, HCT, MCV, MCHC, NEU, LYM, EOS, BAS, MON, PLT]

Pipeline:
  x → MinMaxScaler → autoencoder.predict(x) → x_reconstructed
  mse = mean((x_scaled - x_reconstructed)²)
  anomaly_score = min(1.0, mse / threshold)   threshold = 5.004e-05
  is_anomaly = anomaly_score >= 0.5

Output:
  anomaly_score: 0.0–1.0
  is_anomaly: bool
  severity: low / medium / high
  top_features: top 5 features by reconstruction error
```

Files: `models/cbc_model1_claim_autoencoder.keras`, `models/cbc_model1_scaler.pkl`

---

### Model 2 — Disease Classifier (`cbc_model2_hierarchical_classifier.keras`)

**Purpose:** predict disease category and diagnosis from CBC labs, flag mismatches with claimed diagnosis

```
Input (10 features):
  [HGB, HCT, MCV, MCHC, NEU, LYM, EOS, BAS, MON, PLT]

Pipeline:
  x → StandardScaler → multi-task model.predict(x)
  → [category_probs (3), diagnosis_probs (5)]

Categories: obstetric | respiratory | trauma
Diagnoses:  APH PPH | ASTHMA | PNEUMONIA | PUERPERAL SEPSIS | TBI

Mismatch detection:
  if claimed_diagnosis.upper() != predicted_diagnosis.upper():
      anomaly_score = min(1.0, (1 - diag_confidence) + 0.3)
      is_anomaly = True
  else:
      anomaly_score = 1 - diag_confidence
      is_anomaly = anomaly_score > 0.5

Output:
  predicted_category, predicted_diagnosis
  category_confidence, diagnosis_confidence
  mismatch: bool
  anomaly_score, is_anomaly, severity
```

Files: `models/cbc_model2_hierarchical_classifier.keras`, `models/cbc_model2_scaler.pkl`, `models/cbc_model2_category_encoder.pkl`, `models/cbc_model2_diagnosis_encoder.pkl`

---

### Model 3 — Patient Temporal LSTM (`cbc_model3_patient_temporal_ae.keras`)

**Purpose:** detect anomalous patient visit trajectories over time

```
Input: sequence of 5 visits, each with 13 features:
  [age, sex_encoded, HGB, HCT, MCV, MCHC, NEU, LYM, EOS, BAS, MON, PLT, length_of_stay]

History assembly:
  patient_history = PatientTrajectory.visit_sequence (last 4 visits from DB)
  sequence = patient_history[-4:] + [current_visit]
  if len(sequence) < 5: zero-pad from the left

Pipeline:
  x (5, 13) → flatten → StandardScaler → reshape (1, 5, 13)
  → LSTM autoencoder.predict → x_reconstructed (1, 5, 13)
  per_visit_mse = mean((x_scaled - x_recon)², axis=features)  → shape (5,)
  overall_mse = mean(per_visit_mse)
  trajectory_score = min(1.0, overall_mse / threshold)   threshold = 0.2952
  is_trajectory_anomaly = trajectory_score >= 0.5

Output:
  trajectory_anomaly_score: 0.0–1.0
  is_trajectory_anomaly: bool
  per_visit_errors: [float × 5]
  most_anomalous_visit_index: int
  insufficient_history: bool (True if < 4 prior visits)
```

Files: `models/cbc_model3_patient_temporal_ae.keras`, `models/cbc_model3_patient_scaler.pkl`

---

### Model 4 — Facility Temporal LSTM (`model4_facility_temporal_ae.keras`)

**Purpose:** detect anomalous facility-level billing patterns over 8 weeks

```
Input: sequence of 8 weekly aggregates, each with 19 features:
  [claim_volume, avg_age, age_std, pct_male,
   HGB_mean, HGB_std, HCT_mean, HCT_std,
   MCV_mean, MCV_std, MCHC_mean, MCHC_std,
   NEU_mean, LYM_mean, EOS_mean, BAS_mean, MON_mean, PLT_mean,
   avg_los]

Weekly sequence assembly:
  db_history = FacilityWeeklyMetric (last 8 weeks from DB)
  batch_weeks = aggregate current batch claims by week
  combined = (db_history + batch_weeks)[-8:]
  if len < 8: zero-pad from the left

Pipeline:
  x (8, 19) → flatten → DataFrame → StandardScaler → nan_to_num(0)
  → reshape (1, 8, 19) → LSTM autoencoder.predict → x_reconstructed
  per_week_mse = mean((x_scaled - x_recon)², axis=features)  → shape (8,)
  overall_mse = mean(per_week_mse)
  facility_score = min(1.0, overall_mse / threshold)   threshold = 1.2420
  is_facility_anomaly = facility_score >= 0.5

Output:
  facility_anomaly_score: 0.0–1.0
  is_facility_anomaly: bool
  per_week_errors: [float × 8]
  most_anomalous_week_index: int
  insufficient_history: bool (True if < 8 weeks of history)
```

Files: `models/model4_facility_temporal_ae.keras`, `models/model4_facility_scaler.pkl`

---

## Model Registry — Singleton Load Chain

```
app/core/config.py
  MODELS_DIR = "models"          ← from .env
        │
        ▼
app/services/model_registry.py
  ModelRegistry.load(MODELS_DIR)
    tf.keras.models.load_model("models/cbc_model1_claim_autoencoder.keras")   → .model1
    joblib.load("models/cbc_model1_scaler.pkl")                               → .model1_scaler
    tf.keras.models.load_model("models/cbc_model2_hierarchical_classifier.keras") → .model2
    joblib.load("models/cbc_model2_scaler.pkl")                               → .model2_scaler
    joblib.load("models/cbc_model2_category_encoder.pkl")                     → .model2_category_encoder
    joblib.load("models/cbc_model2_diagnosis_encoder.pkl")                    → .model2_diagnosis_encoder
    tf.keras.models.load_model("models/cbc_model3_patient_temporal_ae.keras") → .model3
    joblib.load("models/cbc_model3_patient_scaler.pkl")                       → .model3_scaler
    tf.keras.models.load_model("models/model4_facility_temporal_ae.keras")    → .model4
    joblib.load("models/model4_facility_scaler.pkl")                          → .model4_scaler

  @lru_cache(maxsize=1)
  get_model_registry() → same instance reused for all requests
        │
        ├── app/main.py lifespan startup
        │     loop.run_in_executor(None, get_model_registry)
        │     (loaded once, cached, never reloaded)
        │
        ├── app/workers/job_worker.py
        │     registry = await loop.run_in_executor(None, get_model_registry)
        │
        └── app/routes/agent.py (sync endpoint)
              registry = await loop.run_in_executor(None, get_model_registry)
```

---

## Parallel Inference Paths

Two code paths call the same registry. Both are active:

```
Path A — Legacy CSV pipeline (app/services/csv_pipeline.py)
  run_csv_pipeline()
    └── run_inference(claim, history, registry)   ← inference_service.py
          ├── run_model1(features, registry)       ← returns Model1Result dataclass
          ├── run_model2(features, diag, registry) ← returns Model2Result dataclass
          └── run_model3(sequence, registry, ...)  ← returns Model3Result dataclass
          (asyncio.gather — concurrent)

Path B — Agent pipeline (app/agent/tools/cbc_tools.py)
  run_orchestrator()
    └── run_cbc_model1(claim, registry)   ← returns dict with top_features
    └── run_cbc_model2(claim, registry)   ← returns dict with mismatch flag
    └── run_cbc_model3(claim, history, registry) ← returns dict with per_visit_errors
    └── run_cbc_model4(weekly_seq, fid, registry) ← returns dict with per_week_errors
    (loop.run_in_executor — non-blocking)
```

Both paths use the same `.keras` model files via the same registry singleton.
Path B (agent) produces richer output — top contributing features, per-visit/per-week error arrays, mismatch detection.

---

## DB Persistence (`app/agent/tools/persistence.py`)

After all models run for a claim:

```
persist_claim_results(session, claim, model_results, job_id, user_id)
  │
  ├── INSERT claim → table: claim
  │     claim_id, job_id, user_id, patient_id, facility_id,
  │     admission_date, discharge_date, claimed_diagnosis
  │
  ├── INSERT CBCData → table: cbcdata
  │     (only if model1 or model3 eligible)
  │     age, sex_encoded, HGB, HCT, MCV, MCHC, NEU, LYM, EOS, BAS, MON, PLT, length_of_stay
  │
  ├── INSERT FraudFlag × N → table: fraudflag
  │     (one row per model that ran)
  │     model_id, anomaly_score, is_anomaly, severity, flag_reason
  │     model2 extras: predicted_category, predicted_diagnosis, confidences
  │     model3 extras: insufficient_history
  │
  ├── UPSERT PatientTrajectory → table: patienttrajectory
  │     visit_sequence (rolling JSON, last 5 visits)
  │     trajectory_anomaly_score, is_trajectory_anomaly
  │     per_visit_errors, most_anomalous_visit_index
  │
  └── UPSERT FacilityWeeklyMetric → table: facilityweeklymetric
        claim_volume += 1
        avg_anomaly_score (running average)
        flagged_claims, high/medium/low_severity_count
```

---

## Batch Queue (Redis)

For files with more than `AGENT_BATCH_SIZE` (default 5) claims:

```
structure_from_file() → 60 rows loaded
  batch = rows[0:5]    → processed immediately
  overflow = rows[5:]  → pushed to Redis

Redis key: rsl:batch_queue:{job_id}
  RPUSH claim_json × 55
  EXPIRE 86400 (24h)

GET /api/v1/agent/queue/{job_id}
  → {"queued_claims": 55, "batch_size": 5}

POST /api/v1/agent/queue/{job_id}/next
  → LPOP × 5 from Redis
  → validation_node → orchestrator → DB
  → {"queued_claims": 50, ...}

Repeat until queued_claims = 0
```

---

## Complete End-to-End Flow: CSV File (60 claims)

```
1. POST /api/v1/agent/process  (file=claims.csv)
   └── create_job(status=pending) → job_id=abc123
   └── save file → uploads/abc123.csv
   └── asyncio.create_task(process_job(abc123, ...))
   └── HTTP 202 {"job_id": "abc123", "status_url": "/api/v1/jobs/abc123/status"}

2. process_job(abc123)
   └── update_job_status(processing)
   └── get_model_registry()  ← loads .keras files if not cached

3. graph.py → run_pipeline(abc123.csv, "csv")
   └── ocr_node: SKIP (csv)
   └── structuring_node:
         structure_from_file("uploads/abc123.csv")
           pandas.read_csv → 60 rows
           _normalise_columns → canonical field names
           per row: _encode_sex, _calc_los, detect_claim_type, validate_model_inputs
           batch = rows[0:5], overflow = rows[5:55]
           Redis RPUSH 55 claims → rsl:batch_queue:abc123
   └── validation_node:
         validate_claim × 5 → 5 validated
   └── inference_node: (placeholder, passes through)

4. orchestrator.py → run_orchestrator(5 claims)
   └── for each claim:
         run_cbc_model1 → anomaly_score
         run_cbc_model2 → predicted_diagnosis, mismatch
         run_cbc_model3 → trajectory_score
         run_cbc_model4 → facility_score (if model4_available)
         persist_claim_results → INSERT claim, cbcdata, fraudflag × 4
         UPSERT patienttrajectory
         UPSERT facilityweeklymetric

5. update_job_status(partial, processed=5, failed=0)

6. GET /api/v1/agent/queue/abc123 → {"queued_claims": 55}

7. POST /api/v1/agent/queue/abc123/next  (repeat 11 times)
   └── LPOP 5 claims from Redis
   └── validation → orchestrator → DB
   └── {"queued_claims": 50} ... {"queued_claims": 0}
```

---

## Complete End-to-End Flow: Scanned CBC Image

```
1. POST /api/v1/agent/process/sync  (file=cbc_claim.jpeg)
   └── create_job → job_id=xyz789
   └── save → uploads/agent_xyz789.jpeg

2. run_pipeline(agent_xyz789.jpeg, "jpeg")
   └── ocr_node:
         Redis cache check → MISS (first time)
         PaddleOCR.ocr("agent_xyz789.jpeg")
           → 145 TextBlock objects
           → full_text = "Haemoglobin : 9.10 [L]\nTotal W.B.C.Count 10560[H]\n..."
         Redis SET ocr:result:{sha256}:p1  TTL=24h

   └── structuring_node:
         structure_from_ocr(full_text)
           gpt-5.4-nano prompt:
             "Extract ALL medical claims from OCR text.
              Return JSON array with CBC fields.
              Missing fields → null."
           → [{"claim_id": "uuid", "HGB": 9.1, "HCT": 27.2, "PLT": 370,
               "age": null, "sex": null, ...}]
           detect_claim_type → "cbc"
           validate_model_inputs:
             model1: INELIGIBLE (missing age, sex_encoded)
             model2: ELIGIBLE   (only needs 10 CBC labs)
             model3: INELIGIBLE (missing age, sex_encoded, length_of_stay)

   └── validation_node:
         no eligible models → passes through with _validation_warning

   └── orchestrator:
         model2 only → run_cbc_model2(claim, registry)
           HGB=9.1, HCT=27.2, ... → StandardScaler → model2.predict
           → predicted_category=obstetric, predicted_diagnosis=APH PPH, conf=0.87
           → mismatch check vs claimed_diagnosis
         persist_claim_results:
           INSERT claim (no CBCData — model1/3 not eligible)
           INSERT fraudflag (model_id=2 only)

3. HTTP 200 {
     "job_id": "xyz789",
     "status": "completed",
     "total_processed": 1,
     "models_triggered": {"model2": 1},
     "results": [{
       "claim_id": "uuid",
       "models_run": ["model2"],
       "model_results": {
         "model2": {
           "predicted_category": "obstetric",
           "predicted_diagnosis": "APH PPH",
           "diagnosis_confidence": 0.87,
           "mismatch": false,
           "anomaly_score": 0.13,
           "is_anomaly": false
         }
       }
     }]
   }
```

---

## Model Eligibility Rules

A model only runs if ALL its required features are present and numeric in the structured claim.

| Model | Required Features | Minimum for Eligibility |
|---|---|---|
| Model 1 | age, sex_encoded + 10 CBC labs | 12 features |
| Model 2 | 10 CBC labs only | 10 features |
| Model 3 | age, sex_encoded + 10 CBC labs + length_of_stay | 13 features |
| Model 4 | facility weekly aggregates (not per-claim) | 8 weeks of history |

A claim is never hard-rejected for missing features — it passes through with a warning and only the eligible models run. This means a scanned image that only captures lab values (no patient demographics) will still trigger Model 2.

---

## Source Files

| File | Role |
|---|---|
| `app/routes/agent.py` | HTTP endpoints, Swagger docs, file validation |
| `app/workers/job_worker.py` | Background task dispatcher |
| `app/agent/graph.py` | LangGraph pipeline definition and runner |
| `app/agent/nodes.py` | ocr_node, structuring_node, validation_node |
| `app/agent/structuring_agent.py` | CSV/Excel pandas path + OCR LLM path, Redis batch queue |
| `app/agent/model_validator.py` | Per-claim model eligibility checker |
| `app/agent/orchestrator.py` | Model dispatch, facility sequence builder, DB write coordinator |
| `app/agent/tools/cbc_tools.py` | run_cbc_model1/2/3/4 — direct model inference functions |
| `app/agent/tools/persistence.py` | DB write: Claim, CBCData, FraudFlag, PatientTrajectory, FacilityWeeklyMetric |
| `app/agent/state.py` | LangGraph PipelineState TypedDict |
| `app/agent/config.py` | get_llm() factory — gpt-5.4-nano via AzureChatOpenAI |
| `app/services/model_registry.py` | Loads all .keras + .pkl files, @lru_cache singleton |
| `app/services/inference_service.py` | Legacy inference path (used by csv_pipeline.py) |
| `app/services/csv_pipeline.py` | Legacy CSV pipeline (still active, parallel to agent path) |
| `app/services/ocr_service.py` | PaddleOCR wrapper with Redis caching |
| `app/utils/pdf_handler.py` | PDF → numpy image arrays via PyMuPDF |
