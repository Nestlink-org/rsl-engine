# Requirements Document

## Introduction

ResultShield Lite (RSL) is a fraud detection engine for Kenyan insurance companies processing CBC/FBC medical claims. This document covers the REST API endpoints only — no frontend. The system accepts claim data via CSV/Excel/PDF upload, runs it through a multi-model ML inference pipeline, persists results to PostgreSQL, and exposes endpoints for querying jobs, claims, facility risk, patient trajectories, dashboard metrics, and ROI reports.

The existing OCR service (`app/services/ocr_service.py`) is already working and must not be modified. PDF claims flow through a LangGraph agent pipeline (OCR → LLM structuring → validation → model inference). CSV/Excel claims bypass OCR and go directly to validation and model inference.

---

## Glossary

- **RSL_API**: The FastAPI application exposing all `/api/v1/` endpoints.
- **Upload_Service**: The component that receives uploaded files, validates them, enqueues a job, and returns a `job_id`.
- **Job_Queue**: The Redis-backed async queue that tracks job state transitions: `pending → processing → completed / failed / partial`.
- **LangGraph_Pipeline**: The four-node agent graph (OCR Agent → LLM_Structuring_Agent → Validation_Agent → Model_Inference_Agent) used for PDF claim processing.
- **CSV_Pipeline**: The direct path for CSV/Excel files: pandas parsing → Validation_Agent → Model_Inference_Agent.
- **Validation_Agent**: The component that checks CBC field presence, numeric ranges, date ordering, duplicate claim IDs, and sex encoding.
- **Model_Inference_Agent**: The component that runs Model 1, Model 2, and Model 3 in parallel via `asyncio.gather`, and gracefully skips Model 4 when unavailable.
- **Model_Registry**: The singleton loader that loads all available Keras models, scalers, and encoders once at application startup.
- **Claim**: A single medical claim record with patient demographics, CBC lab values, admission/discharge dates, facility ID, and claimed diagnosis.
- **CBCData**: The structured CBC lab values linked to a Claim.
- **FraudFlag**: A per-model fraud detection result linked to a Claim, containing anomaly score, severity, and flag reason.
- **PatientTrajectory**: An aggregated sequence of up to 5 visits per patient used by Model 3.
- **FacilityWeeklyMetric**: Weekly aggregate statistics per facility used for facility-level risk scoring.
- **OCR_Service**: The existing PaddleOCR singleton (`app/services/ocr_service.py`) — must not be modified.
- **LLM_Structuring_Agent**: A LangChain/OpenAI GPT-4 agent that converts raw OCR text into structured Claim fields.
- **Dashboard**: The aggregated metrics view covering daily, weekly, and monthly fraud statistics.
- **ROI_Report**: A computed report showing estimated savings from fraud detection over a given period.
- **job_id**: A UUID string identifying an async processing job.
- **claim_id**: A unique string identifier for a single claim, sourced from the uploaded data.
- **facility_id**: A string identifier for a healthcare facility.
- **patient_id**: A string identifier for a patient, used to aggregate trajectory data.

---

## Requirements

### Requirement 1: File Upload and Job Creation

**User Story:** As an insurance analyst, I want to upload a CSV, Excel, PDF, or scanned image file of CBC claims, so that the system can process them asynchronously and return a job ID I can use to track progress.

#### Acceptance Criteria

1. WHEN a `POST /api/v1/upload` request is received with a valid file attachment, THE Upload_Service SHALL validate the file type as one of `.csv`, `.xlsx`, `.xls`, `.pdf`, `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.tiff`.
2. WHEN the uploaded file size exceeds 50 MB, THE Upload_Service SHALL return HTTP 413 with a descriptive error message.
3. WHEN the uploaded file has an unsupported extension, THE Upload_Service SHALL return HTTP 422 with a descriptive error message.
4. WHEN a valid file is received, THE Upload_Service SHALL save the file to the configured upload directory and create a Job record in the database with status `pending`.
5. WHEN a Job record is created, THE Upload_Service SHALL enqueue the job in the Job_Queue and return HTTP 202 with a JSON body containing `job_id`, `status: "pending"`, and `filename`.
6. THE Upload_Service SHALL assign each job a UUID `job_id` that is unique across all jobs.
7. WHEN a CSV or Excel file is uploaded, THE Upload_Service SHALL route the job to the CSV_Pipeline.
8. WHEN a PDF file is uploaded, THE Upload_Service SHALL route the job to the LangGraph_OCR_Pipeline.
9. WHEN an image file (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`) is uploaded, THE Upload_Service SHALL route the job to the LangGraph_OCR_Pipeline using single-page image OCR.

---

### Requirement 2: Job Status Tracking

**User Story:** As an insurance analyst, I want to poll the status of a submitted job, so that I know when processing is complete or if it has failed.

#### Acceptance Criteria

1. WHEN a `GET /api/v1/jobs/{job_id}/status` request is received with a valid `job_id`, THE RSL_API SHALL return the current job status from the Job_Queue.
2. THE RSL_API SHALL return a JSON body containing `job_id`, `status`, `created_at`, `updated_at`, `total_claims`, `processed_claims`, and `failed_claims`.
3. WHEN the `job_id` does not exist in the database, THE RSL_API SHALL return HTTP 404 with a descriptive error message.
4. THE Job_Queue SHALL support exactly four status values: `pending`, `processing`, `completed`, and `failed`.
5. WHEN at least one claim in a job fails validation or inference but others succeed, THE Job_Queue SHALL set the job status to `partial` and record the count of failed claims.
6. WHEN all claims in a job fail processing, THE Job_Queue SHALL set the job status to `failed`.

---

### Requirement 3: Job Results Retrieval

**User Story:** As an insurance analyst, I want to retrieve the fraud detection results for a completed job, so that I can review which claims were flagged.

#### Acceptance Criteria

1. WHEN a `GET /api/v1/jobs/{job_id}/results` request is received and the job status is `completed` or `partial`, THE RSL_API SHALL return a paginated list of claims with their associated FraudFlags.
2. WHEN the job status is `pending` or `processing`, THE RSL_API SHALL return HTTP 202 with a message indicating processing is not yet complete.
3. WHEN the `job_id` does not exist, THE RSL_API SHALL return HTTP 404 with a descriptive error message.
4. THE RSL_API SHALL support `page` and `page_size` query parameters for the results endpoint, with a default `page_size` of 50 and a maximum `page_size` of 500.
5. THE RSL_API SHALL include in each result item: `claim_id`, `patient_id`, `facility_id`, `admission_date`, `discharge_date`, `claimed_diagnosis`, and a `fraud_flags` array containing one entry per model that ran.
6. WHEN a model did not run for a claim (e.g., Model 4 is unavailable), THE RSL_API SHALL omit that model's entry from the `fraud_flags` array rather than returning null values.

---

### Requirement 4: Individual Claim Detail

**User Story:** As a fraud investigator, I want to retrieve the full detail of a single claim including all fraud flags, so that I can make an adjudication decision.

#### Acceptance Criteria

1. WHEN a `GET /api/v1/claims/{claim_id}` request is received with a valid `claim_id`, THE RSL_API SHALL return the full Claim record including all CBCData fields and all associated FraudFlags.
2. WHEN the `claim_id` does not exist in the database, THE RSL_API SHALL return HTTP 404 with a descriptive error message.
3. THE RSL_API SHALL include in the response: all 12 CBC input features (`age`, `sex_encoded`, `HGB`, `HCT`, `MCV`, `MCHC`, `NEU`, `LYM`, `EOS`, `BAS`, `MON`, `PLT`), `length_of_stay`, `admission_date`, `discharge_date`, `facility_id`, `patient_id`, `claimed_diagnosis`, and a `fraud_flags` array.
4. THE RSL_API SHALL include in each FraudFlag entry: `model_id` (1, 2, or 3), `anomaly_score` (0.0–1.0), `is_anomaly` (bool), `severity` (`low`, `medium`, or `high`), and `flag_reason` (human-readable string).
5. WHEN Model 2 ran for the claim, THE RSL_API SHALL include `predicted_category`, `predicted_diagnosis`, `category_confidence`, and `diagnosis_confidence` in the Model 2 FraudFlag entry.
6. WHEN the claimed diagnosis does not match the Model 2 predicted diagnosis, THE RSL_API SHALL set `flag_reason` to a string describing the mismatch, including both the claimed and predicted values.

---

### Requirement 5: Facility Risk Metrics

**User Story:** As a risk manager, I want to view aggregated fraud risk metrics for a specific facility, so that I can identify high-risk providers.

#### Acceptance Criteria

1. WHEN a `GET /api/v1/facilities/{facility_id}/risk` request is received with a valid `facility_id`, THE RSL_API SHALL return aggregated risk metrics for that facility.
2. WHEN the `facility_id` has no claims in the database, THE RSL_API SHALL return HTTP 404 with a descriptive error message.
3. THE RSL_API SHALL include in the response: `facility_id`, `total_claims`, `flagged_claims`, `flag_rate` (flagged_claims / total_claims), `avg_anomaly_score`, `high_severity_count`, `medium_severity_count`, `low_severity_count`, and `weekly_metrics` (array of FacilityWeeklyMetric records).
4. THE RSL_API SHALL support an optional `weeks` query parameter (integer, 1–52) to limit the weekly_metrics window, defaulting to 8 weeks.
5. WHEN Model 4 (facility temporal model) is unavailable, THE RSL_API SHALL compute facility risk metrics from Models 1, 2, and 3 results only, and SHALL include a `model4_available: false` field in the response.

---

### Requirement 6: Patient Trajectory

**User Story:** As a fraud investigator, I want to view a patient's visit trajectory and temporal anomaly score, so that I can detect patterns of repeated fraudulent claims.

#### Acceptance Criteria

1. WHEN a `GET /api/v1/patients/{patient_id}/trajectory` request is received with a valid `patient_id`, THE RSL_API SHALL return the patient's visit sequence and Model 3 trajectory anomaly results.
2. WHEN the `patient_id` has fewer than 2 visits in the database, THE RSL_API SHALL return HTTP 422 with a message indicating insufficient visit history for trajectory analysis.
3. WHEN the `patient_id` does not exist in the database, THE RSL_API SHALL return HTTP 404 with a descriptive error message.
4. THE RSL_API SHALL include in the response: `patient_id`, `total_visits`, `trajectory_anomaly_score` (0.0–1.0), `is_trajectory_anomaly` (bool), `most_anomalous_visit_index` (0-based), `per_visit_errors` (array of floats), and `visits` (array of visit records with CBC values and dates).
5. THE RSL_API SHALL use the most recent 5 visits when a patient has more than 5 visits, consistent with Model 3's sequence length of 5.

---

### Requirement 7: Dashboard Metrics

**User Story:** As an insurance manager, I want a dashboard endpoint that returns aggregated fraud statistics, so that I can monitor system-wide fraud trends.

#### Acceptance Criteria

1. WHEN a `GET /api/v1/dashboard/metrics` request is received, THE RSL_API SHALL return aggregated fraud statistics computed from the database.
2. THE RSL_API SHALL support a `period` query parameter accepting values `daily`, `weekly`, or `monthly`, defaulting to `weekly`.
3. THE RSL_API SHALL include in the response: `period`, `total_claims_processed`, `total_flagged_claims`, `overall_flag_rate`, `total_estimated_fraud_amount`, `claims_by_severity` (counts for low/medium/high), `top_flagged_facilities` (top 10 by flag rate), and `trend_data` (array of time-bucketed counts).
4. WHEN the database contains no claims for the requested period, THE RSL_API SHALL return an empty metrics response with zero counts rather than HTTP 404.
5. THE RSL_API SHALL return dashboard metrics within 3 seconds for datasets up to 1 million claims by using pre-aggregated FacilityWeeklyMetric records and database indexes.

---

### Requirement 8: ROI Report Generation

**User Story:** As an insurance executive, I want to generate an ROI report showing the financial value of fraud detection, so that I can justify the system's cost.

#### Acceptance Criteria

1. WHEN a `POST /api/v1/reports/roi` request is received with a valid request body, THE RSL_API SHALL compute and return an ROI report as a JSON response.
2. THE RSL_API SHALL accept in the request body: `start_date` (ISO 8601 date), `end_date` (ISO 8601 date), `avg_claim_value_kes` (positive float, amount in Kenyan Shillings), and optionally `recovery_rate` (float 0.0–1.0, defaulting to 0.3).
3. WHEN `start_date` is after `end_date`, THE RSL_API SHALL return HTTP 422 with a descriptive error message.
4. THE RSL_API SHALL include in the response: `period_start`, `period_end`, `total_claims_reviewed`, `flagged_claims`, `estimated_fraud_amount_kes` (flagged_claims × avg_claim_value_kes), `estimated_recovered_kes` (estimated_fraud_amount_kes × recovery_rate), `roi_ratio` (estimated_recovered_kes / system_cost_kes where system_cost_kes defaults to 0 if not provided), and `flag_rate`.
5. WHERE a `format=pdf` query parameter is provided, THE RSL_API SHALL return the ROI report as a downloadable PDF file with `Content-Type: application/pdf`.

---

### Requirement 9: CSV/Excel Claim Processing Pipeline

**User Story:** As a data engineer, I want the system to parse CSV and Excel claim files directly, so that batch claims can be processed without OCR overhead.

#### Acceptance Criteria

1. WHEN the CSV_Pipeline receives a CSV or Excel file, THE CSV_Pipeline SHALL parse it using pandas and extract one Claim record per row.
2. THE CSV_Pipeline SHALL expect the following columns: `claim_id`, `patient_id`, `facility_id`, `admission_date`, `discharge_date`, `claimed_diagnosis`, `age`, `sex`, `HGB`, `HCT`, `MCV`, `MCHC`, `NEU`, `LYM`, `EOS`, `BAS`, `MON`, `PLT`, `length_of_stay`.
3. WHEN a required column is missing from the file, THE CSV_Pipeline SHALL mark the entire job as `failed` and return a descriptive error listing the missing columns.
4. WHEN an individual row fails validation, THE CSV_Pipeline SHALL record the row as a failed claim with a reason and continue processing remaining rows.
5. THE CSV_Pipeline SHALL process a batch of 10,000 claims within 10 minutes end-to-end including validation and model inference.

---

### Requirement 10: PDF and Image Claim Processing Pipeline

**User Story:** As a claims processor, I want to upload scanned PDF claim forms or photographed claim images, so that the system can extract and validate CBC data automatically.

#### Acceptance Criteria

1. WHEN the LangGraph_OCR_Pipeline receives a PDF file, THE LangGraph_OCR_Pipeline SHALL invoke the OCR_Service's `process_pdf` method to extract raw text blocks from all pages.
2. WHEN the LangGraph_OCR_Pipeline receives an image file (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`), THE LangGraph_OCR_Pipeline SHALL invoke the OCR_Service's `process_image` method to extract raw text blocks from the single image.
3. WHEN OCR extraction is complete, THE LangGraph_OCR_Pipeline SHALL pass the raw text to the LLM_Structuring_Agent to extract structured Claim fields.
4. WHEN the LLM_Structuring_Agent cannot extract a required field from the OCR text, THE LangGraph_OCR_Pipeline SHALL mark that claim as failed with reason `extraction_failed` and record the missing fields.
5. WHEN structured fields are extracted, THE Validation_Agent SHALL validate the claim before passing it to the Model_Inference_Agent.
6. THE LangGraph_OCR_Pipeline SHALL not modify or re-initialize the OCR_Service singleton.

---

### Requirement 11: Claim Validation

**User Story:** As a data quality engineer, I want all claims to be validated before model inference, so that models only receive well-formed inputs.

#### Acceptance Criteria

1. WHEN a claim is submitted for validation, THE Validation_Agent SHALL verify that all required CBC fields are present: `HGB`, `HCT`, `MCV`, `MCHC`, `NEU`, `LYM`, `EOS`, `BAS`, `MON`, `PLT`, `age`, `sex`.
2. THE Validation_Agent SHALL verify that `HGB` is in the range 1–25, `PLT` is in the range 10–1500, `HCT` is in the range 5–70, `MCV` is in the range 50–150, `MCHC` is in the range 20–40, `NEU` is in the range 0–100, `LYM` is in the range 0–100, `EOS` is in the range 0–50, `BAS` is in the range 0–10, `MON` is in the range 0–30, and `age` is in the range 0–120.
3. WHEN `admission_date` is after `discharge_date`, THE Validation_Agent SHALL reject the claim with reason `invalid_date_range`.
4. WHEN a `claim_id` already exists in the database, THE Validation_Agent SHALL reject the claim with reason `duplicate_claim_id`.
5. WHEN `sex` is not one of `Male`, `Female`, `1`, or `0`, THE Validation_Agent SHALL reject the claim with reason `invalid_sex_value`.
6. WHEN a claim fails validation, THE Validation_Agent SHALL record the failure reason and SHALL NOT pass the claim to the Model_Inference_Agent.

---

### Requirement 12: Model Inference

**User Story:** As a fraud detection engineer, I want all available models to run in parallel on each validated claim, so that inference is fast and multi-dimensional.

#### Acceptance Criteria

1. WHEN a validated claim is received, THE Model_Inference_Agent SHALL run Model 1, Model 2, and Model 3 concurrently using `asyncio.gather`.
2. THE Model_Inference_Agent SHALL complete inference for a single claim within 100 milliseconds.
3. WHEN Model 1 (claim autoencoder) runs, THE Model_Inference_Agent SHALL compute `anomaly_score` as `min(1.0, reconstruction_error / threshold)` using threshold `5.004e-05`, and SHALL set `is_anomaly` to `true` when `reconstruction_error > 5.004e-05`.
4. WHEN Model 2 (hierarchical classifier) runs, THE Model_Inference_Agent SHALL produce `predicted_category` (one of `obstetric`, `respiratory`, `trauma`) and `predicted_diagnosis` (one of `APH PPH`, `ASTHMA`, `PNEUMONIA`, `PUERPERAL SEPSIS`, `TBI`) with confidence scores.
5. WHEN Model 3 (patient temporal LSTM) runs, THE Model_Inference_Agent SHALL retrieve the patient's most recent visits from the PatientTrajectory table, scale the sequence, run inference, and compute `trajectory_anomaly_score` using threshold `0.2952`.
6. WHEN a patient has fewer than 5 prior visits, THE Model_Inference_Agent SHALL pad the sequence with the available visits and record `insufficient_history: true` in the Model 3 FraudFlag entry.
7. WHEN Model 4 files are not present in the models directory, THE Model_Inference_Agent SHALL skip Model 4 inference without raising an error and SHALL NOT include a Model 4 FraudFlag entry in the results.
8. THE Model_Registry SHALL load all available model files, scalers, and encoders exactly once during application startup using a singleton pattern.
9. WHEN severity classification is required, THE Model_Inference_Agent SHALL assign `high` when `anomaly_score > 0.8`, `medium` when `anomaly_score > 0.5`, and `low` otherwise.

---

### Requirement 13: Database Schema and Persistence

**User Story:** As a backend engineer, I want all claim data and fraud results persisted to PostgreSQL via SQLModel, so that results are durable and queryable.

#### Acceptance Criteria

1. THE RSL_API SHALL define the following SQLModel tables: `Claim`, `CBCData`, `FraudFlag`, `PatientTrajectory`, and `FacilityWeeklyMetric`.
2. THE RSL_API SHALL create Alembic migration scripts for all new tables before the application can be started.
3. ALL database operations SHALL use async SQLModel sessions backed by asyncpg.
4. THE `Claim` table SHALL have a unique constraint on `claim_id`.
5. THE `FraudFlag` table SHALL have a foreign key to `Claim` and a `model_id` integer field (1, 2, 3, or 4).
6. THE `PatientTrajectory` table SHALL store the most recent 5 visit feature vectors per `patient_id` as a JSON array, updated after each new claim is processed for that patient.
7. THE `FacilityWeeklyMetric` table SHALL store weekly aggregate statistics per `facility_id` and `week_start_date`, updated after each batch job completes.

---

### Requirement 14: API Structure and Configuration

**User Story:** As a backend engineer, I want the FastAPI application to be properly structured with versioned routes and environment-driven configuration, so that it is maintainable and deployable.

#### Acceptance Criteria

1. THE RSL_API SHALL mount all fraud detection endpoints under the `/api/v1/` prefix.
2. THE RSL_API SHALL preserve the existing `/ocr/test` endpoint without modification.
3. THE RSL_API SHALL load database connection settings (`PSQL_URI`), Redis settings (`REDIS_URL`), and OpenAI API key from environment variables via the existing `Settings` class in `app/core/config.py`.
4. THE RSL_API SHALL add the following settings to the `Settings` class: `PSQL_URI`, `REDIS_URL`, `OPENAI_API_KEY`, `MODELS_DIR` (defaulting to `"models"`), and `MAX_BATCH_SIZE` (defaulting to `10000`).
5. THE RSL_API SHALL return all error responses in the format `{"success": false, "error": "<message>", "detail": "<optional detail>"}` consistent with the existing `ErrorResponse` schema.
6. WHEN the application starts up, THE RSL_API SHALL initialize the database connection pool, the Redis connection, and the Model_Registry before accepting requests.
