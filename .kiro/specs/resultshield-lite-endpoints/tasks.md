# Implementation Plan: ResultShield Lite (RSL) — API Endpoints

## Overview

Implement the RSL fraud detection engine as a FastAPI microservice with PostgreSQL persistence, Redis job queue, Kafka integration, LangGraph pipelines (PDF + chat), multi-model ML inference, and a full REST API under `/api/v1/`. The existing OCR service and `/ocr/test` endpoint must not be modified.

---

## Tasks

- [x] 1. Foundation — Config, DB session, JWT middleware, Alembic setup
  - [x] 1.1 Extend `app/core/config.py` with new settings
    - Add `PSQL_URI: str`, `REDIS_URL: str`, `OPENAI_API_KEY: str`, `MODELS_DIR: str = "models"`, `MAX_BATCH_SIZE: int = 10000` to the `Settings` class
    - Add Kafka settings: `KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"`, `KAFKA_CONSUMER_GROUP: str = "rsl-engine"`
    - _Requirements: 14.3, 14.4_

  - [x] 1.2 Create `app/db/session.py` — async engine and session factory
    - Implement `create_async_engine` with `settings.PSQL_URI`, `pool_pre_ping=True`
    - Implement `AsyncSessionLocal` sessionmaker and `get_async_session()` async generator for FastAPI `Depends`
    - _Requirements: 13.3_

  - [x] 1.3 Create `app/middleware/jwt_middleware.py` — user_id extraction middleware
    - Implement `JWTMiddleware(BaseHTTPMiddleware)` that reads `Authorization: Bearer <jwt>` header
    - Decode JWT payload **without signature verification** using `jwt.decode(..., options={"verify_signature": False})`
    - Extract `user_id` (and optionally `role`) from payload; store on `request.state.user_id` and `request.state.role`
    - If header is absent, set `request.state.user_id = None`
    - _Requirements: 14.1_

  - [x] 1.4 Update `alembic/env.py` for SQLModel + psycopg2 sync migrations
    - Import `SQLModel` and all ORM model classes (to be created in task 2) so their metadata is registered
    - Set `target_metadata = SQLModel.metadata`
    - Replace `asyncpg` with `psycopg2` in the URL for both offline and online migration modes
    - _Requirements: 13.2_

- [x] 2. ORM Models — SQLModel table definitions
  - [x] 2.1 Create `app/models/__init__.py` and `app/models/job.py`
    - Define `Job` table: `id`, `job_id` (UUID str, unique, indexed), `user_id: str`, `filename`, `file_type`, `status` (default `"pending"`), `total_claims`, `processed_claims`, `failed_claims`, `error_detail`, `created_at`, `updated_at`
    - _Requirements: 13.1, 1.4_

  - [x] 2.2 Create `app/models/claim.py`
    - Define `Claim` table: `id`, `claim_id` (unique, indexed), `job_id` (indexed), `user_id: str`, `patient_id` (indexed), `facility_id` (indexed), `admission_date`, `discharge_date`, `claimed_diagnosis`, `created_at`
    - Define `CBCData` table: `id`, `claim_id` (FK → `claim.id`, indexed), `age`, `sex_encoded`, `HGB`, `HCT`, `MCV`, `MCHC`, `NEU`, `LYM`, `EOS`, `BAS`, `MON`, `PLT`, `length_of_stay`
    - _Requirements: 13.1, 13.4_

  - [x] 2.3 Create `app/models/fraud_flag.py`
    - Define `FraudFlag` table: `id`, `claim_id` (FK → `claim.id`, indexed), `model_id` (int 1–4), `anomaly_score`, `is_anomaly`, `severity`, `flag_reason`, `predicted_category`, `predicted_diagnosis`, `category_confidence`, `diagnosis_confidence`, `insufficient_history`, `created_at`
    - _Requirements: 13.1, 13.5_

  - [x] 2.4 Create `app/models/patient_trajectory.py`
    - Define `PatientTrajectory` table: `id`, `patient_id` (unique, indexed), `visit_sequence` (JSON str), `trajectory_anomaly_score`, `is_trajectory_anomaly`, `per_visit_errors` (JSON str), `most_anomalous_visit_index`, `last_updated`
    - _Requirements: 13.1, 13.6_

  - [x] 2.5 Create `app/models/facility_metric.py`
    - Define `FacilityWeeklyMetric` table: `id`, `facility_id` (indexed), `week_start_date` (indexed), `claim_volume`, `avg_anomaly_score`, `flagged_claims`, `high_severity_count`, `medium_severity_count`, `low_severity_count`
    - _Requirements: 13.1, 13.7_

  - [x] 2.6 Create `app/models/audit_log.py`
    - Define `AuditLog` table: `id`, `user_id: str`, `action: str`, `resource_type: str`, `resource_id: str`, `timestamp` (default `datetime.utcnow`), `metadata` (JSON str, optional)
    - _Requirements: 14.1_

  - [x] 2.7 Generate and apply Alembic migration
    - Run `alembic revision --autogenerate -m "initial_rsl_tables"` to generate migration for all 7 tables
    - Run `alembic upgrade head` to apply
    - _Requirements: 13.2_

- [x] 3. Pydantic Schemas
  - [x] 3.1 Create `app/schemas/upload.py` and `app/schemas/job.py`
    - `UploadResponse`: `job_id`, `status`, `filename`
    - `JobStatusResponse`: `job_id`, `status`, `created_at`, `updated_at`, `total_claims`, `processed_claims`, `failed_claims`, `error_detail`
    - `FraudFlagOut`, `ClaimResultItem`, `JobResultsResponse` (paginated)
    - _Requirements: 1.5, 2.2, 3.5_

  - [x] 3.2 Create `app/schemas/claim.py`
    - `CBCDataOut` with all 13 CBC fields
    - `ClaimDetailResponse`: full claim + `cbc_data: CBCDataOut` + `fraud_flags: List[FraudFlagOut]`
    - _Requirements: 4.3, 4.4, 4.5_

  - [x] 3.3 Create `app/schemas/facility.py` and `app/schemas/patient.py`
    - `WeeklyMetricOut`, `FacilityRiskResponse`
    - `VisitRecord`, `PatientTrajectoryResponse`
    - _Requirements: 5.3, 6.4_

  - [x] 3.4 Create `app/schemas/dashboard.py`, `app/schemas/report.py`, and `app/schemas/chat.py`
    - `TrendPoint`, `TopFacility`, `DashboardMetricsResponse`
    - `ROIReportRequest` (with validation: `start_date <= end_date`), `ROIReportResponse`
    - `ChatRequest`: `message: str`, `session_id: str`; `ChatResponse`: `response: str`, `session_id: str`
    - _Requirements: 7.3, 8.2, 8.4_

- [x] 4. Model Registry and Inference Service
  - [x] 4.1 Create `app/services/model_registry.py`
    - Implement `ModelRegistry` class with `load(models_dir)` method loading all Keras models, scalers, and encoders
    - Gracefully skip Model 4 if `cbc_model4_facility_temporal.keras` is absent; set `model4_available = False`
    - Implement `get_model_registry()` with `@lru_cache` singleton pattern (mirrors `get_ocr_service()`)
    - _Requirements: 12.7, 12.8_

  - [x] 4.2 Create `app/services/inference_service.py`
    - Define `Model1Result`, `Model2Result`, `Model3Result` dataclasses
    - Implement `run_model1(features, registry)`: scale → predict → compute MSE → `anomaly_score = min(1.0, mse / 5.004205464072272e-05)` → severity
    - Implement `run_model2(features, claimed_diagnosis, registry)`: scale → predict → decode category/diagnosis → mismatch detection → `flag_reason`
    - Implement `run_model3(sequence, registry, insufficient_history)`: scale sequence → predict → per-visit MSE → `trajectory_anomaly_score = min(1.0, mse / 0.2951826353643561)`
    - Implement `async run_inference(claim_data, patient_history, registry)`: build feature arrays, pad Model 3 sequence to 5 visits, run all three via `asyncio.gather` + `loop.run_in_executor`
    - Severity rule: `> 0.8 → "high"`, `> 0.5 → "medium"`, else `"low"`
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.9_

  - [ ]* 4.3 Write property test for anomaly score formula and bounds (Property 1)
    - **Property 1: Anomaly score = min(1.0, mse / threshold) and result ∈ [0.0, 1.0]**
    - **Validates: Requirements 4.4, 12.3**
    - Use `st.floats(min_value=0.0, allow_nan=False, allow_infinity=False)` for MSE values
    - Place in `tests/test_properties.py`

  - [ ]* 4.4 Write property test for severity classification correctness (Property 2)
    - **Property 2: Severity is always one of {"low", "medium", "high"} and follows the threshold rules**
    - **Validates: Requirements 4.4, 12.9**
    - Use `st.floats(min_value=0.0, max_value=1.0, allow_nan=False)` for anomaly_score

  - [ ]* 4.5 Write property test for Model 2 predicted category domain (Property 12)
    - **Property 12: predicted_category is always one of {"obstetric", "respiratory", "trauma"}**
    - **Validates: Requirements 12.4**
    - Use `st.floats` for CBC feature values; mock model output to return valid category indices

- [x] 5. CSV Pipeline
  - [x] 5.1 Create `app/services/csv_pipeline.py`
    - Implement `parse_csv_file(file_path)` using pandas: validate required columns, raise `ValueError` listing missing columns if any are absent
    - Required columns: `claim_id`, `patient_id`, `facility_id`, `admission_date`, `discharge_date`, `claimed_diagnosis`, `age`, `sex`, `HGB`, `HCT`, `MCV`, `MCHC`, `NEU`, `LYM`, `EOS`, `BAS`, `MON`, `PLT`, `length_of_stay`
    - Implement `run_csv_pipeline(file_path, job_id, user_id, session, registry)`: parse → per-row validation → inference → persist `Claim`, `CBCData`, `FraudFlag` records → update `PatientTrajectory` and `FacilityWeeklyMetric`
    - Per-row failures are collected; processing continues for remaining rows
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [ ]* 5.2 Write property test for CBC range validation rejection (Property 10)
    - **Property 10: Any claim with a CBC field outside its valid range is rejected and never reaches inference**
    - **Validates: Requirements 11.2**
    - Use `st.floats` outside each field's valid range; assert validation raises/returns error

  - [ ]* 5.3 Write property test for duplicate claim rejection (Property 11)
    - **Property 11: Submitting a claim_id that already exists in DB is rejected with "duplicate_claim_id"**
    - **Validates: Requirements 11.4**

- [x] 6. Claim Validation Logic
  - [x] 6.1 Create `app/services/validation_service.py`
    - Implement `validate_claim(claim_dict, existing_claim_ids)` returning `(is_valid: bool, reason: str | None)`
    - Check all required CBC fields present (Req 11.1)
    - Check numeric ranges per Req 11.2 (HGB 1–25, PLT 10–1500, HCT 5–70, MCV 50–150, MCHC 20–40, NEU 0–100, LYM 0–100, EOS 0–50, BAS 0–10, MON 0–30, age 0–120)
    - Check `admission_date <= discharge_date` → reason `"invalid_date_range"` (Req 11.3)
    - Check `claim_id` not in `existing_claim_ids` → reason `"duplicate_claim_id"` (Req 11.4)
    - Check `sex` in `{"Male", "Female", "1", "0"}` → reason `"invalid_sex_value"` (Req 11.5)
    - Encode `sex` to `sex_encoded` (Male/1 → 1, Female/0 → 0)
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

- [x] 7. LangGraph PDF Pipeline
  - [x] 7.1 Create `app/agent/state.py`
    - Define `PipelineState(TypedDict)`: `job_id`, `file_path`, `user_id`, `raw_text_blocks`, `structured_claims`, `validated_claims`, `failed_claims`, `fraud_flags`, `error`
    - _Requirements: 10.1_

  - [x] 7.2 Create `app/agent/nodes.py`
    - Implement `ocr_node(state)`: call `get_ocr_service().process_pdf(state["file_path"])`; on failure set `state["error"]`
    - Implement `llm_structuring_node(state)`: use `ChatOpenAI(model="gpt-4o")` with structured output schema to extract Claim fields from `raw_text_blocks`; per-claim failures → `failed_claims` with `extraction_failed`
    - Implement `validation_node(state)`: call `validate_claim` for each structured claim; failures → `failed_claims`
    - Implement `inference_node(state)`: call `run_inference` for each validated claim; persist `Claim`, `CBCData`, `FraudFlag` to DB; update `PatientTrajectory`
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [x] 7.3 Create `app/agent/graph.py`
    - Build `StateGraph(PipelineState)` with nodes: `ocr → llm_structuring → validation → inference → END`
    - Implement `build_pdf_pipeline()` returning compiled graph
    - Implement `run_pdf_pipeline(file_path, job_id, user_id, session, registry)` that invokes the graph and returns `{"all": [...], "failed": [...]}`
    - _Requirements: 10.1, 10.4_

- [x] 8. Job Service and Worker
  - [x] 8.1 Create `app/services/job_service.py`
    - Implement `create_job(session, filename, file_type, user_id) -> Job`: create DB record + mirror to Redis
    - Implement `get_job(session, job_id, user_id) -> Job | None`: fetch from DB (filter by `user_id`)
    - Implement `update_job_status(session, job_id, status, total?, processed?, failed?, error_detail?)`: update DB + Redis mirror
    - Redis key schema: `job:{job_id}` → JSON blob with status fields
    - Status transitions: `pending → processing → completed | partial | failed` (monotonic)
    - _Requirements: 1.4, 1.5, 2.1, 2.2, 2.4, 2.5, 2.6_

  - [x] 8.2 Create `app/workers/job_worker.py`
    - Implement `async process_job(job_id, file_path, file_type, user_id)`:
      - Set status `"processing"`
      - Route to `run_csv_pipeline` (csv/xlsx/xls) or `run_pdf_pipeline` (pdf)
      - Compute final status: `"completed"` if 0 failures, `"partial"` if some fail, `"failed"` if all fail
      - Update job status with counts; publish audit event to Kafka `rsl.audit.events`
    - _Requirements: 1.7, 1.8, 2.4, 2.5, 2.6_

  - [ ]* 8.3 Write property test for job status domain invariant (Property 4)
    - **Property 4: Job status is always one of {"pending", "processing", "completed", "failed", "partial"}**
    - **Validates: Requirements 2.4, 2.5**
    - Use `st.sampled_from` for status transitions; assert no invalid status ever produced

  - [ ]* 8.4 Write property test for partial status rule (Property 5)
    - **Property 5: When ≥1 claim succeeds and ≥1 fails, final status is "partial"**
    - **Validates: Requirements 2.5**
    - Use `st.integers(min_value=1)` for success and failure counts

  - [ ]* 8.5 Write property test for job ID uniqueness (Property 3)
    - **Property 3: Any two distinct upload requests produce different UUID job_id values**
    - **Validates: Requirements 1.6**
    - Generate N job_ids via `uuid.uuid4()`; assert all unique

- [x] 9. API Routes
  - [x] 9.1 Create `app/routes/__init__.py` and `app/routes/upload.py`
    - `POST /api/v1/upload`: validate extension (`.csv`, `.xlsx`, `.xls`, `.pdf`) → 422 if invalid; validate size ≤ 50 MB → 413 if exceeded; save file; create Job (with `user_id` from `request.state`); `asyncio.create_task(process_job(...))` for background processing; return 202 `UploadResponse`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_

  - [x] 9.2 Create `app/routes/jobs.py`
    - `GET /api/v1/jobs/{job_id}/status`: fetch job (filter by `user_id`) → 404 if not found; return `JobStatusResponse`
    - `GET /api/v1/jobs/{job_id}/results`: 404 if not found; 202 if `pending|processing`; paginated `JobResultsResponse` with `page` and `page_size` (default 50, max 500) for `completed|partial`
    - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [x] 9.3 Create `app/routes/claims.py`
    - `GET /api/v1/claims/{claim_id}`: fetch Claim + CBCData + FraudFlags (filter by `user_id`) → 404 if not found; return `ClaimDetailResponse`; log audit event `view_claim`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 9.4 Create `app/routes/facilities.py`
    - `GET /api/v1/facilities/{facility_id}/risk`: aggregate FraudFlag + FacilityWeeklyMetric (scoped to `user_id` unless admin role) → 404 if no claims; support `weeks` param (1–52, default 8); include `model4_available` field
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 9.5 Create `app/routes/patients.py`
    - `GET /api/v1/patients/{patient_id}/trajectory`: fetch PatientTrajectory (scoped to `user_id`) → 404 if not found; 422 if `total_visits < 2`; return `PatientTrajectoryResponse` with most recent 5 visits
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 9.6 Create `app/routes/dashboard.py`
    - `GET /api/v1/dashboard/metrics`: accept `period` param (`daily|weekly|monthly`, default `weekly`); aggregate from `FraudFlag` + `FacilityWeeklyMetric`; scope to `user_id` unless admin role; return empty metrics (zero counts) if no data
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 9.7 Create `app/routes/reports.py`
    - `POST /api/v1/reports/roi`: validate `start_date <= end_date` → 422; compute ROI fields; if `format=pdf` query param, render PDF with `reportlab` and return `application/pdf`; log audit event `generate_report`
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ]* 9.8 Write property test for pagination result size invariant (Property 6)
    - **Property 6: len(claims) ≤ page_size for any valid page_size in [1, 500]**
    - **Validates: Requirements 3.4**
    - Use `st.integers(min_value=1, max_value=1000)` for page_size

  - [ ]* 9.9 Write property test for flag rate arithmetic (Property 7)
    - **Property 7: flag_rate == flagged_claims / total_claims for all total_claims > 0**
    - **Validates: Requirements 5.3**
    - Use `st.integers(min_value=1)` for total_claims and `st.integers(min_value=0)` for flagged_claims ≤ total

  - [ ]* 9.10 Write property test for ROI estimated fraud amount arithmetic (Property 9)
    - **Property 9: estimated_fraud_amount_kes == flagged_claims * avg_claim_value_kes exactly**
    - **Validates: Requirements 8.4**
    - Use `st.integers(min_value=0)` and `st.floats(min_value=0.01, allow_nan=False)` for inputs

  - [ ]* 9.11 Write property test for unsupported file extension rejection (Property 14)
    - **Property 14: Any extension not in {.csv, .xlsx, .xls, .pdf, .jpg, .jpeg, .png, .bmp, .tiff} returns HTTP 422**
    - **Validates: Requirements 1.3**
    - Use `st.text()` filtered to exclude allowed extensions

- [x] 10. Checkpoint — Core pipeline complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Audit Service
  - [x] 11.1 Create `app/services/audit_service.py`
    - Implement `async log_audit_event(session, user_id, action, resource_type, resource_id, metadata=None)`:
      - Persist `AuditLog` record to DB
      - Publish event to Kafka topic `rsl.audit.events` via `kafka_service`
    - Actions to log: `upload_file`, `view_claim`, `generate_report`, `chat_query`
    - _Requirements: 14.1_

- [x] 12. Kafka Integration
  - [x] 12.1 Create `app/services/kafka_service.py`
    - Implement `KafkaProducerService` using `aiokafka.AIOKafkaProducer`:
      - `async start()` / `async stop()` lifecycle methods
      - `async publish(topic, message_dict)`: serialize to JSON and produce
    - Implement `get_kafka_producer()` singleton (started at app startup)
    - Topics: `rsl.fraud.results` (publish fraud detection results after job completion), `rsl.audit.events` (publish audit log entries)
    - Implement `KafkaConsumerService` for topic `rsl.claims.upload`:
      - Consume incoming upload events from other microservices
      - Trigger `process_job` for each consumed message
    - _Requirements: 14.6_

- [x] 13. Chat Agent
  - [x] 13.1 Create `app/agent/chat_agent.py` — LangGraph chat agent with tools
    - Define all 8 tool functions as `@tool`-decorated async functions using the async DB session:
      - `get_claim_details(claim_id: str)` — fetch Claim + CBCData + FraudFlags
      - `get_job_status(job_id: str)` — fetch Job record
      - `get_facility_risk(facility_id: str, weeks: int = 8)` — aggregate facility metrics
      - `get_patient_trajectory(patient_id: str)` — fetch PatientTrajectory
      - `get_dashboard_metrics(period: str)` — aggregate fraud stats
      - `search_claims(facility_id?, start_date?, end_date?, severity?, diagnosis?)` — filtered Claim query
      - `get_audit_log(user_id?, start_date?, end_date?)` — query AuditLog table
      - `generate_roi_report(start_date, end_date, avg_claim_value_kes)` — compute ROI
    - Build LangGraph `StateGraph` with `ToolNode` and `ChatOpenAI(model="gpt-4o")` as the LLM
    - Session memory: load/save conversation history from Redis keyed by `session_id` (list of messages, JSON-serialized)
    - Implement `async run_chat(message, session_id, user_id, session)` returning the agent's response string
    - _Requirements: 14.1_

  - [x] 13.2 Create `app/routes/chat.py`
    - `POST /api/v1/chat`: accept `ChatRequest`; extract `user_id` from `request.state`; call `run_chat`; log audit event `chat_query`; return `ChatResponse`
    - _Requirements: 14.1_

- [x] 14. Wire everything into `app/main.py`
  - [x] 14.1 Update `app/main.py` startup and router registration
    - Add `JWTMiddleware` to the app
    - Register all new routers under `/api/v1/` prefix: `upload`, `jobs`, `claims`, `facilities`, `patients`, `dashboard`, `reports`, `chat`
    - Preserve existing `/ocr/test` endpoint and CORS middleware without modification
    - In `startup` event: initialize DB engine (create tables if not exist in dev), connect Redis, call `get_model_registry()` to warm up models, start `KafkaProducerService`, start `KafkaConsumerService`
    - In `shutdown` event: stop Kafka producer and consumer
    - Update app title/description to reflect RSL
    - _Requirements: 14.1, 14.2, 14.6_

  - [ ]* 14.2 Write property test for visit sequence length cap (Property 8)
    - **Property 8: PatientTrajectory.visit_sequence always has ≤ 5 entries; oldest evicted when > 5**
    - **Validates: Requirements 6.5, 13.6**
    - Use `st.lists(st.fixed_dictionaries({...}), min_size=1, max_size=20)` for visit sequences

  - [ ]* 14.3 Write property test for insufficient history flag (Property 13)
    - **Property 13: insufficient_history=True when patient has < 5 prior visits, False otherwise**
    - **Validates: Requirements 12.6**
    - Use `st.integers(min_value=0, max_value=10)` for prior visit count

- [x] 15. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- All routes filter data by `user_id` from JWT middleware; admin role bypasses user-scoping on dashboard and facility endpoints
- The existing `app/services/ocr_service.py`, `app/utils/file_handler.py`, `app/utils/pdf_handler.py`, and `app/schemas/ocr.py` must not be modified
- Property tests live in `tests/test_properties.py` and use `hypothesis` with `@settings(max_examples=100)`
- Alembic uses psycopg2 (sync) for migrations; the app runtime uses asyncpg
- Model 4 is always absent — `model4_available` is always `False` in this deployment
