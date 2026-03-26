# OCR Module — ResultShield Lite

PaddleOCR 2.9.1-based text extraction for medical claim images and PDFs.  
Results are cached in Redis by SHA-256 file hash (TTL 24h) and support batch processing.

---

## Architecture

```
POST /api/v1/ocr/extract
POST /api/v1/ocr/batch
GET  /api/v1/ocr/batch/{batch_id}
GET  /api/v1/ocr/cache/{file_hash}
         │
         ▼
  app/routes/ocr.py          ← FastAPI route handlers
         │
         ▼
  app/services/ocr_service.py ← OCRService singleton (PaddleOCR wrapper)
         │
         ├── Redis cache (ocr:result:{hash}:p1 / :pdf)
         └── app/utils/pdf_handler.py  ← PDF → image conversion (PyMuPDF)
```

**Key design decisions:**
- OCR engine initialises **before** TensorFlow at startup — PaddlePaddle and TF both hook into CUDA libs; loading TF first causes a segfault
- All blocking OCR calls run inside `run_in_threadpool()` — keeps the async event loop free
- Singleton protected by `threading.Lock` — safe when called from multiple thread pool workers
- Redis calls also run in `run_in_threadpool()` — sync Redis client, never blocks the event loop

---

## Configuration

Set in `.env` (loaded by `app/core/config.py`):

| Variable | Default | Description |
|---|---|---|
| `OCR_LANG` | `en` | PaddleOCR language model |
| `USE_GPU` | `False` | Enable GPU inference |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection for caching |
| `UPLOAD_DIR` | `uploads` | Temp directory for file processing |
| `MAX_FILE_SIZE` | `52428800` (50MB) | Max upload size in bytes |

Example `.env`:
```env
OCR_LANG=en
USE_GPU=False
REDIS_URL=redis://localhost:6379/0
UPLOAD_DIR=uploads
MAX_FILE_SIZE=52428800
```

---

## Supported File Types

| Type | Extensions |
|---|---|
| Images | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff` |
| Documents | `.pdf` (converted to images per page via PyMuPDF) |

---

## Startup

OCR engine cold-starts in ~20–40s. Start the server with:

```bash
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True .venv/bin/fastapi run app/main.py --port 8000
```

> Do **not** use `fastapi dev` — the file watcher restarts the server when temp files are written to `uploads/`, killing in-flight OCR requests.

Startup log confirms readiness:
```
app.main INFO Initialising OCR engine (blocking, ~20-40s cold start)...
app.main INFO OCR engine ready
app.main INFO RSL Engine started
```

---

## Endpoints

### POST /api/v1/ocr/extract

Extract text from a single image or PDF.

**Request:** `multipart/form-data`
- `file` — image or PDF file

**Response:**
```json
{
  "success": true,
  "filename": "claim.jpeg",
  "file_type": "image",
  "page_count": 1,
  "text_blocks": [
    {
      "text": "Haemoglobin",
      "confidence": 0.9981,
      "bounding_box": { "points": [[81,530],[188,530],[188,553],[81,553]] },
      "page_number": 1
    }
  ],
  "full_text": "Haemoglobin\n9.10 [L]\ngm/dl\n...",
  "processing_time": 0.009,
  "timestamp": "2026-03-24T21:27:16.539000",
  "cached": true,
  "file_hash": "6fe3e2426fd861985b3a..."
}
```

- `cached: true` means result was served from Redis — no OCR was run
- `full_text` is all blocks joined with `\n`, ready to pass to the LLM agent

---

### POST /api/v1/ocr/batch

Process multiple files in one request. Results stored in Redis under `ocr:batch:{batch_id}`.

**Request:** `multipart/form-data`
- `files` — one or more files (repeat the field for each file)

**Response:**
```json
{
  "batch_id": "4b591821-5cc6-4879-8d69-c0a8348db0af",
  "total_files": 2,
  "succeeded": 2,
  "failed": 0,
  "total_blocks": 153,
  "total_elapsed": 0.023,
  "retrieve_url": "/api/v1/ocr/batch/4b591821-5cc6-4879-8d69-c0a8348db0af",
  "results_summary": [
    { "file": "claim1.jpeg", "status": "success", "block_count": 145, "processing_time": 0.012 },
    { "file": "report.pdf",  "status": "success", "block_count": 8,   "processing_time": 0.003 }
  ]
}
```

---

### GET /api/v1/ocr/batch/{batch_id}

Retrieve full batch results (including all `text_blocks` and `full_text` per file) from Redis.  
Results expire after 24 hours.

**Response:** same structure as above but with full `blocks` and `full_text` arrays per file.

---

### GET /api/v1/ocr/cache/{file_hash}

Check if a file's OCR result is already cached.

**Response:**
```json
{ "cached": true, "type": "image", "block_count": 145 }
```
or
```json
{ "cached": false, "file_hash": "6fe3e2426fd861985b3a..." }
```

---

## Testing

### Single image
```bash
curl -X POST 'http://127.0.0.1:8000/api/v1/ocr/extract' \
  -F 'file=@cbc_test_claim.jpeg;type=image/jpeg'
```

### PDF
```bash
curl -X POST 'http://127.0.0.1:8000/api/v1/ocr/extract' \
  -F 'file=@test-pdf.pdf;type=application/pdf'
```

### PNG
```bash
curl -X POST 'http://127.0.0.1:8000/api/v1/ocr/extract' \
  -F 'file=@Masqany.png;type=image/png'
```

### Cache hit (run same file twice — second call is instant)
```bash
# First call: runs OCR (~20s cold, ~2s warm)
curl -X POST 'http://127.0.0.1:8000/api/v1/ocr/extract' \
  -F 'file=@cbc_test_claim.jpeg;type=image/jpeg'

# Second call: Redis cache hit (<0.1s)
curl -X POST 'http://127.0.0.1:8000/api/v1/ocr/extract' \
  -F 'file=@cbc_test_claim.jpeg;type=image/jpeg'
```

### Batch (3 files)
```bash
curl -X POST 'http://127.0.0.1:8000/api/v1/ocr/batch' \
  -F 'files=@cbc_test_claim.jpeg;type=image/jpeg' \
  -F 'files=@cbc_test_claim_2.jpeg;type=image/jpeg' \
  -F 'files=@test-pdf.pdf;type=application/pdf'
```

### Retrieve batch result
```bash
curl 'http://127.0.0.1:8000/api/v1/ocr/batch/{batch_id}'
```

### Check cache by hash
```bash
curl 'http://127.0.0.1:8000/api/v1/ocr/cache/{file_hash}'
```

### Via Swagger UI
Open `http://127.0.0.1:8000/docs` → OCR section.

---

## Redis Cache Keys

| Pattern | Content | TTL |
|---|---|---|
| `ocr:result:{hash}:p1` | Single image result | 24h |
| `ocr:result:{hash}:pdf` | Full PDF result | 24h |
| `ocr:batch:{batch_id}` | Batch job result | 24h |

Verify Redis is running:
```bash
redis-cli ping   # → PONG
redis-cli keys "ocr:*"  # list all cached OCR keys
```

---

## Observed Performance (CPU, no GPU)

| Operation | Time |
|---|---|
| Cold start (first OCR init) | ~20–40s |
| Warm image inference | ~2–5s |
| Redis cache hit | <0.1s |
| 3-file batch (all cached) | ~0.02s |

---

## Source Files

| File | Role |
|---|---|
| `app/routes/ocr.py` | Route handlers, file validation, threadpool dispatch |
| `app/services/ocr_service.py` | OCRService class, Redis cache helpers, singleton |
| `app/schemas/ocr.py` | Pydantic models: `TextBlock`, `BoundingBox`, `OCRResponse` |
| `app/utils/pdf_handler.py` | PDF → numpy image array conversion (PyMuPDF) |
| `app/core/config.py` | `OCR_LANG`, `USE_GPU`, `REDIS_URL`, `MAX_FILE_SIZE` |
| `app/main.py` | Lifespan: OCR init before TF, router registration |
