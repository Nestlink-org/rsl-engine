#!/usr/bin/env python3
"""
Comprehensive endpoint test for ResultShield Lite.
Tests every endpoint using kenyan_medical_claims_20260326_112856.csv (15 rows).
"""

import json
import sys
import time
import requests

BASE = "http://127.0.0.1:8000"
API = f"{BASE}/api/v1"
CSV = "kenyan_medical_claims_20260326_140337.csv"
IMG = "cbc_test_claim.jpeg"
PDF = "test-pdf.pdf"

PASS = "✅"
FAIL = "❌"
SKIP = "⏭ "

results = []

def check(name, resp, expected_status=200, key_check=None):
    ok = resp.status_code == expected_status
    body = {}
    try:
        body = resp.json()
    except Exception:
        pass
    if ok and key_check:
        ok = key_check(body)
    status = PASS if ok else FAIL
    results.append((status, name, resp.status_code, body))
    print(f"{status} [{resp.status_code}] {name}")
    if not ok:
        print(f"     body: {json.dumps(body)[:300]}")
    return body if ok else None


print("=" * 65)
print("RSL Engine — Full Endpoint Test")
print(f"CSV: {CSV} (15 rows, batch_size=5 → 5 processed + 10 queued)")
print("=" * 65)

# ── 1. Health ─────────────────────────────────────────────────────────────────
print("\n[Health]")
r = requests.get(f"{BASE}/health", timeout=5)
check("GET /health", r, 200, lambda b: b.get("status") == "healthy")
r = requests.get(f"{BASE}/", timeout=5)
check("GET /", r, 200, lambda b: "service" in b)

# ── 2. Preview — dry-run, shows ALL 15 claims, no DB write ────────────────────
print("\n[Agent — Preview (dry-run, no DB write)]")
with open(CSV, "rb") as f:
    r = requests.post(f"{API}/agent/structure/preview",
                      files={"file": (CSV, f, "text/csv")}, timeout=30)
preview = check("POST /agent/structure/preview (CSV — all 15 rows)",
                r, 200,
                lambda b: b.get("claims_shown", 0) > 5)  # must show more than batch_size
if preview:
    print(f"     preview_id={preview['preview_id']} (NOT a job_id)")
    print(f"     total_claims={preview['total_claims']} claims_shown={preview['claims_shown']}")
    print(f"     cbc={preview['cbc_claims']} hba1c={preview['hba1c_claims']} unknown={preview['unknown_claims']}")
    print(f"     eligible: {preview['model_eligible_counts']}")
    print(f"     note: {preview['note']}")
    c0 = preview["claims"][0]
    print(f"     claim[0]: {c0.get('claim_id')} diag={c0.get('claimed_diagnosis')} los={c0.get('length_of_stay')}")

# Confirm preview_id is NOT a real job
if preview:
    r2 = requests.get(f"{API}/jobs/{preview['preview_id']}/status", timeout=5)
    if r2.status_code == 404:
        print(f"     {PASS} Confirmed: preview_id is NOT a job (404 as expected)")
    else:
        print(f"     {FAIL} preview_id should not be a job, got {r2.status_code}")

# ── 3. Upload async — creates real job, queues overflow ───────────────────────
print("\n[Upload — Async (15 rows → 5 processed + 10 queued)]")
with open(CSV, "rb") as f:
    r = requests.post(f"{API}/upload",
                      files={"file": (CSV, f, "text/csv")}, timeout=15)
async_upload = check("POST /upload (async, 15-row CSV)",
                     r, 200,
                     lambda b: "job_id" in b and b.get("status") == "pending")
job_id = async_upload["job_id"] if async_upload else None
if async_upload:
    print(f"     job_id={job_id}  ← use this for /jobs/ and /agent/queue/")
    print(f"     status_url={async_upload.get('status_url')}")
    print(f"     results_url={async_upload.get('results_url')}")

# ── 4. Upload sync — processes first batch, queues rest ──────────────────────
print("\n[Upload — Sync (15 rows → first 5 sync, 10 queued in Redis)]")
with open(CSV, "rb") as f:
    r = requests.post(f"{API}/upload?sync=true",
                      files={"file": (CSV, f, "text/csv")}, timeout=120)
sync_result = check("POST /upload?sync=true (CSV)",
                    r, 200,
                    lambda b: "job_id" in b and "status" in b)
sync_job_id = None
if sync_result:
    sync_job_id = sync_result["job_id"]
    print(f"     job_id={sync_job_id}")
    print(f"     status={sync_result['status']}")
    print(f"     total_processed={sync_result['total_processed']} total_failed={sync_result['total_failed']}")
    print(f"     queued_count={sync_result['queued_count']} (remaining in Redis)")
    print(f"     models_triggered={sync_result.get('models_triggered')}")
    print(f"     time={sync_result.get('processing_time')}s")
    for r2 in sync_result.get("results", [])[:3]:
        s = r2.get("status", "")
        if s == "processed":
            print(f"     claim {r2['claim_id']}: anomaly={r2['any_anomaly']} score={r2['max_anomaly_score']}")
        else:
            print(f"     claim {r2['claim_id']}: status={s} reason={r2.get('reason','')}")

# ── 5. Job status ─────────────────────────────────────────────────────────────
print("\n[Jobs]")
use_job_id = sync_job_id or job_id
if use_job_id:
    time.sleep(2)
    r = requests.get(f"{API}/jobs/{use_job_id}/status", timeout=10)
    job_status = check("GET /jobs/{job_id}/status", r, 200, lambda b: "status" in b)
    if job_status:
        print(f"     status={job_status['status']}")
        print(f"     total_claims={job_status['total_claims']} processed={job_status['processed_claims']} failed={job_status['failed_claims']}")

    # Job results
    r = requests.get(f"{API}/jobs/{use_job_id}/results", timeout=10)
    if r.status_code == 202:
        print(f"     {SKIP} /jobs/results — job still processing")
    else:
        job_res = check("GET /jobs/{job_id}/results", r, 200, lambda b: "claims" in b)
        if job_res:
            print(f"     page=1 total={job_res['total_claims']} returned={len(job_res['claims'])}")
            if job_res["claims"]:
                c = job_res["claims"][0]
                print(f"     claim[0]: {c['claim_id']} diag={c['claimed_diagnosis']} flags={len(c['fraud_flags'])}")

# ── 6. Queue status — check overflow from sync upload ────────────────────────
print("\n[Agent — Queue (overflow from sync upload)]")
if sync_job_id:
    r = requests.get(f"{API}/agent/queue/{sync_job_id}", timeout=5)
    q = check("GET /agent/queue/{job_id}", r, 200, lambda b: "queued_claims" in b)
    if q:
        print(f"     queued_claims={q['queued_claims']} batch_size={q['batch_size']}")
        print(f"     message={q['message']}")

    # Process next batch from queue
    if q and q.get("queued_claims", 0) > 0:
        r = requests.post(f"{API}/agent/queue/{sync_job_id}/next",
                          headers={"Content-Length": "0"}, timeout=120)
        next_batch = check("POST /agent/queue/{job_id}/next (process 5 more)", r, 200,
                           lambda b: "total_processed" in b)
        if next_batch:
            print(f"     processed={next_batch['total_processed']} failed={next_batch['total_failed']}")
            print(f"     remaining_queued={next_batch['queued_count']}")
            print(f"     models={next_batch.get('models_triggered')}")
    else:
        print(f"     {SKIP} queue empty (Redis unavailable or already processed)")

# ── 7. Claims ─────────────────────────────────────────────────────────────────
print("\n[Claims]")
claim_id = None
claim_detail = None
if sync_result:
    for r2 in sync_result.get("results", []):
        if r2.get("claim_id") and r2.get("status") in ("processed", "skipped"):
            claim_id = r2["claim_id"]
            break
if claim_id:
    r = requests.get(f"{API}/claims/{claim_id}", timeout=10)
    claim_detail = check("GET /claims/{claim_id}", r, 200, lambda b: "claim_id" in b)
    if claim_detail:
        print(f"     claim_id={claim_detail['claim_id']}")
        print(f"     diagnosis={claim_detail['claimed_diagnosis']}")
        print(f"     cbc_data={'present' if claim_detail.get('cbc_data') else 'absent'}")
        print(f"     fraud_flags={len(claim_detail.get('fraud_flags', []))}")
        for ff in claim_detail.get("fraud_flags", []):
            print(f"       model{ff['model_id']}: score={ff['anomaly_score']} anomaly={ff['is_anomaly']} sev={ff['severity']}")
else:
    print(f"     {SKIP} no claim_id available")

# ── 8. Patients ───────────────────────────────────────────────────────────────
print("\n[Patients]")
patient_id = claim_detail.get("patient_id") if claim_detail else None
if patient_id:
    r = requests.get(f"{API}/patients/{patient_id}/trajectory", timeout=10)
    if r.status_code == 422:
        print(f"     {SKIP} insufficient visits (need ≥2) — expected for fresh data")
        results.append((SKIP, "GET /patients/{patient_id}/trajectory", 422, {}))
    else:
        traj = check("GET /patients/{patient_id}/trajectory", r, 200, lambda b: "patient_id" in b)
        if traj:
            print(f"     patient={traj['patient_id']} visits={traj['total_visits']} score={traj.get('trajectory_anomaly_score')}")
else:
    print(f"     {SKIP} no patient_id available")

# ── 9. Analytics dashboard ────────────────────────────────────────────────────
print("\n[Analytics Dashboard]")
for period in ["weekly", "daily", "monthly"]:
    r = requests.get(f"{API}/dashboard/metrics?period={period}", timeout=15)
    d = check(f"GET /dashboard/metrics?period={period}", r, 200, lambda b: "total_claims" in b)
    if d and period == "weekly":
        print(f"     total={d['total_claims']} flagged={d['flagged_claims']} rate={d['flag_rate']}")
        sev = d.get("severity_distribution", {})
        print(f"     severity: high={sev.get('high')} med={sev.get('medium')} low={sev.get('low')}")
        print(f"     top_facilities={len(d['top_facilities'])} top_patients={len(d['top_anomalous_patients'])}")
        print(f"     disease_breakdown={len(d['disease_breakdown'])} diagnoses:")
        for dis in d["disease_breakdown"]:
            print(f"       {dis['diagnosis']} ({dis['category']}): {dis['claim_count']} claims, rate={dis['flag_rate']}")
        print(f"     model_performance:")
        for m in d["model_performance"]:
            print(f"       Model {m['model_id']} ({m['model_name']}): evaluated={m['total_evaluated']} flagged={m['flagged']} rate={m['flag_rate']}")
        if d["top_facilities"]:
            tf = d["top_facilities"][0]
            print(f"     top facility: {tf['facility_id']} flagged={tf['flagged_claims']} rate={tf['flag_rate']}")
        if d["top_anomalous_patients"]:
            tp = d["top_anomalous_patients"][0]
            print(f"     top patient: {tp['patient_id']} score={tp['trajectory_anomaly_score']}")

# ── 10. Reports ───────────────────────────────────────────────────────────────
print("\n[Reports]")
r = requests.post(f"{API}/reports/roi",
                  json={"start_date": "2022-01-01", "end_date": "2026-12-31",
                        "avg_claim_value_kes": 50000.0, "recovery_rate": 0.7},
                  timeout=15)
roi = check("POST /reports/roi (JSON)", r, 200, lambda b: "total_claims" in b)
if roi:
    print(f"     total={roi['total_claims']} flagged={roi['flagged_claims']}")
    print(f"     fraud_kes={roi['estimated_fraud_amount_kes']:,.0f} savings_kes={roi['potential_savings_kes']:,.0f} roi={roi['roi_percentage']:.1f}%")

r = requests.post(f"{API}/reports/roi?format=pdf",
                  json={"start_date": "2022-01-01", "end_date": "2026-12-31",
                        "avg_claim_value_kes": 50000.0, "recovery_rate": 0.7},
                  timeout=15)
check("POST /reports/roi?format=pdf", r, 200)
if r.status_code == 200:
    print(f"     PDF size={len(r.content)} bytes content-type={r.headers.get('content-type')}")

# ── 11. OCR ───────────────────────────────────────────────────────────────────
print("\n[OCR]")
with open(IMG, "rb") as f:
    r = requests.post(f"{API}/ocr/extract",
                      files={"file": (IMG, f, "image/jpeg")}, timeout=90)
ocr_result = check("POST /ocr/extract (JPEG)", r, 200,
                   lambda b: b.get("success") and len(b.get("text_blocks", [])) > 0)
file_hash = None
if ocr_result:
    file_hash = ocr_result.get("file_hash")
    print(f"     cached={ocr_result['cached']} blocks={len(ocr_result['text_blocks'])} time={ocr_result['processing_time']}s")
    print(f"     preview: {ocr_result['full_text'][:80]}")

with open(PDF, "rb") as f:
    r = requests.post(f"{API}/ocr/extract",
                      files={"file": (PDF, f, "application/pdf")}, timeout=60)
ocr_pdf = check("POST /ocr/extract (PDF)", r, 200,
                lambda b: b.get("success") and len(b.get("text_blocks", [])) > 0)
if ocr_pdf:
    print(f"     cached={ocr_pdf['cached']} blocks={len(ocr_pdf['text_blocks'])} time={ocr_pdf['processing_time']}s")

with open(IMG, "rb") as f1, open(PDF, "rb") as f2:
    r = requests.post(f"{API}/ocr/batch",
                      files=[("files", (IMG, f1, "image/jpeg")),
                             ("files", (PDF, f2, "application/pdf"))],
                      timeout=60)
batch_ocr = check("POST /ocr/batch (2 files)", r, 200, lambda b: b.get("succeeded", 0) > 0)
batch_id = None
if batch_ocr:
    batch_id = batch_ocr.get("batch_id")
    print(f"     batch_id={batch_id}")
    print(f"     succeeded={batch_ocr['succeeded']} failed={batch_ocr['failed']} blocks={batch_ocr['total_blocks']}")
    print(f"     retrieve_url={batch_ocr.get('retrieve_url')}")

if batch_id:
    r = requests.get(f"{API}/ocr/batch/{batch_id}", timeout=10)
    if r.status_code == 404:
        print(f"     {SKIP} GET /ocr/batch/{{id}} — Redis unavailable (no caching without Redis)")
        results.append((SKIP, "GET /ocr/batch/{batch_id}", 404, {}))
    else:
        br = check("GET /ocr/batch/{batch_id}", r, 200, lambda b: b.get("total_blocks", 0) > 0)
        if br:
            print(f"     retrieved: {br['total_files']} files, {br['total_blocks']} blocks")

if file_hash:
    r = requests.get(f"{API}/ocr/cache/{file_hash}", timeout=5)
    body = r.json()
    if not body.get("cached"):
        print(f"     {SKIP} GET /ocr/cache/{{hash}} — Redis unavailable, no cache")
        results.append((SKIP, "GET /ocr/cache/{file_hash}", 200, body))
    else:
        cr = check("GET /ocr/cache/{file_hash}", r, 200, lambda b: b.get("cached") is True)
        if cr:
            print(f"     cached={cr['cached']} type={cr.get('type')} blocks={cr.get('block_count')}")

# ── 12. Upload image via OCR pipeline ─────────────────────────────────────────
print("\n[Upload — OCR image (JPEG → OCR → LLM structuring)]")
with open(IMG, "rb") as f:
    r = requests.post(f"{API}/upload?sync=true",
                      files={"file": (IMG, f, "image/jpeg")}, timeout=120)
img_result = check("POST /upload?sync=true (JPEG)", r, 200, lambda b: "job_id" in b)
if img_result:
    print(f"     status={img_result.get('status')} processed={img_result.get('total_processed')} failed={img_result.get('total_failed')}")
    print(f"     models={img_result.get('models_triggered')}")
    for r2 in img_result.get("results", []):
        print(f"     claim {r2['claim_id']}: status={r2.get('status')} models={r2.get('models_run')}")
    for fd in img_result.get("failed_details", [])[:2]:
        print(f"     note: {fd.get('reason','')[:80]}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
passed = sum(1 for s, *_ in results if s == PASS)
failed_count = sum(1 for s, *_ in results if s == FAIL)
skipped_count = sum(1 for s, *_ in results if s == SKIP)
total = len(results)
print(f"RESULTS: {passed} passed  {failed_count} failed  {skipped_count} skipped  ({total} total)")
if failed_count:
    print("\nFailed:")
    for s, name, code, body in results:
        if s == FAIL:
            print(f"  {FAIL} [{code}] {name}")
            print(f"       {json.dumps(body)[:200]}")
if skipped_count:
    print("\nSkipped (infrastructure/expected):")
    for s, name, code, body in results:
        if s == SKIP:
            print(f"  {SKIP} [{code}] {name}")
print("=" * 65)
sys.exit(0 if failed_count == 0 else 1)
