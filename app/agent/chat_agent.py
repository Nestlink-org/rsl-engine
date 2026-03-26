"""RSL Chat Agent — LangGraph multi-tool agent with Redis session memory.

Tools:
  Data tools (8):  get_claim_details, get_job_status, get_facility_risk,
                   get_patient_trajectory, get_dashboard_metrics, search_claims,
                   get_audit_log, generate_roi_report
  Viz tools (6):   plot_facility_anomaly_scores, plot_patient_trajectory,
                   plot_disease_distribution, plot_anomaly_trend,
                   plot_claim_cbc_profile, plot_top_anomalous_patients

Memory: Redis-backed per session_id, TTL 24h, last 30 turns retained.
"""

import json
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent.config import get_llm
from app.agent.tools.viz_tools import VIZ_TOOLS
from app.db.sessions import async_session

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the ResultShield Lite fraud detection assistant for insurance companies.

You have access to analytical tools that query the fraud detection database and generate visualizations.

Guidelines:
- Always use tools to answer questions — never guess from memory.
- When users ask about charts, plots, trends, or visual analysis, use the visualization tools.
- When users ask about a specific claim, patient, or facility, fetch the data first.
- Visualization tools return a URL — always include it in your response so the user can view the chart.
- Be concise and data-driven. Lead with the key finding, then provide supporting details.
- For anomaly scores: 0.0–0.5 = normal, 0.5–0.8 = medium risk, 0.8–1.0 = high risk.
- Disease categories: obstetric, respiratory, trauma.
- Diagnoses: APH PPH, ASTHMA, PNEUMONIA, PUERPERAL SEPSIS, TBI.

You serve insurance companies reviewing medical claims for fraud. Be professional and precise."""


# ─── Data tools ──────────────────────────────────────────────────────────────

@tool
async def get_claim_details(claim_id: str) -> str:
    """Fetch full claim details including CBC lab values and all fraud detection flags."""
    from sqlalchemy import select
    from app.models.claim import CBCData, Claim
    from app.models.fraud_flag import FraudFlag
    async with async_session() as session:
        result = await session.execute(select(Claim).where(Claim.claim_id == claim_id))
        claim = result.scalar_one_or_none()
        if not claim:
            return f"Claim {claim_id} not found."
        cbc_result = await session.execute(select(CBCData).where(CBCData.claim_id == claim.id))
        cbc = cbc_result.scalar_one_or_none()
        flags_result = await session.execute(select(FraudFlag).where(FraudFlag.claim_id == claim.id))
        flags = [
            {
                "model_id": f.model_id,
                "score": round(f.anomaly_score, 4),
                "is_anomaly": f.is_anomaly,
                "severity": f.severity,
                "reason": f.flag_reason,
                "predicted_diagnosis": f.predicted_diagnosis,
                "diagnosis_confidence": f.diagnosis_confidence,
            }
            for f in flags_result.scalars().all()
        ]
        cbc_data = {}
        if cbc:
            cbc_data = {
                "age": cbc.age, "sex": "Male" if cbc.sex_encoded == 1 else "Female",
                "HGB": cbc.HGB, "HCT": cbc.HCT, "MCV": cbc.MCV, "MCHC": cbc.MCHC,
                "NEU": cbc.NEU, "LYM": cbc.LYM, "EOS": cbc.EOS, "BAS": cbc.BAS,
                "MON": cbc.MON, "PLT": cbc.PLT, "length_of_stay": cbc.length_of_stay,
            }
        return json.dumps({
            "claim_id": claim_id,
            "patient_id": claim.patient_id,
            "facility_id": claim.facility_id,
            "diagnosis": claim.claimed_diagnosis,
            "admission_date": str(claim.admission_date),
            "discharge_date": str(claim.discharge_date),
            "cbc_data": cbc_data,
            "fraud_flags": flags,
        }, indent=2)


@tool
async def get_job_status(job_id: str) -> str:
    """Fetch processing job status and claim counts."""
    from sqlalchemy import select
    from app.models.job import Job
    async with async_session() as session:
        result = await session.execute(select(Job).where(Job.job_id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            return f"Job {job_id} not found."
        return json.dumps({
            "job_id": job_id, "status": job.status,
            "total": job.total_claims, "processed": job.processed_claims,
            "failed": job.failed_claims, "filename": job.filename,
            "created_at": str(job.created_at),
        })


@tool
async def get_facility_risk(facility_id: str, weeks: int = 8) -> str:
    """Get facility risk metrics: claim volume, flag rate, avg anomaly score over N weeks."""
    from datetime import timedelta
    from sqlalchemy import func, select
    from app.models.facility_metric import FacilityWeeklyMetric
    since = date.today() - timedelta(weeks=weeks)
    async with async_session() as session:
        result = await session.execute(
            select(FacilityWeeklyMetric).where(
                FacilityWeeklyMetric.facility_id == facility_id,
                FacilityWeeklyMetric.week_start_date >= since,
            ).order_by(FacilityWeeklyMetric.week_start_date)
        )
        metrics = result.scalars().all()
        if not metrics:
            return f"No metrics found for facility {facility_id} in the last {weeks} weeks."
        total = sum(m.claim_volume for m in metrics)
        flagged = sum(m.flagged_claims for m in metrics)
        avg_score = sum(m.avg_anomaly_score * m.claim_volume for m in metrics) / max(total, 1)
        high = sum(m.high_severity_count for m in metrics)
        weekly = [
            {"week": str(m.week_start_date), "volume": m.claim_volume,
             "flagged": m.flagged_claims, "avg_score": round(m.avg_anomaly_score, 4)}
            for m in metrics
        ]
        return json.dumps({
            "facility_id": facility_id, "weeks_analyzed": weeks,
            "total_claims": total, "flagged_claims": flagged,
            "flag_rate": round(flagged / max(total, 1), 4),
            "avg_anomaly_score": round(avg_score, 4),
            "high_severity_count": high,
            "weekly_breakdown": weekly,
        }, indent=2)


@tool
async def get_patient_trajectory(patient_id: str) -> str:
    """Get patient visit trajectory, anomaly scores, and per-visit reconstruction errors."""
    from sqlalchemy import select
    from app.models.patient_trajectory import PatientTrajectory
    async with async_session() as session:
        result = await session.execute(
            select(PatientTrajectory).where(PatientTrajectory.patient_id == patient_id)
        )
        traj = result.scalar_one_or_none()
        if not traj:
            return f"No trajectory found for patient {patient_id}."
        visits = json.loads(traj.visit_sequence or "[]")
        errors = json.loads(traj.per_visit_errors or "[]")
        return json.dumps({
            "patient_id": patient_id,
            "total_visits": len(visits),
            "trajectory_score": round(traj.trajectory_anomaly_score or 0, 4),
            "is_anomaly": traj.is_trajectory_anomaly,
            "most_anomalous_visit": traj.most_anomalous_visit_index,
            "per_visit_errors": [round(e, 4) for e in errors],
            "last_updated": str(traj.last_updated),
        }, indent=2)


@tool
async def get_dashboard_summary(period: str = "weekly") -> str:
    """Get overall fraud detection summary: totals, flag rates, top facilities, disease breakdown."""
    from sqlalchemy import func, select
    from app.models.facility_metric import FacilityWeeklyMetric
    from app.models.fraud_flag import FraudFlag
    from datetime import timedelta
    from collections import Counter
    cutoff = date.today() - timedelta(weeks=12)
    async with async_session() as session:
        metrics_result = await session.execute(
            select(FacilityWeeklyMetric).where(FacilityWeeklyMetric.week_start_date >= cutoff)
        )
        metrics = metrics_result.scalars().all()
        total = sum(m.claim_volume for m in metrics)
        flagged = sum(m.flagged_claims for m in metrics)
        avg = sum(m.avg_anomaly_score * m.claim_volume for m in metrics) / max(total, 1)

        # Top facilities
        fac_stats: dict = {}
        for m in metrics:
            if m.facility_id not in fac_stats:
                fac_stats[m.facility_id] = {"flagged": 0, "vol": 0}
            fac_stats[m.facility_id]["flagged"] += m.flagged_claims
            fac_stats[m.facility_id]["vol"] += m.claim_volume
        top_fac = sorted(fac_stats.items(), key=lambda x: x[1]["flagged"], reverse=True)[:5]

        # Disease breakdown
        diag_result = await session.execute(
            select(FraudFlag.predicted_diagnosis, FraudFlag.is_anomaly)
            .where(FraudFlag.model_id == 2, FraudFlag.predicted_diagnosis.isnot(None))
        )
        diag_rows = diag_result.all()
        diag_total = Counter(r[0] for r in diag_rows)
        diag_flagged = Counter(r[0] for r in diag_rows if r[1])

    return json.dumps({
        "period": period,
        "total_claims": total,
        "flagged_claims": flagged,
        "flag_rate": round(flagged / max(total, 1), 4),
        "avg_anomaly_score": round(avg, 4),
        "top_facilities": [
            {"facility_id": fid, "flagged": s["flagged"],
             "flag_rate": round(s["flagged"] / max(s["vol"], 1), 4)}
            for fid, s in top_fac
        ],
        "disease_breakdown": [
            {"diagnosis": d, "total": diag_total[d], "flagged": diag_flagged.get(d, 0),
             "flag_rate": round(diag_flagged.get(d, 0) / max(diag_total[d], 1), 4)}
            for d in diag_total
        ],
    }, indent=2)


@tool
async def search_claims(
    facility_id: Optional[str] = None,
    patient_id: Optional[str] = None,
    diagnosis: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    anomalous_only: bool = False,
) -> str:
    """Search claims with filters. Returns up to 20 results with fraud flag summaries."""
    from sqlalchemy import select
    from app.models.claim import Claim
    from app.models.fraud_flag import FraudFlag
    async with async_session() as session:
        filters = []
        if facility_id:
            filters.append(Claim.facility_id == facility_id)
        if patient_id:
            filters.append(Claim.patient_id == patient_id)
        if diagnosis:
            filters.append(Claim.claimed_diagnosis.ilike(f"%{diagnosis}%"))
        if start_date:
            filters.append(Claim.admission_date >= datetime.strptime(start_date, "%Y-%m-%d").date())
        if end_date:
            filters.append(Claim.admission_date <= datetime.strptime(end_date, "%Y-%m-%d").date())

        result = await session.execute(select(Claim).where(*filters).limit(20))
        claims = result.scalars().all()

        output = []
        for c in claims:
            flags_result = await session.execute(
                select(FraudFlag).where(FraudFlag.claim_id == c.id, FraudFlag.model_id == 1)
            )
            flag = flags_result.scalar_one_or_none()
            if anomalous_only and (not flag or not flag.is_anomaly):
                continue
            output.append({
                "claim_id": c.claim_id, "patient_id": c.patient_id,
                "facility_id": c.facility_id, "diagnosis": c.claimed_diagnosis,
                "admission_date": str(c.admission_date),
                "anomaly_score": round(flag.anomaly_score, 4) if flag else None,
                "is_anomaly": flag.is_anomaly if flag else None,
                "severity": flag.severity if flag else None,
            })
        return json.dumps(output, indent=2)


@tool
async def get_audit_log(
    user_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """Query the audit log for user actions (uploads, views, reports)."""
    from sqlalchemy import select
    from app.models.audit_log import AuditLog
    async with async_session() as session:
        filters = []
        if user_id:
            filters.append(AuditLog.user_id == user_id)
        if start_date:
            filters.append(AuditLog.timestamp >= datetime.strptime(start_date, "%Y-%m-%d"))
        if end_date:
            filters.append(AuditLog.timestamp <= datetime.strptime(end_date, "%Y-%m-%d"))
        result = await session.execute(select(AuditLog).where(*filters).order_by(AuditLog.timestamp.desc()).limit(50))
        logs = result.scalars().all()
        return json.dumps([
            {"action": l.action, "resource_type": l.resource_type,
             "resource_id": l.resource_id, "timestamp": str(l.timestamp)}
            for l in logs
        ], indent=2)


@tool
async def generate_roi_report(start_date: str, end_date: str, avg_claim_value_kes: float = 50000.0) -> str:
    """Compute ROI report: estimated fraud amount and potential savings for a date range."""
    from sqlalchemy import select
    from app.models.claim import Claim
    from app.models.fraud_flag import FraudFlag
    sd = datetime.strptime(start_date, "%Y-%m-%d").date()
    ed = datetime.strptime(end_date, "%Y-%m-%d").date()
    async with async_session() as session:
        claims_result = await session.execute(
            select(Claim.id).where(Claim.admission_date >= sd, Claim.admission_date <= ed)
        )
        claim_ids = claims_result.scalars().all()
        total = len(claim_ids)
        flagged = 0
        if claim_ids:
            flags_result = await session.execute(
                select(FraudFlag).where(
                    FraudFlag.claim_id.in_(claim_ids),
                    FraudFlag.model_id == 1,
                    FraudFlag.is_anomaly == True,
                )
            )
            flagged = len(flags_result.scalars().all())
        fraud_amount = flagged * avg_claim_value_kes
        savings = fraud_amount * 0.7
        return json.dumps({
            "period": f"{start_date} to {end_date}",
            "total_claims": total,
            "flagged_claims": flagged,
            "flag_rate": round(flagged / max(total, 1), 4),
            "avg_claim_value_kes": avg_claim_value_kes,
            "estimated_fraud_kes": fraud_amount,
            "potential_savings_kes": savings,
            "roi_percentage": round((savings / max(fraud_amount, 1)) * 100, 1),
        }, indent=2)


# ─── Tool registry ────────────────────────────────────────────────────────────

DATA_TOOLS = [
    get_claim_details, get_job_status, get_facility_risk, get_patient_trajectory,
    get_dashboard_summary, search_claims, get_audit_log, generate_roi_report,
]

ALL_TOOLS = DATA_TOOLS + VIZ_TOOLS


# ─── Agent graph ──────────────────────────────────────────────────────────────

def _get_llm():
    return get_llm(temperature=0).bind_tools(ALL_TOOLS)


def _should_continue(state):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def _build_chat_graph():
    from langgraph.graph import MessagesState
    llm = _get_llm()
    tool_node = ToolNode(ALL_TOOLS)

    def call_model(state):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


_chat_graph = None


def get_chat_graph():
    global _chat_graph
    if _chat_graph is None:
        _chat_graph = _build_chat_graph()
    return _chat_graph


# ─── Redis session memory ─────────────────────────────────────────────────────

def _get_redis_sync():
    """Sync Redis client for session memory."""
    try:
        import redis
        from app.core.config import settings
        client = redis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        client.ping()
        return client
    except Exception:
        return None


def _load_history(session_id: str) -> List[Dict]:
    r = _get_redis_sync()
    if r is None:
        return []
    try:
        raw = r.get(f"chat:{session_id}")
        return json.loads(raw) if raw else []
    except Exception:
        return []


def _save_history(session_id: str, messages: List[Dict]) -> None:
    r = _get_redis_sync()
    if r is None:
        return
    try:
        # Keep last 30 turns, TTL 24h
        r.set(f"chat:{session_id}", json.dumps(messages[-30:]), ex=86400)
    except Exception as e:
        logger.warning(f"[chat] failed to save history session={session_id}: {e}")


def _deserialize_messages(history: List[Dict]) -> List:
    result = []
    for m in history:
        if m["role"] == "human":
            result.append(HumanMessage(content=m["content"]))
        elif m["role"] == "ai":
            result.append(AIMessage(content=m["content"]))
    return result


# ─── Public interface ─────────────────────────────────────────────────────────

async def run_chat(message: str, session_id: str, user_id: str, session: Any) -> str:
    """Run the chat agent with persistent session memory. Returns response text."""
    history = _load_history(session_id)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + _deserialize_messages(history)
    messages.append(HumanMessage(content=message))

    graph = get_chat_graph()
    result = await graph.ainvoke({"messages": messages})
    response_msg = result["messages"][-1]
    response_text = response_msg.content if hasattr(response_msg, "content") else str(response_msg)

    history.append({"role": "human", "content": message})
    history.append({"role": "ai", "content": response_text})
    _save_history(session_id, history)

    logger.info(f"[chat] session={session_id} user={user_id} turns={len(history)//2}")
    return response_text
