"""POST /api/v1/reports/roi"""

import io
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.sessions import get_async_db
from app.models.claim import Claim
from app.models.fraud_flag import FraudFlag
from app.schemas.report import ROIReportRequest, ROIReportResponse
from app.services.audit_service import log_audit_event

router = APIRouter()


def _build_pdf(report: ROIReportResponse) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "ResultShield Lite — ROI Report")
    c.setFont("Helvetica", 12)
    y = 760
    for label, value in [
        ("Period", f"{report.start_date} to {report.end_date}"),
        ("Total Claims", str(report.total_claims)),
        ("Flagged Claims", str(report.flagged_claims)),
        ("Flag Rate", f"{report.flag_rate:.2%}"),
        ("Avg Claim Value (KES)", f"{report.avg_claim_value_kes:,.2f}"),
        ("Estimated Fraud Amount (KES)", f"{report.estimated_fraud_amount_kes:,.2f}"),
        ("Potential Savings (KES)", f"{report.potential_savings_kes:,.2f}"),
        ("ROI %", f"{report.roi_percentage:.1f}%"),
    ]:
        c.drawString(50, y, f"{label}: {value}")
        y -= 24
    c.save()
    return buf.getvalue()


@router.post("/reports/roi")
async def roi_report(
    body: ROIReportRequest,
    request: Request,
    format: str = Query("json", pattern="^(json|pdf)$"),
    session: AsyncSession = Depends(get_async_db),
):
    user_id = getattr(request.state, "user_id", None) or "anonymous"

    claim_filter = [
        Claim.admission_date >= body.start_date,
        Claim.admission_date <= body.end_date,
        Claim.user_id == user_id,
    ]
    claims_result = await session.execute(select(Claim.id).where(*claim_filter))
    claim_ids = claims_result.scalars().all()
    total = len(claim_ids)

    flagged = 0
    if claim_ids:
        flags_result = await session.execute(
            select(FraudFlag).where(FraudFlag.claim_id.in_(claim_ids), FraudFlag.model_id == 1, FraudFlag.is_anomaly == True)
        )
        flagged = len(flags_result.scalars().all())

    recovery_rate = getattr(body, "recovery_rate", 0.7)
    fraud_amount = flagged * body.avg_claim_value_kes
    savings = fraud_amount * recovery_rate
    roi = (savings / max(fraud_amount, 1)) * 100 if fraud_amount > 0 else 0.0

    report = ROIReportResponse(
        start_date=body.start_date, end_date=body.end_date,
        total_claims=total, flagged_claims=flagged,
        flag_rate=round(flagged / max(total, 1), 4),
        avg_claim_value_kes=body.avg_claim_value_kes,
        estimated_fraud_amount_kes=fraud_amount,
        potential_savings_kes=savings,
        roi_percentage=roi,
    )

    await log_audit_event(session, user_id, "generate_report", "report",
                          f"{body.start_date}_{body.end_date}", {"format": format})

    if format == "pdf":
        pdf_bytes = _build_pdf(report)
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=roi_report_{body.start_date}_{body.end_date}.pdf"},
        )
    return report
