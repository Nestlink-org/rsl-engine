"""POST /api/v1/chat — conversational fraud detection assistant."""

import re
from fastapi import APIRouter, Depends, Request
from sqlmodel.ext.asyncio.session import AsyncSession

from app.agent.chat_agent import run_chat
from app.db.sessions import get_async_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.audit_service import log_audit_event

router = APIRouter()

# Extract /static/visualizations/*.png URLs from agent response
_VIZ_URL_RE = re.compile(r"/static/visualizations/[\w\-\.]+\.png")


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Conversational fraud detection assistant",
    description="""
Chat with the RSL fraud detection AI assistant.

The agent has access to 14 tools:
- **Data tools**: claim details, job status, facility risk, patient trajectory, dashboard summary, claim search, audit log, ROI report
- **Visualization tools**: facility anomaly bar chart, patient trajectory heatmap, disease distribution, anomaly trend, CBC radar chart, top anomalous patients

When the agent generates a chart, the response includes the image URL under `visualization_urls`.
Session memory is persisted in Redis (TTL 24h, last 30 turns).

Example questions:
- "Show me the top facilities with high anomaly scores"
- "Plot the anomaly trend for the last 12 weeks"
- "What is the CBC profile for claim CLM-0100?"
- "Which patients have the highest trajectory anomaly scores?"
- "Give me a disease breakdown chart"
    """,
    tags=["Chat"],
)
async def chat(
    body: ChatRequest,
    request: Request,
    session: AsyncSession = Depends(get_async_db),
):
    user_id = getattr(request.state, "user_id", None) or "anonymous"
    response_text = await run_chat(body.message, body.session_id, user_id, session)

    # Extract any visualization URLs embedded in the response
    viz_urls = _VIZ_URL_RE.findall(response_text)

    await log_audit_event(
        session, user_id, "chat_query", "chat", body.session_id,
        {"message_preview": body.message[:100], "viz_count": len(viz_urls)},
    )

    return ChatResponse(
        response=response_text,
        session_id=body.session_id,
        visualization_urls=viz_urls,
    )
