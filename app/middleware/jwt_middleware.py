"""JWT extraction middleware — decodes payload without signature verification.
Auth validation is handled by the auth microservice; this service only extracts
user_id and role from the JWT for request scoping.
"""

import logging
from jose import jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class JWTMiddleware(BaseHTTPMiddleware):
    """Extract user_id and role from Bearer JWT without verifying signature."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request.state.user_id = None
        request.state.role = "user"

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[len("Bearer "):]
            try:
                payload = jwt.decode(
                    token,
                    key="",  # no verification — auth microservice handles that
                    options={"verify_signature": False, "verify_exp": False},
                )
                request.state.user_id = payload.get("sub") or payload.get("user_id")
                request.state.role = payload.get("role", "user")
            except Exception as e:
                logger.debug(f"JWT decode failed (non-fatal): {e}")

        return await call_next(request)
