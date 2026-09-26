"""Typed retrieval failures.

Each carries an HTTP status and a message written for a user. No exception text ever contains a
credential: the provider never interpolates the client id or secret into a message, and upstream
error bodies are truncated and stripped of any Authorization echo before being quoted.
"""


class RetrievalError(Exception):
    """Base: imagery could not be obtained. Never falls back to fake imagery."""

    status = 502
    code = "retrieval_failed"

    def __init__(self, message: str, *, detail: str | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail

    def as_payload(self) -> dict:
        payload = {"code": self.code, "message": self.message}
        if self.detail:
            payload["detail"] = self.detail
        return payload


class CredentialsMissing(RetrievalError):
    status, code = 503, "credentials_missing"


class AuthenticationFailed(RetrievalError):
    status, code = 502, "authentication_failed"


class AreaInvalid(RetrievalError):
    status, code = 400, "aoi_invalid"


class AreaTooLarge(RetrievalError):
    status, code = 413, "aoi_too_large"


class NoImageryFound(RetrievalError):
    status, code = 404, "no_imagery_found"


class ProviderTimeout(RetrievalError):
    status, code = 504, "provider_timeout"


class RateLimited(RetrievalError):
    status, code = 429, "rate_limited"


class ProcessingFailed(RetrievalError):
    status, code = 502, "processing_failed"


class RasterUnreadable(RetrievalError):
    status, code = 502, "raster_unreadable"


class QueryNeedsMultipleDates(RetrievalError):
    """The question needs two or more acquisitions; single-date retrieval cannot answer it.

    Refused explicitly rather than answered from one image (CLAUDE.md section 7).
    """

    status, code = 422, "needs_multiple_dates"
