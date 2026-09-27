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


# --------------------------------------------------------------------------- temporal (two-date) retrieval


class TemporalRangeUnsupported(RetrievalError):
    """The periods the question names cannot be searched: in the future, before Sentinel-2, or too short."""

    status, code = 422, "temporal_range_unsupported"


class NoEarlierImagery(RetrievalError):
    status, code = 404, "no_earlier_imagery"


class NoLaterImagery(RetrievalError):
    status, code = 404, "no_later_imagery"


class OnlyOneAcquisition(RetrievalError):
    """Only one acquisition date qualifies. It is never duplicated to fake a pair."""

    status, code = 404, "only_one_acquisition"


class NoSarImagery(RetrievalError):
    """No SAR scene close enough in time to the optical one. An optical scene is never analysed alone
    in answer to a question that asks for optical and SAR together."""

    status, code = 404, "no_sar_imagery"


class SarTemporalUnsupported(RetrievalError):
    """A radar question about change over time: two-date Sentinel-1 retrieval does not exist yet, and
    answering it with optical imagery instead would ignore what was asked."""

    status, code = 422, "sar_temporal_unsupported"


class GridsIncompatible(RetrievalError):
    """The two rasters do not share a pixel grid, so no pixel-wise comparison is run."""

    status, code = 502, "grids_incompatible"
