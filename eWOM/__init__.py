"""eWOM package."""

from .api import (
    EWOM_MODEL_PATHS_SCHEMA,
    EWOM_SCORE_REQUEST_SCHEMA,
    EWOM_SCORE_RESPONSE_SCHEMA,
    EWOMModelPaths,
    get_ewom_schemas,
    score_review,
)

__all__ = [
    "EWOM_MODEL_PATHS_SCHEMA",
    "EWOM_SCORE_REQUEST_SCHEMA",
    "EWOM_SCORE_RESPONSE_SCHEMA",
    "EWOMModelPaths",
    "get_ewom_schemas",
    "score_review",
]
