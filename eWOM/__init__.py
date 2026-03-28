"""eWOM package."""

from .api import (
    EWOM_MODEL_PATHS_SCHEMA,
    EWOM_REVIEW_SET_REQUEST_SCHEMA,
    EWOM_REVIEW_SET_RESPONSE_SCHEMA,
    EWOM_SCORE_REQUEST_SCHEMA,
    EWOM_SCORE_RESPONSE_SCHEMA,
    EWOMModelPaths,
    get_ewom_schemas,
    score_review,
    score_review_set,
)

__all__ = [
    "EWOM_MODEL_PATHS_SCHEMA",
    "EWOM_REVIEW_SET_REQUEST_SCHEMA",
    "EWOM_REVIEW_SET_RESPONSE_SCHEMA",
    "EWOM_SCORE_REQUEST_SCHEMA",
    "EWOM_SCORE_RESPONSE_SCHEMA",
    "EWOMModelPaths",
    "get_ewom_schemas",
    "score_review",
    "score_review_set",
]
