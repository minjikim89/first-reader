"""Wikipedia access layer: API client, AfC template parser, dataset collector."""

from first_reader.wiki.client import CategoryMember, WikiAPIError, WikiClient
from first_reader.wiki.parser import (
    AfCTemplate,
    ParseResult,
    ParseStats,
    SubmissionStatus,
    fetch_draft_histories,
    fetch_draft_history,
    iter_afc_templates,
    parse_afc_template,
    parse_declines,
    parse_timestamp,
)

__all__ = [
    "AfCTemplate",
    "CategoryMember",
    "ParseResult",
    "ParseStats",
    "SubmissionStatus",
    "WikiAPIError",
    "WikiClient",
    "fetch_draft_histories",
    "fetch_draft_history",
    "iter_afc_templates",
    "parse_afc_template",
    "parse_declines",
    "parse_timestamp",
]
