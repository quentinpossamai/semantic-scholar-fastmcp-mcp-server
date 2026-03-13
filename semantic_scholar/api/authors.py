"""
Author-related API endpoints for the Semantic Scholar API.
"""

from typing import Dict, List, Optional

from fastmcp import Context

from ..config import ErrorType
from ..core.client import S2Client, make_compat_client
from ..core.exceptions import S2ApiError, S2Error, S2ValidationError
from ..core.requests import (
    AuthorBatchDetailsRequest,
    AuthorDetailsRequest,
    AuthorPapersRequest,
    AuthorSearchRequest,
)
from ..mcp import mcp
from ..utils.errors import create_error_response, s2_exception_to_error_response
from ..utils.http import make_request


def _client() -> S2Client:
    return make_compat_client(make_request)


@mcp.tool(description="Search for authors by name with pagination and selectable author metadata fields.")
async def author_search(
    context: Context,
    query: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100,
) -> Dict:
    """Search Semantic Scholar authors by name.

    Returns paginated author matches and lets callers choose which author
    fields to include in the result, such as affiliations, homepage, URL,
    publication counts, and citation metrics.
    """
    try:
        request = AuthorSearchRequest(
            query=query,
            fields=fields,
            offset=offset,
            limit=limit,
        )
        return await _client().search_authors(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool(description="Retrieve detailed metadata for a single author, including metrics and optional paper data.")
async def author_details(
    context: Context,
    author_id: str,
    fields: Optional[List[str]] = None,
) -> Dict:
    """Get detailed information about a specific author.

    Use this when you already know the author ID and need a richer profile,
    including citation counts, h-index, affiliation information, and optionally
    nested paper metadata depending on the selected fields.
    """
    try:
        request = AuthorDetailsRequest(author_id=author_id, fields=fields)
        return await _client().get_author(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2ApiError as exc:
        if "404" in exc.message:
            return create_error_response(
                ErrorType.VALIDATION,
                "Author not found",
                {"author_id": author_id},
            )
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool(description="List papers written by a specific author with pagination and field selection.")
async def author_papers(
    context: Context,
    author_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100,
) -> Dict:
    """Get papers written by a specific author.

    Returns a paginated list of the author's publications and supports choosing
    which paper fields to include in each result, making it suitable for both
    lightweight bibliographies and richer publication summaries.
    """
    try:
        request = AuthorPapersRequest(
            author_id=author_id,
            fields=fields,
            offset=offset,
            limit=limit,
        )
        return await _client().get_author_papers(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2ApiError as exc:
        if "404" in exc.message:
            return create_error_response(
                ErrorType.VALIDATION,
                "Author not found",
                {"author_id": author_id},
            )
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool(description="Retrieve metadata for multiple authors in one batch request.")
async def author_batch_details(
    context: Context,
    author_ids: List[str],
    fields: Optional[str] = None,
) -> Dict:
    """Get details for multiple authors in one request.

    This is the efficient option when you already have a list of author IDs and
    want the same field selection for each record. It returns the same kinds of
    metadata as single-author lookup, but in a batch-friendly form.
    """
    try:
        request = AuthorBatchDetailsRequest(author_ids=author_ids, fields=fields)
        return await _client().batch_authors(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)
