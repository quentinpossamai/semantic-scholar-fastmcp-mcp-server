"""
Paper-related API endpoints for the Semantic Scholar API.
"""

from typing import Dict, List, Optional

from fastmcp import Context

from ..config import ErrorType
from ..core.client import S2Client, make_compat_client
from ..core.exceptions import S2ApiError, S2Error, S2ValidationError
from ..core.requests import (
    PaperAutocompleteRequest,
    PaperAuthorsRequest,
    PaperBatchDetailsRequest,
    PaperBulkSearchRequest,
    PaperCitationsRequest,
    PaperDetailsRequest,
    PaperReferencesRequest,
    PaperRelevanceSearchRequest,
    PaperTitleSearchRequest,
    SnippetSearchRequest,
)
from ..mcp import mcp
from ..utils.errors import create_error_response, s2_exception_to_error_response
from ..utils.http import make_request


def _client() -> S2Client:
    return make_compat_client(make_request)


@mcp.tool(description="Search for papers using relevance-ranked Semantic Scholar results with filters and pagination.")
async def paper_relevance_search(
    context: Context,
    query: str,
    fields: Optional[List[str]] = None,
    publication_types: Optional[List[str]] = None,
    open_access_pdf: bool = False,
    min_citation_count: Optional[int] = None,
    year: Optional[str] = None,
    venue: Optional[List[str]] = None,
    fields_of_study: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 10,
) -> Dict:
    """Search for papers using Semantic Scholar's relevance-based ranking.

    Use this tool for general paper discovery when you have a natural-language
    query and want paginated results with customizable return fields. Supports
    common filters such as publication type, open-access availability, minimum
    citation count, venue, publication year, and field of study.
    """
    try:
        request = PaperRelevanceSearchRequest(
            query=query,
            fields=fields,
            publication_types=publication_types,
            open_access_pdf=open_access_pdf,
            min_citation_count=min_citation_count,
            year=year,
            venue=venue,
            fields_of_study=fields_of_study,
            offset=offset,
            limit=limit,
        )
        return await _client().search_papers(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool(description="Search large paper result sets with bulk pagination and optional sorting.")
async def paper_bulk_search(
    context: Context,
    query: Optional[str] = None,
    token: Optional[str] = None,
    fields: Optional[List[str]] = None,
    sort: Optional[str] = None,
    publication_types: Optional[List[str]] = None,
    open_access_pdf: bool = False,
    min_citation_count: Optional[int] = None,
    publication_date_or_year: Optional[str] = None,
    year: Optional[str] = None,
    venue: Optional[List[str]] = None,
    fields_of_study: Optional[List[str]] = None,
) -> Dict:
    """Search papers in bulk with advanced filtering and sort controls.

    This tool is suited to larger result sets than relevance search and can be
    paged with the Semantic Scholar continuation token. Supports sorting by
    citation count or publication date as well as the same filtering options
    used for standard paper discovery.
    """
    try:
        request = PaperBulkSearchRequest(
            query=query,
            token=token,
            fields=fields,
            sort=sort,
            publication_types=publication_types,
            open_access_pdf=open_access_pdf,
            min_citation_count=min_citation_count,
            publication_date_or_year=publication_date_or_year,
            year=year,
            venue=venue,
            fields_of_study=fields_of_study,
        )
        return await _client().bulk_search_papers(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool(description="Find a paper by title match and return detailed metadata for the best match.")
async def paper_title_search(
    context: Context,
    query: str,
    fields: Optional[List[str]] = None,
    publication_types: Optional[List[str]] = None,
    open_access_pdf: bool = False,
    min_citation_count: Optional[int] = None,
    year: Optional[str] = None,
    venue: Optional[List[str]] = None,
    fields_of_study: Optional[List[str]] = None,
) -> Dict:
    """Find a paper by matching its title against Semantic Scholar records.

    Use this when you know or mostly know the paper title and want the most
    likely matching record rather than a full ranked result list. Supports the
    same field selection and filtering options as other paper search tools.
    """
    try:
        request = PaperTitleSearchRequest(
            query=query,
            fields=fields,
            publication_types=publication_types,
            open_access_pdf=open_access_pdf,
            min_citation_count=min_citation_count,
            year=year,
            venue=venue,
            fields_of_study=fields_of_study,
        )
        return await _client().match_paper_title(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2ApiError as exc:
        if "404" in exc.message:
            return create_error_response(
                ErrorType.VALIDATION,
                "No matching paper found",
                {"original_query": query},
            )
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool(description="Retrieve detailed metadata for a single paper by Semantic Scholar paper ID or external identifier.")
async def paper_details(
    context: Context,
    paper_id: str,
    fields: Optional[List[str]] = None,
) -> Dict:
    """Get detailed information for a single paper.

    Accepts standard Semantic Scholar identifiers and common external paper
    identifiers such as DOI or arXiv IDs when supported by the upstream API.
    Field selection lets callers request only the metadata they need, including
    nested citation and author data.
    """
    try:
        request = PaperDetailsRequest(paper_id=paper_id, fields=fields)
        return await _client().get_paper(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2ApiError as exc:
        if "404" in exc.message:
            return create_error_response(
                ErrorType.VALIDATION,
                "Paper not found",
                {"paper_id": paper_id},
            )
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool(description="Retrieve metadata for multiple papers in one batch request.")
async def paper_batch_details(
    context: Context,
    paper_ids: List[str],
    fields: Optional[str] = None,
) -> Dict:
    """Fetch details for multiple papers in one request.

    This is the efficient option when you already have a list of paper IDs and
    want the same field set for each of them. It supports the same identifier
    formats and field-selection model as the single-paper details endpoint.
    """
    try:
        request = PaperBatchDetailsRequest(paper_ids=paper_ids, fields=fields)
        return await _client().batch_papers(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool(description="List the authors for a given paper with pagination and selectable author fields.")
async def paper_authors(
    context: Context,
    paper_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100,
) -> Dict:
    """Get the authors associated with a specific paper.

    Returns paginated author results for the paper and lets callers choose the
    author metadata fields to include, such as affiliations, URL, publication
    counts, and citation metrics when available.
    """
    try:
        request = PaperAuthorsRequest(
            paper_id=paper_id,
            fields=fields,
            offset=offset,
            limit=limit,
        )
        return await _client().get_paper_authors(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2ApiError as exc:
        if "404" in exc.message:
            return create_error_response(
                ErrorType.VALIDATION,
                "Paper not found",
                {"paper_id": paper_id},
            )
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool(description="List the papers that cite a given paper, including citation context fields when requested.")
async def paper_citations(
    context: Context,
    paper_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100,
) -> Dict:
    """Get papers that cite a specific paper.

    Returns a paginated list of citing papers and can include citation context
    fields such as influence, intents, and surrounding text when those fields
    are requested from the Semantic Scholar API.
    """
    try:
        request = PaperCitationsRequest(
            paper_id=paper_id,
            fields=fields,
            offset=offset,
            limit=limit,
        )
        return await _client().get_paper_citations(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2ApiError as exc:
        if "404" in exc.message:
            return create_error_response(
                ErrorType.VALIDATION,
                "Paper not found",
                {"paper_id": paper_id},
            )
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool(description="List the papers referenced by a given paper, including reference context fields when requested.")
async def paper_references(
    context: Context,
    paper_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100,
) -> Dict:
    """Get the references cited by a specific paper.

    Returns a paginated list of referenced papers and supports the same field
    selection model as citation lookup, including context-oriented fields when
    the upstream API makes them available.
    """
    try:
        request = PaperReferencesRequest(
            paper_id=paper_id,
            fields=fields,
            offset=offset,
            limit=limit,
        )
        return await _client().get_paper_references(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2ApiError as exc:
        if "404" in exc.message:
            return create_error_response(
                ErrorType.VALIDATION,
                "Paper not found",
                {"paper_id": paper_id},
            )
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool(description="Return autocomplete suggestions for a partial paper title query.")
async def paper_autocomplete(
    context: Context,
    query: str,
) -> Dict:
    """Get paper title suggestions for a partial query.

    This is intended for interactive search experiences and lightweight lookup
    flows. The underlying request truncates overly long queries to the API's
    supported length before sending them upstream.
    """
    try:
        request = PaperAutocompleteRequest(query=query)
        return await _client().autocomplete_papers(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool(description="Search matching snippets across Semantic Scholar paper content with optional metadata filters.")
async def snippet_search(
    context: Context,
    query: str,
    fields: Optional[List[str]] = None,
    limit: int = 10,
    paper_ids: Optional[List[str]] = None,
    authors: Optional[List[str]] = None,
    min_citation_count: Optional[int] = None,
    inserted_before: Optional[str] = None,
    publication_date_or_year: Optional[str] = None,
    year: Optional[str] = None,
    venue: Optional[List[str]] = None,
    fields_of_study: Optional[List[str]] = None,
) -> Dict:
    """Search within paper snippets, excerpts, and related text matches.

    Use this when you need text-level matches instead of whole-paper ranking.
    Supports filtering by paper IDs, author names, venue, year, minimum
    citation count, and field of study so callers can narrow snippet results to
    a specific literature slice.
    """
    try:
        request = SnippetSearchRequest(
            query=query,
            fields=fields,
            limit=limit,
            paper_ids=paper_ids,
            authors=authors,
            min_citation_count=min_citation_count,
            inserted_before=inserted_before,
            publication_date_or_year=publication_date_or_year,
            year=year,
            venue=venue,
            fields_of_study=fields_of_study,
        )
        return await _client().search_snippets(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)
