import pytest

import semantic_scholar.api.authors  # noqa: F401
import semantic_scholar.api.papers  # noqa: F401
import semantic_scholar.api.recommendations  # noqa: F401
from semantic_scholar.mcp import mcp


TOOL_NAMES = [
    "paper_relevance_search",
    "paper_bulk_search",
    "paper_title_search",
    "paper_details",
    "paper_batch_details",
    "paper_authors",
    "paper_citations",
    "paper_references",
    "paper_autocomplete",
    "snippet_search",
    "author_search",
    "author_details",
    "author_papers",
    "author_batch_details",
    "get_paper_recommendations_single",
    "get_paper_recommendations_multi",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", TOOL_NAMES)
async def test_registered_tools_have_descriptions(tool_name):
    tool = await mcp._tool_manager.get_tool(tool_name)

    assert tool is not None
    assert tool.description
    assert tool.description.strip()