#!/usr/bin/env python3
"""MCP server: one typed tool — search_docs — backed by LanceDB hybrid search."""

from __future__ import annotations

import json
from typing import Optional

from mcp.server.fastmcp import FastMCP

from hybrid import hybrid_search, seed_table

mcp = FastMCP("lancedb-hybrid")


@mcp.tool()
def search_docs(
    query: str,
    after_date: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 5,
) -> str:
    """Hybrid-search product docs in LanceDB (vector + full-text + SQL filters).

    Use this when the user asks for documents and may constrain by date or category.
    Examples: pricing docs after March, security notes in 2024-Q2.

    Args:
        query: Natural-language search text (embedded + used for FTS).
        after_date: Optional ISO date (YYYY-MM-DD); keep rows with date > after_date.
        category: Optional category filter, e.g. pricing, product, security, ops.
        limit: Max rows to return (default 5).
    """
    seed_table(force=False)

    clauses: list[str] = []
    if after_date:
        # ISO dates compare correctly as strings under LanceDB SQL.
        clauses.append(f"date > '{after_date}'")
    if category:
        safe = category.replace("'", "")
        clauses.append(f"category = '{safe}'")
    sql_filter = " AND ".join(clauses) if clauses else None

    hits = hybrid_search(query, sql_filter=sql_filter, limit=limit)
    return json.dumps(
        {
            "query": query,
            "sql_filter": sql_filter,
            "hits": hits,
        },
        indent=2,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
