"""LanceDB hybrid search: Titan embed + SQL prefilter + FTS/vector RRF fusion.

This is the file to narrate on camera. Every line maps to LanceDB's pitch:
vector search, SQL filter push-down, and full-text rank fusion in one call.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import boto3
import lancedb
from lancedb.rerankers import RRFReranker

DB_DIR = Path(__file__).resolve().parent / ".lancedb"
TABLE_NAME = "docs"
EMBED_MODEL = os.environ.get("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v1")
AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

# Small corpus: enough to show filter push-down (date/category) without noise.
SAMPLE_DOCS = [
    {
        "id": "d01",
        "title": "Starter plan pricing",
        "category": "pricing",
        "date": "2024-04-02",
        "text": "Starter plan pricing is $29 per seat per month with a 14-day trial.",
    },
    {
        "id": "d02",
        "title": "Enterprise pricing tiers",
        "category": "pricing",
        "date": "2024-05-18",
        "text": "Enterprise pricing adds SSO, audit logs, and volume discounts above 100 seats.",
    },
    {
        "id": "d03",
        "title": "Legacy pricing sheet",
        "category": "pricing",
        "date": "2024-01-10",
        "text": "Legacy pricing from January listed Starter at $19 — superseded after March.",
    },
    {
        "id": "d04",
        "title": "Usage-based pricing FAQ",
        "category": "pricing",
        "date": "2024-06-01",
        "text": "Usage-based pricing bills embeddings by token; overage is $0.02 per thousand.",
    },
    {
        "id": "d05",
        "title": "Hybrid search overview",
        "category": "product",
        "date": "2024-04-20",
        "text": "Hybrid search combines vector similarity with full-text matching and SQL filters.",
    },
    {
        "id": "d06",
        "title": "Vector index tuning",
        "category": "product",
        "date": "2024-05-05",
        "text": "Tune IVF partitions and PQ for billion-scale vector search latency.",
    },
    {
        "id": "d07",
        "title": "Full-text search notes",
        "category": "product",
        "date": "2024-03-15",
        "text": "Full-text search uses a Tantivy-compatible FTS index on the text column.",
    },
    {
        "id": "d08",
        "title": "Security whitepaper",
        "category": "security",
        "date": "2024-04-11",
        "text": "Encryption at rest, row-level ACLs, and private VPC endpoints for enterprise.",
    },
    {
        "id": "d09",
        "title": "SOC2 controls",
        "category": "security",
        "date": "2024-02-28",
        "text": "SOC2 Type II controls cover access reviews and change management.",
    },
    {
        "id": "d10",
        "title": "Onboarding checklist",
        "category": "ops",
        "date": "2024-04-08",
        "text": "Onboarding checklist: create table, embed docs, build FTS index, run hybrid query.",
    },
    {
        "id": "d11",
        "title": "Pricing calculator guide",
        "category": "pricing",
        "date": "2024-03-22",
        "text": "Pricing calculator estimates monthly cost from seats plus embedding volume.",
    },
    {
        "id": "d12",
        "title": "March pricing webinar",
        "category": "pricing",
        "date": "2024-03-05",
        "text": "March webinar previewed the April pricing refresh for Starter and Growth.",
    },
    {
        "id": "d13",
        "title": "Growth plan pricing",
        "category": "pricing",
        "date": "2024-04-30",
        "text": "Growth plan pricing is $79 per seat and includes hybrid search and rerankers.",
    },
    {
        "id": "d14",
        "title": "Incident response runbook",
        "category": "ops",
        "date": "2024-05-12",
        "text": "Incident response runbook: page on-call, snapshot table, restore from version.",
    },
    {
        "id": "d15",
        "title": "Embedding best practices",
        "category": "product",
        "date": "2024-06-10",
        "text": "Chunk by section, store metadata for SQL filters, and rerank hybrid hits with RRF.",
    },
]


def titan_embed(text: str) -> list[float]:
    """Embed one string with Bedrock Titan — the vector half of hybrid search."""
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    body = json.dumps({"inputText": text})
    resp = client.invoke_model(
        modelId=EMBED_MODEL,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    payload = json.loads(resp["body"].read())
    return payload["embedding"]


def connect_table(create: bool = False):
    db = lancedb.connect(str(DB_DIR))
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    if not create:
        raise FileNotFoundError(f"table {TABLE_NAME!r} missing — run seed_table() first")
    return None


def seed_table(*, force: bool = False) -> int:
    """Build a local LanceDB table with Titan vectors + an FTS index on text."""
    db = lancedb.connect(str(DB_DIR))
    if TABLE_NAME in db.table_names():
        if not force:
            return db.open_table(TABLE_NAME).count_rows()
        db.drop_table(TABLE_NAME)

    rows: list[dict[str, Any]] = []
    for doc in SAMPLE_DOCS:
        vector = titan_embed(doc["text"])
        rows.append({**doc, "vector": vector})

    table = db.create_table(TABLE_NAME, rows)
    # FTS index = the keyword half; hybrid search requires this.
    table.create_fts_index("text", replace=True)
    return len(rows)


def hybrid_search(
    query: str,
    *,
    sql_filter: Optional[str] = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """One LanceDB call: embed → vector + FTS → SQL prefilter → RRF rerank."""
    table = connect_table(create=False)
    # 1) Embedding: Titan turns the natural-language query into a vector.
    query_vector = titan_embed(query)

    # 2) Hybrid query: vector ANN + full-text on the same table.
    q = (
        table.search(query_type="hybrid")
        .vector(query_vector)
        .text(query)
        .limit(limit)
    )

    # 3) SQL filter push-down (prefilter=True): date/category shrink the candidate set
    #    before scoring — LanceDB's metadata+vector differentiator.
    if sql_filter:
        q = q.where(sql_filter, prefilter=True)

    # 4) Rank fusion: RRF merges vector ranks and FTS ranks into _relevance_score.
    q = q.rerank(reranker=RRFReranker())

    rows = q.to_list()
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "id": row.get("id"),
                "title": row.get("title"),
                "category": row.get("category"),
                "date": row.get("date"),
                "text": row.get("text"),
                "relevance": row.get("_relevance_score"),
            }
        )
    return out


if __name__ == "__main__":
    n = seed_table(force=True)
    print(f"seeded {n} docs into {DB_DIR}/{TABLE_NAME}")
    hits = hybrid_search(
        "pricing plans and seat cost",
        sql_filter="date > '2024-03-31' AND category = 'pricing'",
        limit=3,
    )
    print(json.dumps(hits, indent=2))
