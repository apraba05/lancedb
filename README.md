# LanceDB-as-MCP-Tool: Agentic Hybrid Search

LLM/agentic tooling (MCP, LangChain, Bedrock) applied directly on top of LanceDB's query surface, proving fluency in both the AI-infra layer and the retrieval semantics LanceDB sells.

**Live demo:** https://lancedb.ashanpraba.com

The demo runs entirely in the browser against seeded data — no API keys,
no accounts, and no external services required.

## Stack

- Python
- LanceDB
- MCP
- LangChain
- AWS Bedrock (Titan embeddings)

## How it works

- A small LanceDB table (10-20 rows) of sample docs with metadata fields (date, category) and Bedrock Titan embeddings.
- Write a LanceDB query function that does vector search + SQL filter + full-text rerank in one call.
- Wrap that function as a single MCP tool (search_docs) with typed input schema.
- Connect a minimal LangChain/MCP client that asks a natural-language question requiring a filter (e.g. 'docs about pricing from after March').
- Run it live: show the agent choosing the tool, the filter it derived, and the LanceDB hybrid result.
- Narrate the query code on screen so every line (embedding, filter push-down, rank fusion) is explainable.

## Running locally

```bash
cd src
bash run.sh
```

Then open the printed URL. A prebuilt static version of the UI lives in
`src/web/` and can be opened directly with no server.
