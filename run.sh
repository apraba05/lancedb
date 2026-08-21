#!/usr/bin/env bash
# One-command demo: seed LanceDB → hybrid search → MCP tool → LangChain agent.
set -euo pipefail
cd "$(dirname "$0")"

export AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
export AWS_DEFAULT_REGION="$AWS_REGION"
export BEDROCK_CHAT_MODEL="${BEDROCK_CHAT_MODEL:-anthropic.claude-3-haiku-20240307-v1:0}"
export BEDROCK_EMBED_MODEL="${BEDROCK_EMBED_MODEL:-amazon.titan-embed-text-v1}"

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "python3.11 is required (mcp / langchain need >=3.10)." >&2
  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI required for Bedrock (Titan embeddings + Haiku agent)." >&2
  exit 1
fi

if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "AWS credentials required for Bedrock (aws sts get-caller-identity failed)." >&2
  echo "Need bedrock:InvokeModel on:" >&2
  echo "  - ${BEDROCK_EMBED_MODEL}" >&2
  echo "  - ${BEDROCK_CHAT_MODEL}" >&2
  exit 1
fi

echo "==> Python venv + deps"
if [[ ! -d .venv ]]; then
  python3.11 -m venv .venv
fi
.venv/bin/pip install -q -r requirements.txt

echo "==> Seed LanceDB table (Titan embeddings + FTS index)"
.venv/bin/python hybrid.py

echo
echo "==> LangChain agent via MCP tool search_docs"
.venv/bin/python demo.py
