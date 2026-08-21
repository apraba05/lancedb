#!/usr/bin/env python3
"""LangChain agent → MCP search_docs → LanceDB hybrid search.

Asks a filtered natural-language question and prints the tool call + answer.
Agent LLM and Titan embeddings both go through AWS Bedrock.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_aws import ChatBedrock
from langchain_mcp_adapters.client import MultiServerMCPClient

ROOT = Path(__file__).resolve().parent
AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
CHAT_MODEL = os.environ.get(
    "BEDROCK_CHAT_MODEL",
    "anthropic.claude-3-haiku-20240307-v1:0",
)
QUESTION = os.environ.get(
    "DEMO_QUESTION",
    "Find docs about pricing from after March 2024. Summarize what you find.",
)


async def main() -> None:
    client = MultiServerMCPClient(
        {
            "lancedb": {
                "command": sys.executable,
                "args": [str(ROOT / "mcp_server.py")],
                "transport": "stdio",
                "cwd": str(ROOT),
            }
        }
    )
    tools = await client.get_tools()
    print("MCP tools:", [t.name for t in tools])

    model = ChatBedrock(
        model_id=CHAT_MODEL,
        region_name=AWS_REGION,
        model_kwargs={"temperature": 0},
    )
    agent = create_agent(
        model,
        tools,
        system_prompt=(
            "You are a docs assistant. For any document lookup you MUST call the "
            "search_docs tool. Derive after_date / category from the user question "
            "when present (e.g. 'after March 2024' → after_date='2024-03-31', "
            "category='pricing'). Then answer briefly from the tool hits."
        ),
    )

    print("\n=== User question ===")
    print(QUESTION)

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": QUESTION}]},
    )

    print("\n=== Agent trace (tool calls + final answer) ===")
    for msg in result["messages"]:
        role = getattr(msg, "type", msg.__class__.__name__)
        if role == "human":
            continue
        if role == "ai":
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    print("\n→ tool call:", tc["name"])
                    print(json.dumps(tc.get("args", {}), indent=2))
            content = msg.content
            if isinstance(content, list):
                texts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                content = "\n".join(t for t in texts if t)
            if content:
                print("\n→ model:", content)
        elif role == "tool":
            print("\n← tool result:")
            body = msg.content
            if isinstance(body, str) and len(body) > 2000:
                body = body[:2000] + "\n…(truncated)"
            print(body)


if __name__ == "__main__":
    asyncio.run(main())
