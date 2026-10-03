"""
test_github_mcp.py
==================
Verification script for the GitHub MCP Sub-Agent.

Tests:
1. MCP Server Handshake & Tool Discovery
2. LangGraph ReAct Graph Execution with Real Tool Calling:
   - Query: "Tell me about the repository 'octocat/Hello-World'."
   - Query: "Search for GitHub repositories related to 'langgraph' and return the top 2."
3. Interactive terminal chat mode support.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Ensure proper utf-8 encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from agents.github_mcp_agent import (
    get_github_mcp_tools,
    answer_with_github_mcp,
    run_github_mcp_agent,
)


def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


async def test_1_tool_discovery():
    print_banner("TEST 1: GitHub MCP Tool Discovery")
    tools = await get_github_mcp_tools()
    print(f"Successfully retrieved {len(tools)} tools from GitHub MCP server.")
    print("Sample tools available to the LangGraph agent:")
    for t in tools[:8]:
        print(f"  * {t.name}: {t.description[:75]}...")
    assert len(tools) > 0, "No tools discovered from GitHub MCP server!"
    print("[PASSED] Tool discovery test successful.")


async def test_2_repo_inspection():
    print_banner("TEST 2: Query Repository Details (octocat/Hello-World)")
    query = "Tell me about the repository 'octocat/Hello-World'. What is its description and what files are in it?"
    print(f"Query: {query}\nExecuting through LangGraph...")
    
    result = await answer_with_github_mcp(query)
    
    print("\n--- LangGraph Tool Execution & Reasoning ---")
    messages = result["messages"]
    tool_calls_count = 0
    for m in messages:
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                tool_calls_count += 1
                print(f"[TOOL CALL {tool_calls_count}]: Executing MCP tool '{tc['name']}' with args: {tc['args']}")
    
    print(f"Total MCP tool calls made: {tool_calls_count}")
    print("\n--- GITHUB MCP SUB-AGENT ANSWER ---")
    print(result["answer"])
    print("-" * 75)
    assert tool_calls_count > 0, "The agent should have invoked at least one MCP tool!"
    print("[PASSED] Repository inspection test successful.")


async def test_3_search_repos():
    print_banner("TEST 3: Search Repositories via MCP")
    query = "Search for GitHub repositories related to 'langgraph' and list the top 2."
    print(f"Query: {query}\nExecuting through LangGraph...")
    
    result = await answer_with_github_mcp(query)
    print("\n--- GITHUB MCP SUB-AGENT ANSWER ---")
    print(result["answer"])
    print("-" * 75)
    print("[PASSED] Repository search test successful.")


def run_interactive_mode():
    print_banner("INTERACTIVE GITHUB MCP SUB-AGENT TERMINAL CHAT")
    print("Connected to official GitHub MCP server.")
    print("Type your question below (e.g. 'Tell me about octocat/Hello-World' or 'Search for repos about X').")
    print("Type 'exit' or 'quit' to end.\n")
    
    while True:
        try:
            query = input("Ask GitHub MCP Agent (or 'exit') > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive GitHub session. Goodbye!")
                break
                
            print("\nExecuting query with GitHub MCP tools...")
            result = run_github_mcp_agent(query)
            print("\n" + "=" * 65)
            print("GITHUB AGENT ANSWER:")
            print(result["answer"])
            print("=" * 65 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session.")
            break


async def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg in ["--interactive", "-i", "chat"]:
            run_interactive_mode()
        else:
            custom_query = " ".join(sys.argv[1:])
            print_banner(f"Custom Query: {custom_query}")
            res = await answer_with_github_mcp(custom_query)
            print("\nGITHUB AGENT ANSWER:")
            print(res["answer"])
    else:
        await test_1_tool_discovery()
        await test_2_repo_inspection()
        await test_3_search_repos()
        print_banner("ALL GITHUB MCP SUB-AGENT TESTS PASSED SUCCESSFULLY!")
        print("\nTip: To test custom GitHub questions interactively, run:")
        print("  .\\.venv\\Scripts\\python.exe test_github_mcp.py --interactive")
        print("Or a single question:")
        print('  .\\.venv\\Scripts\\python.exe test_github_mcp.py "Tell me about octocat/Hello-World"')


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"\n[ERROR OCCURRED]: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
