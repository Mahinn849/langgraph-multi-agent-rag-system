"""
agents/github_mcp_agent.py
==========================
GitHub MCP (Model Context Protocol) Sub-Agent for LangGraph.

Architecture:
1. Spawns the official GitHub MCP server via stdio using `MultiServerMCPClient`.
2. Automatically discovers and converts MCP tools (e.g. search_repositories,
   get_file_contents, list_commits, create_issue) into LangChain BaseTools.
3. Builds a LangGraph ReAct StateGraph:
   - agent_node: LLM (gpt-4o-mini) bound with MCP tools. Decides what tool to run.
   - tools_node: LangGraph ToolNode that executes the MCP tool over JSON-RPC.
   - tools_condition: Conditional edge routing back to agent_node after tool execution,
     or to END when the final response is synthesized.
"""

import os
import sys
import asyncio
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

# Windows console UTF-8 support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

# LangChain & LangGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# LangChain MCP Adapters
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


# ==============================================================================
# 1. State Definition for GitHub MCP Sub-Agent
# ==============================================================================
class GitHubAgentState(MessagesState):
    """
    State for the GitHub MCP Sub-Agent.
    Inherits `messages: list[BaseMessage]` from MessagesState.
    """
    pass


import threading

# ==============================================================================
# 2. MCP Client & Persistent Event Loop
# ==============================================================================
_mcp_client: Optional[MultiServerMCPClient] = None
_github_tools: Optional[List[Any]] = None
_mcp_loop: Optional[asyncio.AbstractEventLoop] = None
_mcp_thread: Optional[threading.Thread] = None


def get_mcp_event_loop() -> asyncio.AbstractEventLoop:
    """
    Returns a dedicated, persistent background event loop for the GitHub MCP client.
    This guarantees that the stdio MCP connection and transports stay open across
    multiple conversational turns without 'Event loop is closed' errors.
    """
    global _mcp_loop, _mcp_thread
    if _mcp_loop is None or _mcp_loop.is_closed():
        _mcp_loop = asyncio.new_event_loop()
        _mcp_thread = threading.Thread(target=_mcp_loop.run_forever, daemon=True)
        _mcp_thread.start()
    return _mcp_loop


def reset_mcp_client():
    """Resets the cached MCP client and tools if a connection reset is needed."""
    global _mcp_client, _github_tools
    _mcp_client = None
    _github_tools = None


def get_mcp_client() -> MultiServerMCPClient:
    """Initializes and returns the MultiServerMCPClient for GitHub."""
    global _mcp_client
    if _mcp_client is not None:
        return _mcp_client

    token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not token or token.startswith("your_"):
        raise ValueError("GITHUB_PERSONAL_ACCESS_TOKEN is not configured in .env file.")

    _mcp_client = MultiServerMCPClient({
        "github": {
            "transport": "stdio",
            "command": "npx.cmd" if sys.platform == "win32" else "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": token,
                "PATH": os.environ.get("PATH", ""),
                "USERPROFILE": os.environ.get("USERPROFILE", ""),
                "APPDATA": os.environ.get("APPDATA", ""),
            }
        }
    })
    return _mcp_client


async def get_github_mcp_tools() -> List[Any]:
    """Fetches and caches the list of LangChain-adapted tools from GitHub MCP server."""
    global _github_tools
    if _github_tools is not None:
        return _github_tools

    client = get_mcp_client()
    tools = await client.get_tools()
    _github_tools = tools
    return _github_tools


# ==============================================================================
# 3. LangGraph Sub-Agent Assembly
# ==============================================================================
async def build_github_mcp_sub_graph():
    """
    Constructs and compiles the GitHub MCP Sub-Agent LangGraph.
    
    Graph Topology:
    [START]
       ↓
    [agent_node] <----------------------+
       ↓ (conditional: tools_condition) |
    [tools_node] -----------------------+
       ↓ (when no more tool calls)
     [END]
    """
    tools = await get_github_mcp_tools()
    
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    llm_with_tools = llm.bind_tools(tools)

    system_instruction = (
        "You are an expert GitHub Assistant with access to the official GitHub MCP server tools.\n"
        "You can inspect repositories, search code, read file contents, check commits, list issues, "
        "and perform actions requested by the user.\n"
        "RULES:\n"
        "1. When asked about a repository or code, use your GitHub MCP tools to inspect real live data.\n"
        "2. To inspect a repository overview, first search or check its details with search_repositories.\n"
        "3. When reading file contents, if a branch 'main' fails, try 'master' or check default branch.\n"
        "4. Provide clear, structured, and helpful responses based on the tool outputs.\n"
        "5. Never guess or make up repository details; always ground answers in the MCP tool results.\n"
        "6. STRICT LANGUAGE RULE: Always respond in the EXACT same language as the user's query. If the user asks in English, respond strictly in English. If the user asks in Roman Urdu, respond in Roman Urdu."
    )

    async def agent_node(state: GitHubAgentState) -> Dict[str, Any]:
        messages = state["messages"]
        # Ensure system prompt is present at the beginning
        if not messages or not isinstance(messages[0], SystemMessage):
            all_messages = [SystemMessage(content=system_instruction)] + messages
        else:
            all_messages = messages

        response = await llm_with_tools.ainvoke(all_messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools, handle_tool_errors=True)

    workflow = StateGraph(GitHubAgentState)

    # Add Nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # Add Edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    # Compile the graph
    app = workflow.compile()
    return app


# ==============================================================================
# 4. Public High-Level Entrypoints
# ==============================================================================
async def answer_with_github_mcp(query: str) -> Dict[str, Any]:
    """
    Asynchronous entry point to query the GitHub MCP Sub-Agent.
    Returns the final state containing the conversation messages and final answer.
    """
    app = await build_github_mcp_sub_graph()
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    final_state = await app.ainvoke(initial_state)
    
    # Extract the last AI response text
    last_message = final_state["messages"][-1]
    final_answer = last_message.content if isinstance(last_message, AIMessage) else str(last_message)
    
    return {
        "query": query,
        "answer": final_answer,
        "messages": final_state["messages"],
    }


def run_github_mcp_agent(query: str) -> Dict[str, Any]:
    """
    Thread-safe synchronous wrapper to run the GitHub MCP Sub-Agent using a dedicated,
    persistent background event loop so that the stdio MCP connection remains
    active across multiple conversational turns without 'Event loop is closed' errors.
    """
    loop = get_mcp_event_loop()
    try:
        future = asyncio.run_coroutine_threadsafe(answer_with_github_mcp(query), loop)
        return future.result()
    except Exception as e:
        if "Event loop is closed" in str(e) or "closed" in str(e).lower():
            reset_mcp_client()
            loop = get_mcp_event_loop()
            future = asyncio.run_coroutine_threadsafe(answer_with_github_mcp(query), loop)
            return future.result()
        raise e
