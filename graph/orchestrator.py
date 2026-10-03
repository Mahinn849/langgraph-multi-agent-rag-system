"""
graph/orchestrator.py
=====================
Main Multi-Agent LangGraph Orchestrator.

Architecture:
-------------
                      [START]
                         │
                         ▼
                  [ router_node ]
                         │
           ┌─────────────┼─────────────┬─────────────┐
           ▼             ▼             ▼             ▼
      [rag_node]   [github_node] [workspace_node] [general_node]
           │             │             │             │
           └─────────────┴──────┬──────┴─────────────┘
                                ▼
                              [END]

Features:
- StateGraph with typed MainState.
- Dynamic routing with conditional edges based on LLM classification.
- Sub-agent integration:
  * RAG Sub-Agent (Hosted Pinecone vector database + Agentic AI roadmap PDF).
  * GitHub MCP Sub-Agent (stdio MCP server for repositories, code, commits).
  * Google Workspace MCP Sub-Agent (Composio Calendar listing/booking/deletion + Gmail sending/drafting/invitations).
- Graceful error handling and beginner-friendly structure.
"""

import os
import sys
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

# Windows console UTF-8 support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from graph.state import MainState
from graph.router import router_node
from agents.rag_agent import answer_with_rag
from agents.github_mcp_agent import run_github_mcp_agent, answer_with_github_mcp
from agents.calendar_mcp_agent import answer_with_calendar_mcp

load_dotenv()


# ==============================================================================
# 1. Sub-Agent Execution Nodes
# ==============================================================================
def rag_node(state: MainState) -> Dict[str, Any]:
    """Invokes the RAG Sub-Agent on the user query using Pinecone hosted vector store."""
    query = state.get("query", "")
    messages = state.get("messages", [])
    
    result = answer_with_rag(query, history=messages)
    answer = result.get("answer", "Maaf kijiye, RAG agent se answer generate nahi ho saka.")
    
    sources = []
    for d in result.get("retrieved_docs", []):
        if hasattr(d, "metadata") and isinstance(d.metadata, dict):
            sources.append(d.metadata.get("source", "PDF"))
        elif isinstance(d, dict):
            sources.append(d.get("source", "PDF"))
        else:
            sources.append("PDF")

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "metadata": {
            **state.get("metadata", {}),
            "executed_by": "rag_sub_agent",
            "sources": sources,
        },
    }


def github_node(state: MainState) -> Dict[str, Any]:
    """Invokes the GitHub MCP Sub-Agent via stdio MCP protocol."""
    query = state.get("query", "")
    
    try:
        result = run_github_mcp_agent(query)
        answer = result.get("answer", "GitHub MCP agent se answer praapt nahi ho saka.")
    except Exception as e:
        answer = f"GitHub MCP Sub-Agent execution error: {str(e)}"

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "metadata": {
            **state.get("metadata", {}),
            "executed_by": "github_mcp_agent",
        },
    }


def workspace_node(state: MainState) -> Dict[str, Any]:
    """Invokes the Google Workspace (Calendar + Gmail) MCP Sub-Agent."""
    query = state.get("query", "")
    messages = state.get("messages", [])
    
    result = answer_with_calendar_mcp(query, history=messages)
    answer = result.get("answer", "Workspace agent se answer praapt nahi ho saka.")
    
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "metadata": {
            **state.get("metadata", {}),
            "executed_by": "workspace_mcp_agent",
        },
    }


def general_node(state: MainState) -> Dict[str, Any]:
    """Handles general conversation, greetings, and system capability explanations."""
    query = state.get("query", "")
    messages = state.get("messages", [])
    
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)
    
    system_prompt = (
        "You are an intelligent, friendly AI Assistant orchestrating three specialized sub-agents:\n"
        "1. RAG Sub-Agent: Answers questions grounded in the Agentic AI roadmap PDF via Pinecone vector store.\n"
        "2. GitHub MCP Sub-Agent: Interacts directly with GitHub (repos, commits, code search, files) using official MCP.\n"
        "3. Google Workspace Sub-Agent: Manages Google Calendar (lists calendars, books meetings, deletes events) "
        "and Gmail (sends direct emails, writes drafts, and dispatches meeting invitations).\n\n"
        "Respond warmly in a helpful and polite tone. Explain your capabilities clearly if the user asks.\n\n"
        "STRICT LANGUAGE MATCHING RULE:\n"
        "- ALWAYS match the language of the user's latest message:\n"
        "  * If the user addresses you in English -> You MUST respond STRICTLY in English. Do NOT reply in Roman Urdu or Hindi.\n"
        "  * If the user addresses you in Roman Urdu / Roman English -> You MUST respond in Roman Urdu (in Latin/English alphabet, e.g. 'Main ek AI assistant hoon...'). Do NOT use Arabic/Urdu script unless the user typed in Arabic/Urdu script.\n"
        "  * If the user addresses you in Urdu script -> Respond in Urdu script.\n"
        "- NEVER respond in Roman Urdu if the user asks in English!"
    )
    
    chat_history = [SystemMessage(content=system_prompt)]
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            chat_history.append(msg)
            
    if not messages or (messages and messages[-1].content != query):
        chat_history.append(HumanMessage(content=query))
        
    response = llm.invoke(chat_history)
    answer = response.content
    
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "metadata": {
            **state.get("metadata", {}),
            "executed_by": "general_node",
        },
    }


# ==============================================================================
# 2. Conditional Edge Router Selector
# ==============================================================================
def route_selector(state: MainState) -> str:
    """Returns the node name matching the classified route in state['route']."""
    route = state.get("route", "general")
    if route == "rag":
        return "rag_agent"
    elif route == "github":
        return "github_agent"
    elif route == "workspace":
        return "workspace_agent"
    else:
        return "general_agent"


from langgraph.checkpoint.memory import MemorySaver

# Singleton MemorySaver instance for persistent session memory across conversation turns
_global_memory_saver = MemorySaver()
_compiled_app_cache = None


# ==============================================================================
# 3. Main StateGraph Assembly & Compilation
# ==============================================================================
def build_main_orchestrator(checkpointer: Optional[Any] = None):
    """
    Constructs and compiles the Main LangGraph Orchestrator.
    
    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. MemorySaver) to persist
                      session states across turns using thread_id.
    
    Workflow Topology:
    [START] -> [router] -> (conditional edge) -> [sub_agent] -> [END]
    """
    workflow = StateGraph(MainState)

    # Add Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("rag_agent", rag_node)
    workflow.add_node("github_agent", github_node)
    workflow.add_node("workspace_agent", workspace_node)
    workflow.add_node("general_agent", general_node)

    # Add Direct Edges
    workflow.add_edge(START, "router")

    # Add Conditional Edges from Router to Sub-Agents
    workflow.add_conditional_edges(
        "router",
        route_selector,
        {
            "rag_agent": "rag_agent",
            "github_agent": "github_agent",
            "workspace_agent": "workspace_agent",
            "general_agent": "general_agent",
        },
    )

    # Sub-Agents all terminate at END
    workflow.add_edge("rag_agent", END)
    workflow.add_edge("github_agent", END)
    workflow.add_edge("workspace_agent", END)
    workflow.add_edge("general_agent", END)

    # Compile Graph (with or without MemorySaver checkpointer)
    if checkpointer is not None:
        orchestrator_app = workflow.compile(checkpointer=checkpointer)
    else:
        orchestrator_app = workflow.compile()
        
    return orchestrator_app


def get_orchestrator_app(use_memory: bool = True):
    """Returns a compiled instance of the orchestrator app, caching the MemorySaver version."""
    global _compiled_app_cache
    if use_memory:
        if _compiled_app_cache is None:
            _compiled_app_cache = build_main_orchestrator(checkpointer=_global_memory_saver)
        return _compiled_app_cache
    return build_main_orchestrator(checkpointer=None)


# ==============================================================================
# 4. Public High-Level Invocation Helper
# ==============================================================================
def run_multi_agent_system(
    query: str,
    thread_id: str = "default_session",
    history: Optional[List[BaseMessage]] = None,
) -> Dict[str, Any]:
    """
    Executes a single user turn through the complete Multi-Agent Orchestrator.
    
    Session Memory with MemorySaver:
    - If `thread_id` is supplied, LangGraph's MemorySaver automatically tracks
      and remembers all previous messages and state updates across turns!
    
    Args:
        query: User's query string.
        thread_id: Unique session identifier for MemorySaver checkpointing.
        history: Optional previous conversation messages for backward compatibility.
        
    Returns:
        Dict containing:
        - query: Original query
        - route: Selected route ('rag', 'github', 'workspace', 'general')
        - route_reasoning: Why the route was chosen
        - final_answer: The final textual response
        - messages: Complete updated message history (including past turns from checkpoint)
        - metadata: Execution diagnostics
        - thread_id: The session thread ID used
    """
    app = get_orchestrator_app(use_memory=True)
    config = {"configurable": {"thread_id": thread_id}}
    
    # If explicit history was passed (for callers that maintain history outside), use it;
    # otherwise pass the new HumanMessage and let MemorySaver load the thread history!
    if history:
        messages = list(history)
        if not messages or messages[-1].content != query:
            messages.append(HumanMessage(content=query))
    else:
        messages = [HumanMessage(content=query)]
        
    initial_state = {
        "messages": messages,
        "query": query,
        "route": "general",
        "route_reasoning": "",
        "final_answer": "",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "thread_id": thread_id,
        },
    }
    
    final_state = app.invoke(initial_state, config=config)
    
    return {
        "query": query,
        "route": final_state.get("route"),
        "route_reasoning": final_state.get("route_reasoning"),
        "final_answer": final_state.get("final_answer"),
        "messages": final_state.get("messages", []),
        "metadata": final_state.get("metadata", {}),
        "thread_id": thread_id,
    }
