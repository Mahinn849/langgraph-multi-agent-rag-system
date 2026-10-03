"""
graph/state.py
==============
Global Shared State for the Main Multi-Agent LangGraph Orchestrator.

Concept Explanation (Beginner Friendly):
----------------------------------------
In LangGraph, 'State' is a centralized data structure (like a shared dictionary or dataclass)
that flows through every node in the graph. Every node receives the current state, performs its task,
and returns an updated dictionary of fields to merge into the state.

Comparison to OpenAI Agents SDK / Swarm:
- In OpenAI Agents SDK, context is passed via `agent.run(messages=..., context=...)` or runner variables.
- In LangGraph, state is explicitly typed with TypedDict, and fields can have 'reducers'
  (like `add_messages`) so new messages are automatically appended instead of overwriting history.
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class MainState(TypedDict):
    """
    Main State schema shared across all nodes in the Orchestrator graph.
    
    Fields:
        messages: Full conversation history, automatically managed with `add_messages` reducer.
        query: The raw query text provided by the user.
        route: The classification decided by the Router Node:
               - 'rag': PDF knowledge extraction (Agentic AI roadmap, concepts, etc.)
               - 'github': GitHub MCP repository operations (commits, files, issues, search)
               - 'workspace': Google Calendar meetings & Gmail messaging (send, draft, invites)
               - 'general': General chit-chat or fallback queries
        route_reasoning: Brief rationale from the router explaining why a sub-agent was chosen.
        final_answer: The synthesized textual response delivered to the user.
        metadata: Diagnostic metadata such as execution timestamps, node name, etc.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    route: Literal["rag", "github", "workspace", "general"]
    route_reasoning: str
    final_answer: str
    metadata: Dict[str, Any]
