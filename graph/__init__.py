"""
graph/__init__.py
=================
Main LangGraph Multi-Agent Orchestrator Package.
Connects RAG, GitHub MCP, and Google Workspace (Calendar + Gmail) MCP Sub-Agents.
"""

from graph.state import MainState
from graph.orchestrator import build_main_orchestrator, run_multi_agent_system

__all__ = ["MainState", "build_main_orchestrator", "run_multi_agent_system"]
