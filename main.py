"""
main.py
=======
Main Entry Point & Interactive CLI for the LangGraph Multi-Agent System.

Integrates:
1. RAG Sub-Agent (Pinecone Serverless + Agentic AI Roadmap PDF)
2. GitHub MCP Sub-Agent (Official Model Context Protocol Server via stdio)
3. Google Workspace MCP Sub-Agent (Composio Google Calendar & Gmail)

Usage:
------
1. Interactive Mode:
   python main.py
   python main.py --interactive

2. Single Query Mode:
   python main.py --query "What are the 4 key components of an AI Agent according to the PDF?"
   python main.py --query "Tell me about the latest commit on octocat/Hello-World repo."
   python main.py --query "Show my upcoming meetings for today."
   python main.py --query "Write a draft email in Gmail to abdullah@gmail.com about project update."

3. View Graph Diagram:
   python main.py --diagram
"""

import sys
import os
import argparse
from typing import List, Optional

# Windows console UTF-8 support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from graph.orchestrator import run_multi_agent_system, build_main_orchestrator

load_dotenv()


def print_banner():
    print("=" * 80)
    print("  🚀 LANGGRAPH MULTI-AGENT ORCHESTRATOR SYSTEM")
    print("  Powered by LangGraph, OpenAI, Pinecone, GitHub MCP, and Google Workspace MCP")
    print("=" * 80)
    print("Sub-Agents Available:")
    print("  📚 [1] RAG Sub-Agent: Q&A over Agentic AI Roadmap PDF (Hosted Pinecone)")
    print("  🐙 [2] GitHub MCP Sub-Agent: Inspect repositories, commits, and code files")
    print("  📅 [3] Google Workspace MCP: Multi-calendar booking/listing + Gmail send/draft")
    print("=" * 80)


def print_diagram():
    """Prints the Mermaid diagram representation of the LangGraph multi-agent orchestrator."""
    app = build_main_orchestrator()
    try:
        mermaid_code = app.get_graph().draw_mermaid()
        print("\n=== Multi-Agent Orchestrator Graph (Mermaid) ===")
        print(mermaid_code)
        print("================================================\n")
    except Exception as e:
        print(f"\nCould not generate Mermaid graph: {e}")
        print("\nGraph Topology:")
        print("  START -> router")
        print("  router -> (conditional edge) -> [rag_agent | github_agent | workspace_agent | general_agent]")
        print("  [rag_agent | github_agent | workspace_agent | general_agent] -> END\n")


def format_agent_badge(route: str) -> str:
    badges = {
        "rag": "📚 [RAG SUB-AGENT]",
        "github": "🐙 [GITHUB MCP SUB-AGENT]",
        "workspace": "📅 [GOOGLE WORKSPACE MCP SUB-AGENT]",
        "general": "💬 [GENERAL ASSISTANT]",
    }
    return badges.get(route, f"🤖 [{route.upper()}]")


def execute_turn(query: str, thread_id: str = "default_session", history: Optional[List[BaseMessage]] = None) -> str:
    """Executes a single user turn, prints diagnostics, and returns the AI answer."""
    print(f"\nThinking & Routing...", end="", flush=True)
    
    result = run_multi_agent_system(query=query, thread_id=thread_id, history=history)
    
    route = result.get("route", "general")
    reasoning = result.get("route_reasoning", "")
    answer = result.get("final_answer", "")
    
    print("\r" + " " * 30 + "\r", end="")  # Clear thinking line
    
    print(f"\n{format_agent_badge(route)}")
    print(f"🎯 Route Decision: '{route}'")
    if reasoning:
        print(f"💡 Router Reasoning: {reasoning}")
    print("-" * 80)
    print(f"🤖 Answer:\n{answer}")
    print("-" * 80)
    
    return answer


def run_interactive_mode():
    """Runs a multi-turn interactive chat loop in the console with MemorySaver session tracking."""
    from datetime import datetime
    session_id = f"session_{datetime.now().strftime('%H%M%S')}"
    
    print_banner()
    print(f"\nInteractive Chat started! (Session Thread: {session_id})")
    print("Type your question or task below.")
    print("Commands: 'exit' or 'quit' to stop, 'clear' to reset chat history.\n")
    
    history: List[BaseMessage] = []
    
    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n👋 Shukriya! Multi-Agent Orchestrator session closed. Allah Hafiz!\n")
                break
                
            if user_input.lower() == "clear":
                history = []
                session_id = f"session_{datetime.now().strftime('%H%M%S')}"
                print(f"\n🔄 Conversation history cleared & new session created: {session_id}")
                continue
                
            answer = execute_turn(query=user_input, thread_id=session_id, history=history)
            
            # Maintain multi-turn history
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=answer))
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


def run_single_query_mode(query: str):
    """Runs a single query from CLI flags and exits."""
    print_banner()
    print(f"\n👤 Input Query: {query}")
    execute_turn(query=query, history=[])


def main():
    parser = argparse.ArgumentParser(description="LangGraph Multi-Agent Orchestrator CLI")
    parser.add_argument("--query", "-q", type=str, help="Single query to process through the multi-agent system")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive multi-turn chat CLI")
    parser.add_argument("--diagram", "-d", action="store_true", help="Display LangGraph Mermaid diagram")
    
    args = parser.parse_args()
    
    if args.diagram:
        print_diagram()
    elif args.query:
        run_single_query_mode(args.query)
    else:
        run_interactive_mode()


if __name__ == "__main__":
    main()
