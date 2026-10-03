"""
graph/router.py
===============
Intelligent Router Node for the Multi-Agent LangGraph Orchestrator.

Role:
-----
Analyzes the incoming user query and conversation history, and classifies the intent
to route execution to the appropriate specialized sub-agent:
1. 'rag': Queries about the Agentic AI roadmap PDF, AI agent architectures, RAG, tool calling, etc.
2. 'github': Queries interacting with GitHub repositories, code search, commits, files, or issues.
3. 'workspace': Queries interacting with Google Calendar (scheduling, listing, deleting events)
                or Gmail (sending direct emails, saving drafts, attendee invitations).
4. 'general': General conversational queries, greetings, or meta-questions.
"""

import os
from typing import Literal, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from graph.state import MainState

load_dotenv()


class RouteDecision(BaseModel):
    """Structured decision output from the router LLM."""
    route: Literal["rag", "github", "workspace", "general"] = Field(
        description="The specialized sub-agent destination for the query."
    )
    reasoning: str = Field(
        description="Brief 1-sentence explanation of why this route was selected."
    )


ROUTER_SYSTEM_PROMPT = """You are the master routing controller for a multi-agent AI system.
Your job is to classify the user's intent and select the single best specialized sub-agent.

Available Routes:
1. 'rag':
   - Questions about AI agents, agent architectures, memory, reasoning patterns (ReAct, Plan-and-Solve).
   - Questions about LLM tools, embeddings, vector databases, RAG concepts.
   - Any query referencing the roadmap, learning path, or uploaded/saved PDF documentation.
   - Any question asking what PDF is stored or to show/summarize the stored PDF (e.g. 'what PDF do you have?', 'show me stored PDF', 'stored PDF mein kya hai', 'tell me about your saved PDF data', 'summarize the roadmap PDF').

2. 'github':
   - Any operation involving GitHub repositories, searching repos, reading files from GitHub.
   - Viewing commits, commit history, pull requests, or issues on GitHub.
   - Example repos: 'octocat/Hello-World', user's repositories, or generic GitHub code queries.

3. 'workspace':
   - Google Calendar operations: booking meetings, appointments, listing upcoming events, deleting meetings.
   - Gmail operations: sending emails to recipients, creating/writing draft emails in Gmail.
   - Meeting invitations, attendee coordination, checking availability.

4. 'general':
   - Greetings ('hello', 'hi', 'salam', 'kya haal hai'), capabilities inquiries ('what can you do?').
   - General conversation that does not require external tools or the PDF knowledge base.

Analyze the user's query carefully and return your structured decision.
"""


def get_router_llm():
    """Initializes the LLM configured with structured output for routing."""
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    return llm.with_structured_output(RouteDecision)


def router_node(state: MainState) -> Dict[str, Any]:
    """
    LangGraph Node: Evaluates state['query'] and decides the routing branch.
    Updates the state with 'route' and 'route_reasoning'.
    """
    query = state.get("query", "")
    if not query and state.get("messages"):
        last_msg = state["messages"][-1]
        query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    router_chain = get_router_llm()
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    try:
        decision: RouteDecision = router_chain.invoke(messages)
        route = decision.route
        reasoning = decision.reasoning
    except Exception as e:
        # Resilient fallback
        route = "general"
        reasoning = f"Defaulted to general due to routing error: {str(e)}"

    return {
        "route": route,
        "route_reasoning": reasoning,
        "metadata": {
            **state.get("metadata", {}),
            "router_selected": route,
            "router_reasoning": reasoning,
        },
    }
