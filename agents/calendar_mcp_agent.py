"""
agents/calendar_mcp_agent.py
============================
Google Workspace (Calendar & Gmail) MCP Sub-Agent powered by Composio.

Core Capabilities:
1. Google Calendar Integration:
   - Lists user calendars (preventing blind default to primary).
   - Prompts user to select which calendar to use if not specified.
   - Lists upcoming meetings & events.
   - Creates appointments at specific dates & times.
   - Deletes / cancels scheduled events.
2. Gmail Email Integration:
   - Directly writes and sends emails to any specific recipient (`send_email`).
   - Writes and saves draft emails in Gmail (`create_email_draft`).
   - Automatically sends an invitation email to meeting attendees when scheduling events.
   - If user doesn't provide an attendee email when booking a meeting, politely asks
     if an email invitation should be sent and requests their email address.
"""

import os
import sys
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

# Windows console UTF-8 support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

# LangChain & LangGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# Composio SDK
from composio import Composio

load_dotenv()


# ==============================================================================
# 1. State Definition
# ==============================================================================
class CalendarAgentState(MessagesState):
    """
    State for the Google Workspace (Calendar + Gmail) MCP Sub-Agent.
    Tracks conversation messages, tool executions, and LLM synthesis.
    """
    pass


# ==============================================================================
# 2. Composio Connections (Calendar & Gmail)
# ==============================================================================
_composio_client: Optional[Composio] = None
_calendar_account: Optional[Dict[str, Any]] = None
_gmail_account: Optional[Dict[str, Any]] = None


def get_composio_client() -> Composio:
    """Returns singleton Composio client initialized with COMPOSIO_API_KEY from .env."""
    global _composio_client
    if _composio_client is not None:
        return _composio_client

    api_key = os.getenv("COMPOSIO_API_KEY")
    if not api_key or api_key.startswith("your_"):
        raise ValueError("COMPOSIO_API_KEY is not configured in .env file.")

    _composio_client = Composio(api_key=api_key)
    return _composio_client


def get_account_for_toolkit(target_slug: str) -> Dict[str, Any]:
    """Retrieves the active connected account for a given toolkit (googlecalendar or gmail)."""
    composio = get_composio_client()
    response = composio.connected_accounts.list()
    items = getattr(response, "items", None) or []
    
    for acc in items:
        toolkit = getattr(acc, "toolkit", None)
        slug = getattr(toolkit, "slug", "") if toolkit else ""
        status = str(getattr(acc, "status", "")).upper()
        if target_slug in slug and status in ["ACTIVE", "CONNECTED"]:
            return {
                "id": getattr(acc, "id"),
                "user_id": getattr(acc, "user_id"),
                "status": status,
            }
            
    # Fallback to first matching slug
    for acc in items:
        toolkit = getattr(acc, "toolkit", None)
        slug = getattr(toolkit, "slug", "") if toolkit else ""
        if target_slug in slug:
            return {
                "id": getattr(acc, "id"),
                "user_id": getattr(acc, "user_id"),
                "status": str(getattr(acc, "status", "")),
            }

    raise RuntimeError(f"No connected account found for '{target_slug}' in Composio project.")


def get_calendar_account() -> Dict[str, Any]:
    global _calendar_account
    if _calendar_account is None:
        conn_id = os.getenv("CALENDAR_CONNECTION_ID")
        user_id = os.getenv("COMPOSIO_USER_ID", "default")
        if conn_id and not conn_id.startswith("your_"):
            _calendar_account = {
                "id": conn_id,
                "user_id": user_id,
                "status": "ACTIVE",
            }
        else:
            _calendar_account = get_account_for_toolkit("googlecalendar")
    return _calendar_account


def get_gmail_account() -> Dict[str, Any]:
    global _gmail_account
    if _gmail_account is None:
        conn_id = os.getenv("GMAIL_CONNECTION_ID")
        user_id = os.getenv("COMPOSIO_USER_ID", "default")
        if conn_id and not conn_id.startswith("your_"):
            _gmail_account = {
                "id": conn_id,
                "user_id": user_id,
                "status": "ACTIVE",
            }
        else:
            _gmail_account = get_account_for_toolkit("gmail")
    return _gmail_account


# ==============================================================================
# 3. Calendar Tools
# ==============================================================================
@tool
def list_user_calendars() -> List[Dict[str, Any]]:
    """
    Lists all available Google Calendars in the user's account.
    Call this tool whenever the user wants to book an appointment or check events
    so you know which calendars exist (e.g. Primary, Work, Personal, etc.).
    Returns a list of calendars with their names, IDs, and whether they are primary.
    """
    composio = get_composio_client()
    account = get_calendar_account()
    
    result = composio.tools.execute(
        slug="GOOGLECALENDAR_LIST_CALENDARS",
        arguments={"max_results": 250, "show_hidden": True},
        connected_account_id=account["id"],
        user_id=account["user_id"],
        dangerously_skip_version_check=True,
    )
    
    data = result.get("data", {}) if isinstance(result, dict) else getattr(result, "data", {})
    raw_calendars = data.get("calendars") or data.get("items") or []
    
    calendars = []
    for c in raw_calendars:
        summary = c.get("summary") or c.get("summaryOverride") or "(No Name)"
        cid = c.get("id")
        is_primary = bool(c.get("primary", False))
        calendars.append({
            "name": summary,
            "id": cid,
            "is_primary": is_primary,
            "description": c.get("description", ""),
        })
        
    return calendars


@tool
def list_calendar_events(calendar_id: str = "primary", max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieves upcoming events/meetings from a specific Google Calendar.
    Args:
        calendar_id: The ID of the calendar to check (use 'primary' or an exact calendar ID from list_user_calendars).
        max_results: Maximum number of events to return (default 10).
    """
    composio = get_composio_client()
    account = get_calendar_account()
    
    result = composio.tools.execute(
        slug="GOOGLECALENDAR_EVENTS_LIST",
        arguments={
            "calendar_id": calendar_id,
            "max_results": max_results,
        },
        connected_account_id=account["id"],
        user_id=account["user_id"],
        dangerously_skip_version_check=True,
    )
    
    data = result.get("data", {}) if isinstance(result, dict) else getattr(result, "data", {})
    raw_events = data.get("items") or data.get("events") or []
    
    events = []
    for e in raw_events:
        events.append({
            "id": e.get("id"),
            "summary": e.get("summary", "(No Title)"),
            "start": e.get("start", {}).get("dateTime") or e.get("start", {}).get("date"),
            "end": e.get("end", {}).get("dateTime") or e.get("end", {}).get("date"),
            "htmlLink": e.get("htmlLink", ""),
        })
        
    return events


@tool
def create_calendar_event(
    summary: str,
    start_datetime: str,
    calendar_id: str,
    duration_hours: int = 1,
    duration_minutes: int = 0,
    description: str = "",
    timezone: str = "Asia/Karachi",
    attendee_email: Optional[str] = None,
    user_confirmed_without_email: bool = False,
) -> Dict[str, Any]:
    """
    Creates/books a new meeting or appointment in a specific Google Calendar.

    CRITICAL RULES:
    1. NEVER CALL THIS TOOL ON AN INITIAL BOOKING REQUEST if the user has not yet been asked about the email invitation!
    2. Before calling this tool, you MUST have already presented the calendar to the user AND asked if they want to send an email invitation to the person/attendee (requesting their email address).
    3. You can ONLY call this tool if:
       - Either `attendee_email` is provided (e.g. 'person@gmail.com').
       - OR `user_confirmed_without_email=True` (user explicitly said 'no' or declined sending an email invitation).
    If neither condition is met, DO NOT call this tool — ask the user the invitation question first!
    """
    # Guard check: Block creation if neither email was provided nor user confirmed booking without email
    if not attendee_email and not user_confirmed_without_email:
        return {
            "status": "blocked",
            "message": (
                "ACTION REQUIRED: Meeting event creation is BLOCKED! "
                "You are NOT allowed to create the calendar event yet. "
                "You MUST first present the available calendar(s) to the user and ALWAYS explicitly ask: "
                "'Would you like to send an email invitation to [Person / attendee]? If yes, please provide their email address.' "
                "Do NOT call create_calendar_event again until the user answers this question!"
            ),
        }

    composio = get_composio_client()
    account = get_calendar_account()
    
    result = composio.tools.execute(
        slug="GOOGLECALENDAR_CREATE_EVENT",
        arguments={
            "calendar_id": calendar_id,
            "summary": summary,
            "start_datetime": start_datetime,
            "timezone": timezone,
            "event_duration_hour": duration_hours,
            "event_duration_minutes": duration_minutes,
            "description": description,
            "create_meeting_room": False,
        },
        connected_account_id=account["id"],
        user_id=account["user_id"],
        dangerously_skip_version_check=True,
    )
    
    data = result.get("data", {}) if isinstance(result, dict) else getattr(result, "data", {})
    resp_data = data.get("response_data", {})
    event_html_link = resp_data.get("htmlLink") or data.get("display_url", "")
    
    # Automatically send invitation email if attendee_email is provided
    email_dispatched = False
    if attendee_email and "@" in attendee_email:
        try:
            send_email.invoke({
                "recipient_email": attendee_email,
                "subject": f"Invitation: {summary}",
                "body": (
                    f"Hello,\n\nYou have been invited to the following meeting:\n\n"
                    f"• Title: {summary}\n"
                    f"• Date & Time: {start_datetime} ({timezone})\n"
                    f"• Duration: {duration_hours} hour(s) {duration_minutes} minute(s)\n"
                    f"• Calendar Link: {event_html_link}\n\n"
                    f"Best regards,\nAutomated Scheduling Assistant"
                )
            })
            email_dispatched = True
        except Exception:
            email_dispatched = False

    return {
        "status": "success" if result.get("successful") else "failed",
        "id": resp_data.get("id"),
        "summary": resp_data.get("summary", summary),
        "htmlLink": event_html_link,
        "start": resp_data.get("start", {}).get("dateTime") or start_datetime,
        "calendar_id": calendar_id,
        "attendee_email": attendee_email,
        "email_dispatched": email_dispatched,
    }


@tool
def delete_calendar_event(calendar_id: str, event_id: str) -> Dict[str, Any]:
    """
    Deletes an existing event/meeting from a Google Calendar.
    Args:
        calendar_id: The ID of the calendar where the event exists (use 'primary' or specific calendar ID).
        event_id: The unique identifier of the event to delete.
    """
    composio = get_composio_client()
    account = get_calendar_account()
    
    result = composio.tools.execute(
        slug="GOOGLECALENDAR_DELETE_EVENT",
        arguments={
            "calendar_id": calendar_id,
            "event_id": event_id,
        },
        connected_account_id=account["id"],
        user_id=account["user_id"],
        dangerously_skip_version_check=True,
    )
    
    return {
        "status": "success" if result.get("successful") else "failed",
        "deleted_event_id": event_id,
        "calendar_id": calendar_id,
    }


# ==============================================================================
# 4. Gmail Email Tools
# ==============================================================================
@tool
def send_email(recipient_email: str, subject: str, body: str) -> Dict[str, Any]:
    """
    Sends an email directly to a recipient using Gmail.
    Use this to send direct emails or automatic meeting invitations.
    Args:
        recipient_email: The target email address (e.g. 'person@example.com').
        subject: Subject line of the email.
        body: Content/body of the message (plain text or HTML).
    """
    composio = get_composio_client()
    account = get_gmail_account()
    
    result = composio.tools.execute(
        slug="GMAIL_SEND_EMAIL",
        arguments={
            "recipient_email": recipient_email,
            "subject": subject,
            "body": body,
            "is_html": False,
        },
        connected_account_id=account["id"],
        user_id=account["user_id"],
        dangerously_skip_version_check=True,
    )
    
    return {
        "status": "success" if result.get("successful") else "failed",
        "recipient": recipient_email,
        "subject": subject,
        "details": result.get("data", {}),
    }


@tool
def create_email_draft(recipient_email: str, subject: str, body: str) -> Dict[str, Any]:
    """
    Creates and saves a draft email in Gmail without sending it immediately.
    Args:
        recipient_email: Target recipient email address.
        subject: Draft email subject.
        body: Draft email body text.
    """
    composio = get_composio_client()
    account = get_gmail_account()
    
    result = composio.tools.execute(
        slug="GMAIL_CREATE_EMAIL_DRAFT",
        arguments={
            "recipient_email": recipient_email,
            "subject": subject,
            "body": body,
            "is_html": False,
        },
        connected_account_id=account["id"],
        user_id=account["user_id"],
        dangerously_skip_version_check=True,
    )
    
    data = result.get("data", {}) if isinstance(result, dict) else getattr(result, "data", {})
    return {
        "status": "success" if result.get("successful") else "failed",
        "draft_id": data.get("id"),
        "display_url": data.get("display_url"),
        "recipient": recipient_email,
        "subject": subject,
    }


WORKSPACE_TOOLS = [
    list_user_calendars,
    list_calendar_events,
    create_calendar_event,
    delete_calendar_event,
    send_email,
    create_email_draft,
]


# ==============================================================================
# 5. LangGraph Sub-Agent Assembly
# ==============================================================================
def build_calendar_mcp_sub_graph():
    """
    Constructs and compiles the Google Workspace (Calendar + Gmail) MCP Sub-Agent LangGraph.
    
    Graph Topology:
    [START] -> [agent] <---> [tools] -> [END]
    """
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    llm_with_tools = llm.bind_tools(WORKSPACE_TOOLS)

    current_date_str = datetime.now().strftime("%Y-%m-%d (%A)")
    
    system_prompt = (
        f"You are an expert Google Workspace Assistant (Calendar & Gmail) powered by Composio MCP tools.\n"
        f"Today's date is: {current_date_str}. User's timezone is Asia/Karachi.\n\n"
        "CAPABILITIES:\n"
        "1. Google Calendar: List calendars, list upcoming meetings, create events at specific times, delete events.\n"
        "2. Gmail: Send emails directly (`send_email`), write & save draft emails (`create_email_draft`), and send automatic meeting invitations.\n\n"
        "STRICT LANGUAGE MATCHING RULE:\n"
        "- ALWAYS detect the language of the user's latest message and respond in the EXACT same language:\n"
        "  * If the user speaks in English -> You MUST respond STRICTLY in English. Do NOT use Roman Urdu or Hindi.\n"
        "  * If the user speaks in Roman Urdu / Roman English -> You MUST respond in Roman Urdu (using Latin/English alphabet).\n"
        "  * If the user speaks in Urdu script -> You MUST respond in Urdu script.\n"
        "- NEVER respond in Roman Urdu if the user's query is in English!\n\n"
        "CRITICAL WORKFLOW RULES:\n\n"
        "A. CALENDAR APPOINTMENT BOOKING & INVITATION WORKFLOW (MANDATORY 2-STEP PROTOCOL EVERY TIME):\n"
        "1. MANDATORY STEP 1 - CALENDAR DETECTION & EMAIL INVITATION INQUIRY (NEVER SKIP THIS):\n"
        "   - Whenever the user asks to schedule, book, or create an appointment/meeting (e.g. 'Please create an appointment on my calender today at 8 pm , meeting name project decision with Mahin'):\n"
        "     a) You MUST call `list_user_calendars` first to check available calendars.\n"
        "     b) DO NOT call `create_calendar_event`! Creating the event in this step is STRICTLY FORBIDDEN and will be blocked by the tool!\n"
        "     c) In your response, you MUST ALWAYS provide BOTH of these points:\n"
        "        1. State the detected calendar (e.g. Primary calendar: nancyorla.seo@gmail.com).\n"
        "        2. MANDATORY EMAIL INVITATION QUESTION (YOU MUST ALWAYS ASK THIS IN EVERY SINGLE BOOKING):\n"
        "           - If a person's name is in the prompt or title (e.g. Mahin, Abdullah, Ali, etc.):\n"
        "             * English: 'Would you like to send an email invitation to [Person]? If yes, please provide their email address.'\n"
        "             * Roman Urdu: 'Kya aap [Person] ko is meeting ki email invitation bhejna chahte hain? Agar haan to unka email address bata dein.'\n"
        "           - If no specific person's name is mentioned:\n"
        "             * English: 'Would you like to send an email invitation to anyone for this meeting? If yes, please provide their email address.'\n"
        "             * Roman Urdu: 'Kya aap is meeting k liye kisi ko email invitation bhejna chahte hain? Agar haan to unka email address bata dein.'\n"
        "     d) You MUST STOP here and wait for the user's answer before creating any event!\n\n"
        "2. STEP 2 - EVENT CREATION AFTER USER RESPONDS:\n"
        "   - When the user answers your confirmation question:\n"
        "     * If user provides an email (e.g. 'mahin@gmail.com' or 'yes, send to mahin@example.com'):\n"
        "       - Call `create_calendar_event` with `attendee_email='<email>'`.\n"
        "       - Call `send_email` to send the invitation with meeting details.\n"
        "       - Confirm both the calendar booking and the email invitation dispatch to the user.\n"
        "     * If user declines (e.g. 'no', 'nahi', 'proceed without email', 'just book it'):\n"
        "       - Call `create_calendar_event` with `user_confirmed_without_email=True`.\n"
        "       - Confirm the event booking on the calendar without email.\n\n"
        "3. ONLY ONE EXCEPTION TO STEP 1:\n"
        "   - ONLY if the user already provided BOTH the calendar name AND an explicit email address in their very first prompt (e.g. 'Book meeting at 3pm on primary calendar with Ali and invite ali@example.com'), can you create the event and send the invitation email in Turn 1.\n\n"
        "B. EMAIL REQUESTS:\n"
        "1. DIRECT EMAIL: If user asks to send an email to someone (e.g. 'Send an email to user@domain.com saying...'), call `send_email`.\n"
        "2. DRAFT EMAIL: If user asks to write/create a draft email (e.g. 'Write a draft email to...'), call `create_email_draft`.\n\n"
        "C. READING & DELETING EVENTS:\n"
        "1. List meetings with `list_calendar_events`.\n"
        "2. To delete an event, find its ID using `list_calendar_events`, then call `delete_calendar_event`."
    )

    def agent_node(state: CalendarAgentState) -> Dict[str, Any]:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            all_messages = [SystemMessage(content=system_prompt)] + messages
        else:
            all_messages = messages

        response = llm_with_tools.invoke(all_messages)
        return {"messages": [response]}

    tool_node = ToolNode(WORKSPACE_TOOLS, handle_tool_errors=True)

    workflow = StateGraph(CalendarAgentState)

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
# 6. Public High-Level Entrypoint
# ==============================================================================
def answer_with_calendar_mcp(query: str, history: Optional[List[BaseMessage]] = None) -> Dict[str, Any]:
    """
    Main entry point to run queries on the Google Workspace (Calendar + Gmail) MCP Sub-Agent.
    Supports multi-turn conversations through `history`.
    """
    app = build_calendar_mcp_sub_graph()
    messages = list(history or [])
    if not messages or messages[-1].content != query:
        messages.append(HumanMessage(content=query))
    
    final_state = app.invoke({"messages": messages})
    last_msg = final_state["messages"][-1]
    answer = last_msg.content if isinstance(last_msg, AIMessage) else str(last_msg)
    
    return {
        "query": query,
        "answer": answer,
        "messages": final_state["messages"],
    }
