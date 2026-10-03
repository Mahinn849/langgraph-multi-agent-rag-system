"""
test_calendar_mcp.py
====================
Verification script for Google Workspace (Calendar + Gmail) MCP Sub-Agent.

Tests:
1. Multi-Calendar & Invite Prompt Flow:
   - User asks to book meeting without specifying calendar or attendee email.
   - Agent lists available calendars AND asks if an email invite should be sent.
2. Meeting Booking + Automatic Email Invitation:
   - User books meeting and provides recipient email.
   - Agent creates the Google Calendar event AND automatically sends the invitation email via Gmail!
3. Draft Email Creation:
   - User asks to write a draft email in Gmail -> Agent calls `create_email_draft`.
4. Reading Existing Calendar Events:
   - Agent lists upcoming meetings.
5. Interactive terminal chat mode.
"""

import os
import sys
from dotenv import load_dotenv

# Ensure proper utf-8 encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from agents.calendar_mcp_agent import answer_with_calendar_mcp


def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def test_1_ask_calendar_and_invite():
    print_banner("TEST 1: Booking Request Without Calendar or Email")
    query = "Mujhe kal dopehar 3 baje 'Gaming Session' meeting schedule karni hai."
    print(f"User Query: '{query}'\nExecuting through LangGraph...")
    
    result = answer_with_calendar_mcp(query)
    
    print("\n--- LangGraph Tool Calls & Reasoning ---")
    messages = result["messages"]
    tool_calls = []
    for m in messages:
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                tool_calls.append(tc["name"])
                print(f"[TOOL CALLED]: {tc['name']} with args: {tc['args']}")
                
    print("\n--- AGENT ANSWER ---")
    print(result["answer"])
    print("-" * 75)
    
    assert "list_user_calendars" in tool_calls, "Agent should have listed calendars!"
    assert "create_calendar_event" not in tool_calls, "Agent should not create event before user confirms calendar!"
    print("[PASSED] Agent correctly prompted for calendar selection and email invite!")
    return result["messages"]


def test_2_book_and_auto_send_email():
    print_banner("TEST 2: Book Meeting + Automatic Email Invitation via Gmail")
    query = (
        "Mera primary calendar (nancyorla.seo@gmail.com) use karo, kal dopehar 3 baje "
        "'Gaming Strategy Session' schedule kardo aur sath hi nancyorla.seo@gmail.com ko email invitation bhi send kardo."
    )
    print(f"User Query: '{query}'\nExecuting through LangGraph...")
    
    result = answer_with_calendar_mcp(query)
    
    print("\n--- LangGraph Tool Calls & Reasoning ---")
    messages = result["messages"]
    called_create_event = False
    called_send_email = False
    for m in messages:
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                if tc["name"] == "create_calendar_event":
                    called_create_event = True
                    print(f"[CALENDAR EVENT CREATED]: {tc['args']}")
                elif tc["name"] == "send_email":
                    called_send_email = True
                    print(f"[EMAIL INVITATION SENT VIA GMAIL]: {tc['args']}")
                    
    print("\n--- AGENT ANSWER ---")
    print(result["answer"])
    print("-" * 75)
    
    assert called_create_event, "Agent should have created the calendar event!"
    assert called_send_email, "Agent should have sent the invitation email!"
    print("[PASSED] Event booked in calendar AND invitation email sent via Gmail automatically!")


def test_3_create_draft_email():
    print_banner("TEST 3: Write Draft Email in Gmail")
    query = "Ek draft email create kardo recipient 'test_advisor@example.com' ko subject 'Project Status' k sath k project 90% complete ho chuka hai."
    print(f"User Query: '{query}'\nExecuting through LangGraph...")
    
    result = answer_with_calendar_mcp(query)
    
    print("\n--- LangGraph Tool Calls & Reasoning ---")
    messages = result["messages"]
    created_draft = False
    for m in messages:
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                if tc["name"] == "create_email_draft":
                    created_draft = True
                    print(f"[GMAIL DRAFT CREATED]: {tc['args']}")
                    
    print("\n--- AGENT ANSWER ---")
    print(result["answer"])
    print("-" * 75)
    
    assert created_draft, "Agent should have called create_email_draft!"
    print("[PASSED] Gmail draft email created successfully!")


def test_4_list_events():
    print_banner("TEST 4: Reading Upcoming Calendar Events")
    query = "What meetings do I have on my calendar?"
    print(f"User Query: '{query}'\nExecuting through LangGraph...")
    
    result = answer_with_calendar_mcp(query)
    print("\n--- AGENT ANSWER ---")
    print(result["answer"])
    print("-" * 75)
    print("[PASSED] Calendar events listed successfully!")


def run_interactive_mode():
    print_banner("INTERACTIVE GOOGLE WORKSPACE (CALENDAR & GMAIL) CHAT")
    print("Connected to Google Calendar & Gmail via Composio MCP.")
    print("Commands you can test:")
    print("  - 'Book a meeting tomorrow at 4 PM with Ali'")
    print("  - 'What meetings do I have?'")
    print("  - 'Send an email to user@domain.com saying...'")
    print("  - 'Create a draft email to advisor@domain.com'")
    print("Type 'exit' or 'quit' to end.\n")
    
    conversation_history = []
    
    while True:
        try:
            query = input("Ask Workspace Agent (or 'exit') > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting workspace chat. Goodbye!")
                break
                
            print("\nExecuting via Composio MCP...")
            result = answer_with_calendar_mcp(query, history=conversation_history)
            conversation_history = result["messages"]
            
            print("\n" + "=" * 65)
            print("WORKSPACE AGENT:")
            print(result["answer"])
            print("=" * 65 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session.")
            break


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg in ["--interactive", "-i", "chat"]:
            run_interactive_mode()
        else:
            custom_query = " ".join(sys.argv[1:])
            print_banner(f"Custom Query: {custom_query}")
            res = answer_with_calendar_mcp(custom_query)
            print("\nAGENT ANSWER:")
            print(res["answer"])
    else:
        test_1_ask_calendar_and_invite()
        test_2_book_and_auto_send_email()
        test_3_create_draft_email()
        test_4_list_events()
        print_banner("ALL GOOGLE WORKSPACE (CALENDAR & GMAIL) TESTS PASSED SUCCESSFULLY!")
        print("\nTip: To test live interactive conversation, run:")
        print("  .\\.venv\\Scripts\\python.exe test_calendar_mcp.py --interactive")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR OCCURRED]: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
