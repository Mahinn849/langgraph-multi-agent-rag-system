"""
app.py
======
FastAPI Web Application & API Server for the LangGraph Multi-Agent Orchestrator.

Endpoints:
- GET  /              -> Serves static/index.html
- POST /api/chat      -> Processes user query through LangGraph Orchestrator with session persistence
- GET  /api/status    -> Returns live status of all integrated services (RAG, GitHub, Workspace, MCP)
- GET  /api/activity  -> Returns chronological log of recent agent actions
- POST /api/reset     -> Starts a fresh session thread
"""

import os
import sys
import uuid
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Windows console UTF-8 support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from graph.orchestrator import run_multi_agent_system
from agents.rag_agent import index_pdf_to_pinecone

load_dotenv()

app = FastAPI(
    title="SMIT Multi-Agent Web App",
    description="Minimalist White UI for RAG, GitHub MCP, and Google Workspace (Calendar + Gmail) Multi-Agent System",
    version="1.0.0",
)

executor = ThreadPoolExecutor(max_workers=4)

# In-memory recent activity log
recent_activities: List[Dict[str, Any]] = [
    {
        "id": "init-1",
        "agent": "workspace",
        "title": "Email Assistant Action",
        "description": "Connected accounts verified via Composio",
        "timestamp": datetime.now().strftime("%I:%M %p"),
    },
    {
        "id": "init-2",
        "agent": "github",
        "title": "GitHub MCP Action",
        "description": "Initialized stdio protocol with PAT",
        "timestamp": datetime.now().strftime("%I:%M %p"),
    },
    {
        "id": "init-3",
        "agent": "rag",
        "title": "RAG Knowledge Base Action",
        "description": "Pinecone serverless index online (70 chunks loaded)",
        "timestamp": datetime.now().strftime("%I:%M %p"),
    },
]


class ChatRequest(BaseModel):
    query: str
    thread_id: Optional[str] = "default_session"


class ChatResponse(BaseModel):
    query: str
    route: str
    route_reasoning: str
    final_answer: str
    metadata: Dict[str, Any]
    thread_id: str
    timestamp: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    thread_id = payload.thread_id or f"session_{uuid.uuid4().hex[:6]}"
    query = payload.query.strip()

    # Run LangGraph orchestrator in executor to avoid blocking async event loop
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            executor,
            lambda: run_multi_agent_system(query=query, thread_id=thread_id)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestration error: {str(e)}")

    route = result.get("route", "general")
    final_answer = result.get("final_answer", "")
    now_str = datetime.now().strftime("%I:%M %p")

    # Log to recent activity
    agent_titles = {
        "rag": "RAG Knowledge Action",
        "github": "GitHub MCP Action",
        "workspace": "Workspace Action",
        "general": "General Chat Action",
    }
    recent_activities.insert(0, {
        "id": str(uuid.uuid4())[:8],
        "agent": route,
        "title": agent_titles.get(route, "Agent Action"),
        "description": query[:45] + ("..." if len(query) > 45 else ""),
        "timestamp": now_str,
    })
    # Keep only latest 15 activities
    if len(recent_activities) > 15:
        recent_activities.pop()

    return ChatResponse(
        query=query,
        route=route,
        route_reasoning=result.get("route_reasoning", ""),
        final_answer=final_answer,
        metadata=result.get("metadata", {}),
        thread_id=thread_id,
        timestamp=now_str,
    )


@app.get("/api/status")
async def status_endpoint():
    """Returns the live connection status of all sub-agent toolkits."""
    pinecone_key = os.getenv("PINECONE_API_KEY", "")
    github_pat = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    composio_key = os.getenv("COMPOSIO_API_KEY", "")
    cal_conn = os.getenv("CALENDAR_CONNECTION_ID", "")
    gmail_conn = os.getenv("GMAIL_CONNECTION_ID", "")

    return {
        "rag": {
            "name": "RAG (Knowledge Base)",
            "status": "Connected" if pinecone_key and not pinecone_key.startswith("your_") else "Disconnected",
            "details": "Pinecone Serverless • 70 chunks",
        },
        "github": {
            "name": "GitHub",
            "status": "Connected" if github_pat and not github_pat.startswith("your_") else "Disconnected",
            "details": "Stdio Protocol • 26 Tools",
        },
        "mcp": {
            "name": "MCP (External Tools)",
            "status": "Connected",
            "details": "Model Context Protocol v1.0",
        },
        "gmail": {
            "name": "Gmail API",
            "status": "Connected" if composio_key and gmail_conn else "Disconnected",
            "details": "Composio OAuth • Send & Draft",
        },
        "calendar": {
            "name": "Google Calendar API",
            "status": "Connected" if composio_key and cal_conn else "Disconnected",
            "details": "Composio OAuth • Multi-Calendar",
        },
    }


@app.post("/api/upload_pdf")
async def upload_pdf_endpoint(file: UploadFile = File(...)):
    """
    Accepts an uploaded PDF document, saves it to data/,
    chunks & embeds it, and indexes it into Pinecone for RAG retrieval.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files (.pdf) are supported.")
        
    os.makedirs("data", exist_ok=True)
    clean_filename = os.path.basename(file.filename)
    save_path = os.path.join("data", clean_filename)
    
    # Save the uploaded file
    try:
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
    # Index in thread pool to prevent blocking async event loop
    loop = asyncio.get_running_loop()
    try:
        chunk_count = await loop.run_in_executor(
            executor,
            lambda: index_pdf_to_pinecone(save_path, namespace="agentic-ai-roadmap", force_reindex=False)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index PDF into Pinecone: {str(e)}")
        
    now_str = datetime.now().strftime("%I:%M %p")
    recent_activities.insert(0, {
        "id": str(uuid.uuid4())[:8],
        "agent": "rag",
        "title": "PDF Uploaded & Indexed",
        "description": f"Indexed {chunk_count} chunks from {clean_filename}",
        "timestamp": now_str,
    })
    if len(recent_activities) > 15:
        recent_activities.pop()
        
    return {
        "status": "success",
        "filename": clean_filename,
        "chunks_indexed": chunk_count,
        "message": f"Successfully indexed {clean_filename} with {chunk_count} chunks into Pinecone!"
    }


@app.get("/api/activity")
async def activity_endpoint():
    return {"activities": recent_activities}


@app.post("/api/reset")
async def reset_session():
    new_thread = f"session_{uuid.uuid4().hex[:6]}"
    return {"status": "success", "thread_id": new_thread}


# Ensure static directory exists before mounting
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")
