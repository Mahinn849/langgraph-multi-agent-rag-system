/**
 * static/app.js
 * =============
 * Frontend Controller for SMIT LangGraph Multi-Agent Assistant.
 * 
 * Handles:
 * - Real-time chat streaming to FastAPI /api/chat with MemorySaver thread_id
 * - Markdown rendering via marked.js
 * - Real-time live status updates for tools (RAG, GitHub, Workspace, MCP)
 * - Real-time recent activity feed
 * - Theme switcher (Minimalist White default / Sleek Dark mode)
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const chatForm = document.getElementById("chatForm");
  const messageInput = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");
  const messagesContainer = document.getElementById("messagesContainer");
  const welcomeCard = document.getElementById("welcomeCard");
  const themeToggleBtn = document.getElementById("themeToggleBtn");
  const themeIcon = document.getElementById("themeIcon");
  const toolsList = document.getElementById("toolsList");
  const activityFeed = document.getElementById("activityFeed");
  
  // Modal Elements
  const infoModal = document.getElementById("infoModal");
  const modalTitle = document.getElementById("modalTitle");
  const modalBody = document.getElementById("modalBody");
  const modalCloseBtn = document.getElementById("modalCloseBtn");

  // Session Thread Management (MemorySaver support)
  let threadId = localStorage.getItem("smit_agent_thread_id");
  if (!threadId) {
    threadId = "session_" + Math.random().toString(36).substring(2, 9);
    localStorage.setItem("smit_agent_thread_id", threadId);
  }

  // Auto-resize input textarea
  messageInput.addEventListener("input", () => {
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + "px";
  });

  // Handle Enter key (Shift+Enter for newline)
  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  // Theme Toggle Logic
  const savedTheme = localStorage.getItem("smit_agent_theme") || "light";
  if (savedTheme === "dark") {
    document.body.classList.add("dark-theme");
    document.body.classList.remove("light-theme");
    themeIcon.textContent = "☀️";
  } else {
    document.body.classList.add("light-theme");
    document.body.classList.remove("dark-theme");
    themeIcon.textContent = "🌙";
  }

  themeToggleBtn.addEventListener("click", () => {
    if (document.body.classList.contains("dark-theme")) {
      document.body.classList.remove("dark-theme");
      document.body.classList.add("light-theme");
      themeIcon.textContent = "🌙";
      localStorage.setItem("smit_agent_theme", "light");
    } else {
      document.body.classList.add("dark-theme");
      document.body.classList.remove("light-theme");
      themeIcon.textContent = "☀️";
      localStorage.setItem("smit_agent_theme", "dark");
    }
  });

  // Modal Controls
  modalCloseBtn.addEventListener("click", () => {
    infoModal.classList.add("hidden");
  });
  infoModal.addEventListener("click", (e) => {
    if (e.target === infoModal) infoModal.classList.add("hidden");
  });

  // Sidebar Nav Item Clicks
  document.querySelectorAll(".sidebar-nav .nav-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".sidebar-nav .nav-item").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      const tab = btn.dataset.tab;
      const prompt = btn.dataset.prompt;

      if (prompt) {
        window.fillPrompt(prompt);
      } else if (tab === "tools") {
        openToolsModal();
      } else if (tab === "settings") {
        openSettingsModal();
      }
    });
  });

  // ============================================================================
  // PDF Upload & Dynamic Indexing Handling
  // ============================================================================
  const btnAttach = document.getElementById("btnAttach");
  const pdfFileInput = document.getElementById("pdfFileInput");

  if (btnAttach && pdfFileInput) {
    btnAttach.addEventListener("click", () => {
      pdfFileInput.click();
    });

    pdfFileInput.addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      if (!file.name.toLowerCase().endsWith(".pdf")) {
        alert("Please select a valid PDF file (.pdf)");
        return;
      }

      if (welcomeCard) {
        welcomeCard.style.display = "none";
      }

      const nowTime = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      appendUserMessage(`📎 Uploaded document: ${file.name}`, nowTime);

      const thinkingEl = appendThinkingMessage();
      const thinkingText = thinkingEl.querySelector(".thinking-card span");
      if (thinkingText) {
        thinkingText.textContent = `Extracting, cleaning & indexing ${file.name} to Pinecone...`;
      }
      scrollToBottom();

      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await fetch("/api/upload_pdf", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || "Upload failed");
        }

        const data = await response.json();
        thinkingEl.remove();

        appendAssistantMessage({
          route: "rag",
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          final_answer: `✅ **Document Successfully Indexed!**\n\n• **File Name**: \`${data.filename}\`\n• **Chunks Indexed in Pinecone**: \`${data.chunks_indexed}\`\n\nYou can now ask any question about this document or the Agentic AI roadmap!`,
          route_reasoning: `Uploaded new document to Pinecone serverless vector store.`,
          metadata: { chunks: data.chunks_indexed },
        });

        fetchToolsStatus();
        fetchActivityFeed();
      } catch (err) {
        thinkingEl.remove();
        appendAssistantMessage({
          route: "general",
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          final_answer: `❌ **Failed to upload/index PDF**: ${err.message}`,
          route_reasoning: "PDF upload error",
          metadata: {},
        });
      } finally {
        pdfFileInput.value = "";
        scrollToBottom();
      }
    });
  }

  // ============================================================================
  // Chat Message Sending & Rendering
  // ============================================================================
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = messageInput.value.trim();
    if (!query) return;

    // Reset input
    messageInput.value = "";
    messageInput.style.height = "auto";
    sendBtn.disabled = true;

    // Hide welcome hero on first message
    if (welcomeCard) {
      welcomeCard.style.display = "none";
    }

    // 1. Render User Message
    const userTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    appendUserMessage(query, userTime);
    scrollToBottom();

    // 2. Render Animated "Thinking & Routing..." Card
    const thinkingElement = appendThinkingMessage();
    scrollToBottom();

    try {
      // 3. Send to FastAPI backend
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          thread_id: threadId,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Server error while processing query.");
      }

      const data = await response.json();

      // Remove thinking element
      thinkingElement.remove();

      // 4. Render Assistant Message with Sub-Agent Badge
      appendAssistantMessage(data);
      scrollToBottom();

      // Refresh activity and status
      fetchLiveStatus();
      fetchRecentActivity();

    } catch (err) {
      thinkingElement.remove();
      appendAssistantMessage({
        route: "general",
        final_answer: `❌ **Error:** ${err.message}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      });
      scrollToBottom();
    } finally {
      sendBtn.disabled = false;
      messageInput.focus();
    }
  });

  function appendUserMessage(text, time) {
    const row = document.createElement("div");
    row.className = "message-row user";
    row.innerHTML = `
      <div class="msg-bubble-wrap">
        <div class="msg-bubble">${escapeHtml(text)}</div>
        <span class="msg-time">${time}</span>
      </div>
      <div class="msg-avatar user-av">FH</div>
    `;
    messagesContainer.appendChild(row);
  }

  function appendThinkingMessage() {
    const row = document.createElement("div");
    row.className = "message-row assistant thinking-row";
    row.innerHTML = `
      <div class="msg-avatar ai-av">✦</div>
      <div class="msg-bubble-wrap">
        <div class="thinking-card">
          <div class="loading-dots">
            <span></span><span></span><span></span>
          </div>
          <span>Thinking, routing, & executing tools...</span>
        </div>
      </div>
    `;
    messagesContainer.appendChild(row);
    return row;
  }

  function appendAssistantMessage(data) {
    const route = data.route || "general";
    const time = data.timestamp || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    // Choose appropriate badge
    let badgeText = "💬 General Assistant";
    let badgeClass = "badge-general";
    
    if (route === "rag") {
      badgeText = "📖 RAG Knowledge Agent";
      badgeClass = "badge-rag";
    } else if (route === "github") {
      badgeText = "🐙 GitHub MCP Agent";
      badgeClass = "badge-github";
    } else if (route === "workspace") {
      badgeText = "✉️ Google Workspace (Calendar & Gmail)";
      badgeClass = "badge-workspace";
    }

    // Parse Markdown safely
    let formattedHtml = "";
    if (window.marked) {
      formattedHtml = marked.parse(data.final_answer || "");
    } else {
      formattedHtml = escapeHtml(data.final_answer || "").replace(/\n/g, "<br>");
    }

    const row = document.createElement("div");
    row.className = "message-row assistant";
    row.innerHTML = `
      <div class="msg-avatar ai-av">✦</div>
      <div class="msg-bubble-wrap">
        <div class="msg-meta-bar">
          <span class="agent-badge ${badgeClass}">${badgeText}</span>
        </div>
        <div class="msg-bubble">${formattedHtml}</div>
        <span class="msg-time">${time}</span>
      </div>
    `;
    messagesContainer.appendChild(row);
  }

  function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function escapeHtml(string) {
    const div = document.createElement("div");
    div.innerText = string;
    return div.innerHTML;
  }

  // ============================================================================
  // Global Helper: Fill prompt from chips
  // ============================================================================
  window.fillPrompt = function (text) {
    messageInput.value = text;
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + "px";
    messageInput.focus();
  };

  // ============================================================================
  // Live Status & Activity Polling
  // ============================================================================
  async function fetchLiveStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) return;
      const data = await res.json();

      updateToolStatus("status-rag", data.rag);
      updateToolStatus("status-github", data.github);
      updateToolStatus("status-mcp", data.mcp);
      updateToolStatus("status-gmail", data.gmail);
      updateToolStatus("status-calendar", data.calendar);
    } catch (e) {
      console.warn("Could not fetch status:", e);
    }
  }

  function updateToolStatus(elementId, toolData) {
    if (!toolData) return;
    const card = document.getElementById(elementId);
    if (!card) return;
    
    const badge = card.querySelector(".status-badge");
    const detail = card.querySelector(".tool-detail");
    
    if (badge) {
      badge.textContent = toolData.status;
      badge.className = `status-badge ${toolData.status.toLowerCase()}`;
    }
    if (detail && toolData.details) {
      detail.textContent = toolData.details;
    }
  }

  async function fetchRecentActivity() {
    try {
      const res = await fetch("/api/activity");
      if (!res.ok) return;
      const data = await res.json();
      const list = data.activities || [];

      if (list.length === 0) return;

      activityFeed.innerHTML = "";
      list.slice(0, 5).forEach((item) => {
        const itemEl = document.createElement("div");
        itemEl.className = "activity-item";

        let icon = "⚡";
        let iconClass = "workspace";
        if (item.agent === "github") { icon = "🐙"; iconClass = "github"; }
        else if (item.agent === "rag") { icon = "📖"; iconClass = "rag"; }
        else if (item.agent === "workspace") { icon = "✉️"; iconClass = "email"; }

        itemEl.innerHTML = `
          <div class="activity-icon-wrap ${iconClass}">${icon}</div>
          <div class="activity-content">
            <div class="activity-title">${escapeHtml(item.title)}</div>
            <div class="activity-desc">${escapeHtml(item.description)}</div>
            <div class="activity-time">${escapeHtml(item.timestamp)}</div>
          </div>
        `;
        activityFeed.appendChild(itemEl);
      });
    } catch (e) {
      console.warn("Could not fetch activities:", e);
    }
  }

  // Modals Info
  function openToolsModal() {
    modalTitle.textContent = "Integrated Sub-Agents & Tools";
    modalBody.innerHTML = `
      <ul style="list-style: none; display: flex; flex-direction: column; gap: 14px; padding: 0;">
        <li>
          <strong>📖 RAG Sub-Agent (Hosted Pinecone):</strong><br>
          <span style="font-size: 0.85rem; color: var(--text-muted);">Indexes 'Agentic AI Roadmap' PDF into Pinecone Serverless vector database. Provides grounded Q&A with strict zero-hallucination guardrail.</span>
        </li>
        <li>
          <strong>🐙 GitHub MCP Sub-Agent:</strong><br>
          <span style="font-size: 0.85rem; color: var(--text-muted);">Executes live via Model Context Protocol stdio server. Capabilities: inspect repositories, search code, list commits, view issues, and read files.</span>
        </li>
        <li>
          <strong>📅 Google Calendar MCP (Composio):</strong><br>
          <span style="font-size: 0.85rem; color: var(--text-muted);">Lists user calendars, asks for confirmation before booking, schedules appointments, and cancels meetings.</span>
        </li>
        <li>
          <strong>✉️ Gmail MCP (Composio):</strong><br>
          <span style="font-size: 0.85rem; color: var(--text-muted);">Sends direct emails, writes drafts in Gmail, and automatically dispatches calendar invites to attendees.</span>
        </li>
      </ul>
    `;
    infoModal.classList.remove("hidden");
  }

  function openSettingsModal() {
    modalTitle.textContent = "Session & System Settings";
    modalBody.innerHTML = `
      <p style="margin-bottom: 12px;"><strong>Current Session Thread:</strong> <code style="padding: 2px 6px; background: var(--bg-tertiary); border-radius: 4px;">${threadId}</code></p>
      <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 16px;">
        LangGraph <code>MemorySaver</code> maintains the state of this conversation under this thread.
      </p>
      <button id="btnResetSession" style="padding: 8px 14px; background: #ef4444; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-weight: 600;">
        Reset Chat Session
      </button>
    `;
    infoModal.classList.remove("hidden");

    document.getElementById("btnResetSession").addEventListener("click", async () => {
      try {
        const res = await fetch("/api/reset", { method: "POST" });
        const data = await res.json();
        threadId = data.thread_id;
        localStorage.setItem("smit_agent_thread_id", threadId);
        messagesContainer.innerHTML = "";
        infoModal.classList.add("hidden");
        alert("Session reset! New clean chat session started.");
      } catch (e) {
        alert("Could not reset session: " + e.message);
      }
    });
  }

  // Initial Fetches
  fetchLiveStatus();
  fetchRecentActivity();

  // Poll status every 15 seconds
  setInterval(() => {
    fetchLiveStatus();
    fetchRecentActivity();
  }, 15000);
});
