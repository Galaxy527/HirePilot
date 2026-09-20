/**
 * HirePilot streaming chat + collapsible sessions + expandable citations
 */
(function () {
  const root = document.getElementById("chatApp");
  if (!root) return;

  const streamUrl = root.dataset.streamUrl;
  const chatBase = (root.dataset.chatBase || "").replace(/\/$/, "");
  const messagesEl = document.getElementById("chatMessages");
  const form = document.getElementById("chatForm");
  const input = document.getElementById("chatInput");
  const sendBtn = document.getElementById("chatSend");
  const sessionList = document.getElementById("sessionList");
  const sessionToggle = document.getElementById("sessionToggle");
  const sessionBody = document.getElementById("sessionBody");
  let sessionId = root.dataset.sessionId ? Number(root.dataset.sessionId) : null;
  let busy = false;

  function csrfHeaders(extra) {
    const token = document.querySelector('meta[name="csrf-token"]');
    const h = Object.assign({ "X-Requested-With": "fetch", Accept: "application/json" }, extra || {});
    if (token) h["X-CSRFToken"] = token.content;
    return h;
  }

  const COLLAPSE_KEY = "hirepilot_chat_sidebar_collapsed";

  function applySidebar(collapsed) {
    if (!sessionList) return;
    sessionList.classList.toggle("collapsed", collapsed);
    root.classList.toggle("sidebar-collapsed", collapsed);
    if (sessionToggle) {
      sessionToggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
      sessionToggle.textContent = collapsed ? "›" : "‹";
      sessionToggle.title = collapsed ? "展开会话列表" : "收起会话列表";
    }
    try {
      localStorage.setItem(COLLAPSE_KEY, collapsed ? "1" : "0");
    } catch (_) {}
  }

  if (sessionToggle) {
    const saved = (() => {
      try {
        return localStorage.getItem(COLLAPSE_KEY) === "1";
      } catch (_) {
        return false;
      }
    })();
    applySidebar(saved);
    sessionToggle.addEventListener("click", () => {
      applySidebar(!sessionList.classList.contains("collapsed"));
    });
  }

  function scrollBottom(force) {
    if (!messagesEl) return;
    const nearBottom =
      messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 120;
    if (force || nearBottom) {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  }

  function clearEmpty() {
    const empty = messagesEl.querySelector(".empty");
    if (empty) empty.remove();
  }

  function stripCitations(text) {
    return String(text || "")
      .replace(/\n*来源[：:].*$/gim, "")
      .replace(/\n*引用来源[：:].*$/gim, "")
      .replace(/\[简历#[^\]]+\]/g, "")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
  }

  function appendBubble(role, text) {
    clearEmpty();
    const div = document.createElement("div");
    div.className = "bubble " + role;
    const body = document.createElement("div");
    body.className = "bubble-text";
    body.textContent = stripCitations(text || "");
    div.appendChild(body);
    messagesEl.appendChild(div);
    scrollBottom(true);
    return div;
  }

  function appendCitations(bubble, sources) {
    if (!sources || !sources.length) return;
    const details = document.createElement("details");
    details.className = "cite-block";
    const summary = document.createElement("summary");
    summary.textContent = "引用 " + sources.length;
    details.appendChild(summary);
    const ul = document.createElement("ul");
    ul.className = "cite-list";
    sources.forEach((s) => {
      const li = document.createElement("li");
      const src = document.createElement("div");
      src.className = "cite-src";
      src.textContent = s.source || "";
      li.appendChild(src);
      if (s.snippet) {
        const snip = document.createElement("div");
        snip.className = "cite-snip";
        snip.textContent = s.snippet;
        li.appendChild(snip);
      }
      ul.appendChild(li);
    });
    details.appendChild(ul);
    bubble.appendChild(details);
  }

  async function deleteSession(id) {
    if (!confirm("确定删除该对话？")) return;
    const res = await fetch(chatBase + "/" + id + "/delete", {
      method: "POST",
      headers: csrfHeaders(),
    });
    if (!res.ok) {
      alert("删除失败");
      return;
    }
    if (sessionId === id) {
      window.location.href = chatBase;
    } else {
      window.location.reload();
    }
  }

  document.querySelectorAll("[data-delete-session]").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      deleteSession(Number(btn.dataset.deleteSession));
    });
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (busy) return;
    const text = (input.value || "").trim();
    if (!text) return;

    busy = true;
    sendBtn.disabled = true;
    input.value = "";
    appendBubble("user", text);
    const assistantBubble = appendBubble("assistant", "");
    assistantBubble.classList.add("streaming");
    const textEl = assistantBubble.querySelector(".bubble-text");
    const cursor = document.createElement("span");
    cursor.className = "stream-cursor";
    textEl.appendChild(cursor);

    try {
      const res = await fetch(streamUrl, {
        method: "POST",
        credentials: "same-origin",
        headers: csrfHeaders({
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        }),
        body: JSON.stringify({ message: text, session_id: sessionId }),
      });
      if (!res.ok || !res.body) {
        const detail = await res.text().catch(() => "");
        throw new Error(
          "流式请求失败 (" + res.status + ")" + (detail ? ": " + detail.slice(0, 120) : "")
        );
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let full = "";
      let sources = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() || "";
        for (const block of chunks) {
          const line = block
            .split("\n")
            .map((l) => l.trim())
            .find((l) => l.startsWith("data:"));
          if (!line) continue;
          let payload;
          try {
            payload = JSON.parse(line.slice(5).trim());
          } catch (_) {
            continue;
          }
          if (payload.type === "meta" && payload.session_id) {
            sessionId = payload.session_id;
            root.dataset.sessionId = String(sessionId);
            const url = new URL(window.location.href);
            url.searchParams.set("session_id", String(sessionId));
            window.history.replaceState({}, "", url.toString());
          } else if (payload.type === "status") {
            if (!full) {
              textEl.textContent = payload.text || "生成中…";
              textEl.appendChild(cursor);
            }
          } else if (payload.type === "token") {
            full += payload.text || "";
            textEl.textContent = stripCitations(full);
            textEl.appendChild(cursor);
            scrollBottom(true);
          } else if (payload.type === "done") {
            sources = payload.sources || [];
            if (payload.session_id) sessionId = payload.session_id;
          } else if (payload.type === "error") {
            full = full || payload.message || "出错了";
            textEl.textContent = stripCitations(full);
          }
        }
      }

      cursor.remove();
      assistantBubble.classList.remove("streaming");
      textEl.textContent = stripCitations(full) || "（无回复）";
      appendCitations(assistantBubble, sources);
      scrollBottom(true);

      if (sessionBody && sessionId) {
        let link = sessionBody.querySelector('[data-session-id="' + sessionId + '"]');
        if (!link) {
          const wrap = document.createElement("div");
          wrap.className = "session-item";
          wrap.innerHTML =
            '<a data-session-id="' +
            sessionId +
            '" class="active" href="' +
            chatBase +
            "?session_id=" +
            sessionId +
            '">' +
            text.slice(0, 40) +
            '</a><button type="button" class="session-del" data-delete-session="' +
            sessionId +
            '" title="删除">×</button>';
          sessionBody.insertBefore(wrap, sessionBody.firstChild);
          wrap.querySelector("[data-delete-session]").addEventListener("click", (ev) => {
            ev.preventDefault();
            ev.stopPropagation();
            deleteSession(sessionId);
          });
        }
      }
    } catch (err) {
      cursor.remove();
      assistantBubble.classList.remove("streaming");
      const msg = String(err && err.message ? err.message : err);
      const hint =
        msg === "Failed to fetch" || /NetworkError|network/i.test(msg)
          ? "发送失败：连接中断（服务可能正在重启）。请刷新页面后重试；运维可在 /ops/logs 查看运行日志。"
          : "发送失败：" + msg;
      textEl.textContent = hint;
    } finally {
      busy = false;
      sendBtn.disabled = false;
      input.focus();
      scrollBottom(true);
    }
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  scrollBottom(true);
})();
