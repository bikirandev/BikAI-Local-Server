import { useState, useEffect, useRef } from "react";
import { fetchStatus } from "../api";
import { Send, Trash2 } from "lucide-react";

interface Message {
  role: "user" | "assistant";
  content: string;
}

function getAiBase(port: number): string {
  const { protocol, hostname, port: locPort } = window.location;
  if (!locPort || locPort === "80" || locPort === "443") return "/v1";
  return `${protocol}//${hostname}:${port}/v1`;
}

export default function Playground() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [aiBase, setAiBase] = useState<string | null>(null);
  const [serverRunning, setServerRunning] = useState(true);
  const [error, setError] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    fetchStatus()
      .then((s) => {
        setAiBase(getAiBase(s.port));
        setServerRunning(s.running);
      })
      .catch(() => {
        setAiBase(getAiBase(8000));
      });
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send() {
    const text = input.trim();
    if (!text || streaming || !aiBase) return;

    setInput("");
    setError("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    const history: Message[] = [...messages, { role: "user", content: text }];
    setMessages([...history, { role: "assistant", content: "" }]);
    setStreaming(true);

    try {
      const key = localStorage.getItem("bikai_api_key") || "";
      const resp = await fetch(`${aiBase}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": key,
        },
        body: JSON.stringify({
          messages: history,
          stream: true,
          temperature: 0.7,
          max_tokens: 2048,
        }),
      });

      if (!resp.ok) {
        const err = (await resp.json().catch(() => ({}))) as {
          detail?: string;
        };
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }

      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let accumulated = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop()!;
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6).trim();
          if (data === "[DONE]") break;
          try {
            const chunk = JSON.parse(data) as {
              choices?: { delta?: { content?: string } }[];
            };
            const token = chunk.choices?.[0]?.delta?.content || "";
            accumulated += token;
            setMessages((m) => {
              const copy = [...m];
              copy[copy.length - 1] = {
                role: "assistant",
                content: accumulated,
              };
              return copy;
            });
          } catch {
            /* skip malformed */
          }
        }
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      setMessages((m) => m.slice(0, -1));
    } finally {
      setStreaming(false);
      setTimeout(() => textareaRef.current?.focus(), 50);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  function clear() {
    setMessages([]);
    setError("");
    textareaRef.current?.focus();
  }

  if (!serverRunning && aiBase !== null) {
    return (
      <div className="card" style={{ textAlign: "center", padding: 48 }}>
        <div style={{ fontSize: 36, marginBottom: 12 }}>⚠️</div>
        <div style={{ fontWeight: 600, fontSize: 16, marginBottom: 8 }}>
          AI Server is not running
        </div>
        <div className="text-muted" style={{ fontSize: 13 }}>
          Go to Dashboard and start the server to use the playground.
        </div>
      </div>
    );
  }

  return (
    <div className="playground">
      <div className="pg-messages">
        {messages.length === 0 && (
          <div className="pg-empty">
            <div className="pg-empty-icon">💬</div>
            <div className="pg-empty-title">Chat with your local AI</div>
            <div className="pg-empty-sub">
              Your messages stay private — everything runs on your server
            </div>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`pg-row ${msg.role}`}>
            <div className="pg-label">{msg.role === "user" ? "You" : "AI"}</div>
            <div
              className={`pg-bubble${streaming && i === messages.length - 1 && msg.role === "assistant" ? " pg-streaming" : ""}`}
            >
              {msg.content ? (
                <span style={{ whiteSpace: "pre-wrap" }}>{msg.content}</span>
              ) : streaming && i === messages.length - 1 ? (
                <span className="pg-cursor" />
              ) : null}
            </div>
          </div>
        ))}
        {error && (
          <div className="alert alert-error" style={{ margin: "4px 0" }}>
            {error}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="pg-inputbar">
        <textarea
          ref={textareaRef}
          className="form-input"
          style={{
            flex: 1,
            resize: "none",
            fontFamily: "var(--font)",
            fontSize: 14,
            minHeight: 44,
            maxHeight: 160,
            lineHeight: 1.5,
            padding: "10px 12px",
          }}
          rows={1}
          placeholder="Message your AI… (Enter to send, Shift+Enter for newline)"
          value={input}
          onChange={(e) => {
            setInput(e.target.value);
            e.target.style.height = "auto";
            e.target.style.height = Math.min(e.target.scrollHeight, 160) + "px";
          }}
          onKeyDown={handleKeyDown}
          disabled={streaming}
          autoFocus
        />
        <div className="pg-actions">
          <button
            className="btn btn-ghost btn-sm"
            onClick={clear}
            disabled={messages.length === 0}
            title="Clear conversation"
          >
            <Trash2 size={14} />
          </button>
          <button
            className="btn btn-primary"
            onClick={send}
            disabled={!input.trim() || streaming || !aiBase}
          >
            {streaming ? (
              <>
                <span className="spin">⟳</span> Generating
              </>
            ) : (
              <>
                <Send size={13} /> Send
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
