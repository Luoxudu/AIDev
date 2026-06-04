import type { Session, SSEEvent, TreeData } from "./types";

export async function fetchSessions(): Promise<Session[]> {
  const res = await fetch("/api/sessions");
  return res.json();
}

export async function createSession(): Promise<Session> {
  const res = await fetch("/api/sessions", { method: "POST" });
  return res.json();
}

export async function deleteSession(id: string): Promise<void> {
  await fetch(`/api/sessions/${id}`, { method: "DELETE" });
}

export async function renameSession(
  id: string,
  title: string
): Promise<void> {
  await fetch(`/api/sessions/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
}

export async function fetchTree(sessionId: string): Promise<TreeData> {
  const res = await fetch(`/api/sessions/${sessionId}/tree`);
  return res.json();
}

export async function deleteMessage(humanMsgId: string): Promise<void> {
  const res = await fetch(`/api/messages/${humanMsgId}`, { method: "DELETE" });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail?.message || body.detail || `删除失败 (${res.status})`);
  }
}

export async function fetchBranchMessages(
  sessionId: string,
  leafAiId: string | null
) {
  const params = leafAiId ? `?leaf_ai_id=${leafAiId}` : "";
  const res = await fetch(`/api/sessions/${sessionId}/messages${params}`);
  return res.json();
}

export async function fetchSystemPrompt(): Promise<string> {
  const res = await fetch("/api/system-prompt");
  const data = await res.json();
  return data.content;
}

export async function updateSystemPrompt(content: string): Promise<void> {
  await fetch("/api/system-prompt", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
}

export interface BuildStatus {
  status: "idle" | "building" | "done" | "error" | "pending";
  error: string | null;
}

export async function fetchBuildStatus(): Promise<BuildStatus> {
  const res = await fetch("/api/documents/status");
  if (!res.ok) throw new Error(`获取状态失败 (${res.status})`);
  return res.json();
}

export async function triggerRebuild(): Promise<void> {
  const res = await fetch("/api/documents/rebuild", { method: "POST" });
  if (!res.ok) throw new Error(`重建失败 (${res.status})`);
}

export async function uploadDocument(file: File): Promise<{ ok: boolean; message?: string; filename: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/documents/upload", { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.message || `上传失败 (${res.status})`);
  }
  return res.json();
}

export async function listDocuments(): Promise<{ files: string[] }> {
  const res = await fetch("/api/documents");
  if (!res.ok) throw new Error(`获取文件列表失败 (${res.status})`);
  return res.json();
}

export async function deleteDocument(filename: string): Promise<void> {
  const res = await fetch(`/api/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.message || `删除失败 (${res.status})`);
  }
}

export async function copyMessage(
  sourceHumanId: string,
  targetHumanId: string,
  targetSessionId: string,
  mode: "above" | "below" | "replace"
): Promise<{ ok: boolean; target_session_id: string; new_human_id: string }> {
  const res = await fetch("/api/messages/copy", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      source_human_id: sourceHumanId,
      target_human_id: targetHumanId,
      target_session_id: targetSessionId,
      mode,
    }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const msg = typeof body.detail === "string"
      ? body.detail
      : body.detail?.message || `复制失败 (${res.status})`;
    throw new Error(msg);
  }
  return res.json();
}

export function streamChat(
  sessionId: string,
  query: string,
  parentMessageId: string | null,
  onEvent: (event: SSEEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  return new Promise((resolve, reject) => {
    fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        query,
        parent_message_id: parentMessageId,
      }),
      signal,
    })
      .then((res) => {
        if (!res.ok) {
          return res
            .json()
            .then((d) => {
              onEvent({
                type: "error",
                message: d.detail?.message || d.detail || "请求失败",
              });
              resolve();
            })
            .catch(() => {
              onEvent({ type: "error", message: `请求失败 (${res.status})` });
              resolve();
            });
        }
        const reader = res.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let currentEvent = "";

        function processChunk(chunk: string) {
          buffer += chunk;
          const lines = buffer.split("\n");
          buffer = lines.pop()!;

          for (const line of lines) {
            if (line.startsWith("event: ")) {
              currentEvent = line.slice(7).trim();
            } else if (line.startsWith("data: ")) {
              const raw = line.slice(6);
              try {
                const data = JSON.parse(raw);
                if (currentEvent === "token") {
                  onEvent({ type: "token", text: data.text });
                } else if (currentEvent === "tool_start") {
                  onEvent({
                    type: "tool_start",
                    name: data.name,
                    input: data.input,
                  });
                } else if (currentEvent === "tool_end") {
                  onEvent({
                    type: "tool_end",
                    name: data.name,
                    output: data.output,
                  });
                } else if (currentEvent === "done") {
                  onEvent({ type: "done" });
                } else if (currentEvent === "error") {
                  onEvent({ type: "error", message: data.message });
                }
              } catch {
                // ignore parse errors
              }
              currentEvent = "";
            }
          }
        }

        function read() {
          reader
            .read()
            .then(({ done, value }) => {
              if (done) {
                resolve();
                return;
              }
              processChunk(decoder.decode(value, { stream: true }));
              read();
            })
            .catch((err) => {
              if (err.name === "AbortError") {
                resolve();
              } else {
                reject(err);
              }
            });
        }
        read();
      })
      .catch((err) => {
        if (err.name === "AbortError") {
          resolve();
        } else {
          reject(err);
        }
      });
  });
}
