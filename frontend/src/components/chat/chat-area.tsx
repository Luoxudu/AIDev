import { useEffect, useRef, useState } from "react";
import { useAppState } from "../../lib/AppContext";
import { MessageBubble } from "./message-bubble";
import { CopyNodeModal } from "./copy-node-modal";

export function ChatArea() {
  const {
    messages,
    currentSession,
    sendMessage,
    switchBranch,
    deleteNode,
    selectSession,
    loadSessions,
  } = useAppState();
  const bottomRef = useRef<HTMLDivElement>(null);

  const [copySource, setCopySource] = useState<{
    humanId: string;
    content: string;
  } | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleEdit = (humanId: string, newContent: string) => {
    const idx = messages.findIndex((m) => m.id === humanId);
    if (idx === -1) return;
    const prevAi =
      idx > 0 && messages[idx - 1].role === "ai"
        ? messages[idx - 1].id
        : null;
    sendMessage(newContent, prevAi, idx);
  };

  const handleCopied = async (targetSessionId: string, newHumanId: string) => {
    await loadSessions();
    const { fetchSessions } = await import("../../lib/api");
    const all = await fetchSessions();
    const target = all.find((s) => s.id === targetSessionId);
    if (target) {
      await selectSession(target, newHumanId);
    }
  };

  if (!currentSession) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-500">
        <div className="text-center">
          <div className="text-4xl mb-4">💬</div>
          <p className="text-lg">选择或新建一个会话开始对话</p>
        </div>
      </div>
    );
  }

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-500">
        <div className="text-center">
          <p className="text-lg">发送消息开始对话</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6">
      {messages.map((msg) => (
        <MessageBubble
          key={msg.id}
          id={msg.id}
          role={msg.role}
          content={msg.content}
          toolCalls={msg.toolCalls}
          interrupted={msg.interrupted}
          streaming={msg.streaming}
          siblings={msg.siblings}
          siblingIndex={msg.siblingIndex}
          onSwitchBranch={(humanId) => switchBranch(humanId, null)}
          onEdit={handleEdit}
          onDelete={deleteNode}
          onCopy={
            msg.role === "human"
              ? (humanId, content) =>
                  setCopySource({ humanId, content })
              : undefined
          }
        />
      ))}
      <div ref={bottomRef} />

      {copySource && currentSession && (
        <CopyNodeModal
          sourceHumanId={copySource.humanId}
          sourceContent={copySource.content}
          currentSessionId={currentSession.id}
          onClose={() => setCopySource(null)}
          onCopied={handleCopied}
        />
      )}
    </div>
  );
}
