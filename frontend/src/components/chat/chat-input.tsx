import { useState, useRef, useEffect } from "react";
import { useAppState } from "../../lib/AppContext";

export function ChatInput() {
  const { isStreaming, sendMessage, stopGeneration, currentSession } =
    useAppState();
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 150) + "px";
    }
  }, [input]);

  if (!currentSession) return null;

  const handleSend = () => {
    const query = input.trim();
    if (!query || isStreaming) return;
    setInput("");
    sendMessage(query);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t border-gray-700 p-4">
      <div className="flex gap-2 items-end max-w-4xl mx-auto">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入消息..."
          disabled={isStreaming}
          rows={1}
          className="flex-1 bg-gray-700 text-white rounded-lg px-4 py-2.5 resize-none outline-none focus:ring-2 focus:ring-blue-500 placeholder-gray-400 disabled:opacity-50"
        />
        {isStreaming ? (
          <button
            onClick={stopGeneration}
            className="px-4 py-2.5 bg-red-600 hover:bg-red-700 rounded-lg text-sm font-medium transition-colors whitespace-nowrap"
          >
            ⏹ 停止
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={!input.trim()}
            className="px-4 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg text-sm font-medium transition-colors whitespace-nowrap"
          >
            发送
          </button>
        )}
      </div>
    </div>
  );
}
