import { useState } from "react";
import type { Sibling, ToolCall } from "../../lib/types";
import { useAppState } from "../../lib/AppContext";
import { BranchIndicator } from "./branch-indicator";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

interface Props {
  id: string;
  role: "human" | "ai";
  content: string;
  toolCalls: ToolCall[];
  interrupted: boolean;
  streaming?: boolean;
  siblings: Sibling[];
  siblingIndex: number;
  onSwitchBranch: (humanId: string) => void;
  onEdit: (humanId: string, newContent: string) => void;
  onDelete: (humanId: string) => void;
  onCopy?: (humanId: string, content: string) => void;
}

export function MessageBubble({
  id,
  role,
  content,
  toolCalls,
  interrupted,
  streaming,
  siblings,
  siblingIndex,
  onSwitchBranch,
  onEdit,
  onDelete,
  onCopy,
}: Props) {
  const [expandedTools, setExpandedTools] = useState<Record<number, boolean>>(
    {}
  );
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState("");
  const { isStreaming } = useAppState();

  const isHuman = role === "human";

  const handleEditStart = () => {
    setEditValue(content);
    setEditing(true);
  };

  const handleEditSubmit = () => {
    const trimmed = editValue.trim();
    if (trimmed && trimmed !== content) {
      onEdit(id, trimmed);
    }
    setEditing(false);
  };

  return (
    <div className={`flex ${isHuman ? "justify-end" : "justify-start"} mb-1`}>
      <div
        className={`max-w-[80%] rounded-lg px-4 py-3 group relative ${
          isHuman ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-100"
        }`}
      >
        {/* Branch indicator (above human messages) */}
        {isHuman && siblings.length > 1 && (
          <BranchIndicator
            siblings={siblings}
            currentIndex={siblingIndex}
            onSwitch={onSwitchBranch}
          />
        )}

        {/* Action buttons (shown on hover, hidden during streaming) */}
        {isHuman && !isStreaming && !editing && (
          <div className="absolute -top-3 right-2 hidden group-hover:flex gap-1">
            <button
              onClick={handleEditStart}
              className="px-1.5 py-0.5 bg-gray-600 hover:bg-gray-500 rounded text-xs"
              title="编辑生成新分支"
            >
              ✏
            </button>
            <button
              onClick={() => onCopy?.(id, content)}
              className="px-1.5 py-0.5 bg-gray-600 hover:bg-gray-500 rounded text-xs"
              title="复制到其他会话"
            >
              📋
            </button>
            <button
              onClick={() => onDelete(id)?.catch(() => {})}
              className="px-1.5 py-0.5 bg-gray-600 hover:bg-red-600 rounded text-xs"
              title="删除"
            >
              🗑
            </button>
          </div>
        )}

        {/* Editing mode */}
        {editing ? (
          <div>
            <textarea
              className="w-full bg-blue-700 px-2 py-1 rounded text-sm resize-none outline-none"
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleEditSubmit();
                }
                if (e.key === "Escape") setEditing(false);
              }}
              autoFocus
              rows={2}
            />
            <div className="flex gap-1 mt-1 justify-end">
              <button
                onClick={() => setEditing(false)}
                className="px-2 py-0.5 text-xs bg-gray-600 rounded hover:bg-gray-500"
              >
                取消
              </button>
              <button
                onClick={handleEditSubmit}
                className="px-2 py-0.5 text-xs bg-blue-500 rounded hover:bg-blue-400"
              >
                发送
              </button>
            </div>
          </div>
        ) : (
          <>
            {/* Tool calls */}
            {toolCalls.map((tc, i) => (
              <div key={i} className="mb-2">
                <button
                  onClick={() =>
                    setExpandedTools((prev) => ({ ...prev, [i]: !prev[i] }))
                  }
                  className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-200 transition-colors"
                >
                  <span>{expandedTools[i] ? "▼" : "▶"}</span>
                  <span>🔧 {tc.name}</span>
                  {!tc.output && <span className="animate-pulse">...</span>}
                </button>
                {expandedTools[i] && (
                  <div className="mt-1 p-2 bg-gray-800 rounded text-xs text-gray-300 font-mono whitespace-pre-wrap max-h-40 overflow-y-auto">
                    <div>
                      <strong>输入:</strong> {tc.input}
                    </div>
                    {tc.output && (
                      <div className="mt-1">
                        <strong>输出:</strong> {tc.output}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}

            {/* Message content */}
            {isHuman ? (
              <div className="whitespace-pre-wrap">{content}</div>
            ) : (
              <div className="prose prose-invert prose-sm max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    code({ className, children, ...props }) {
                      const match = /language-(\w+)/.exec(className || "");
                      const inline = !match;
                      if (inline) {
                        return (
                          <code
                            className="bg-gray-600 px-1.5 py-0.5 rounded text-sm"
                            {...props}
                          >
                            {children}
                          </code>
                        );
                      }
                      return (
                        <SyntaxHighlighter
                          style={oneDark}
                          language={match[1]}
                          PreTag="div"
                          className="rounded-md !text-sm"
                        >
                          {String(children).replace(/\n$/, "")}
                        </SyntaxHighlighter>
                      );
                    },
                  }}
                >
                  {content}
                </ReactMarkdown>
              </div>
            )}

            {interrupted && (
              <div className="text-xs text-yellow-400 mt-1">(已中断)</div>
            )}
            {streaming && (
              <span className="inline-block w-2 h-4 bg-gray-300 animate-pulse ml-0.5" />
            )}
          </>
        )}
      </div>
    </div>
  );
}
