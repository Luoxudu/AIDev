import { useState } from "react";
import { useAppState } from "../../lib/AppContext";
import { renameSession } from "../../lib/api";

export function SessionList() {
  const {
    sessions,
    currentSession,
    loadSessions,
    selectSession,
    createNewSession,
    removeSession,
  } = useAppState();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");

  const handleRename = async (id: string) => {
    if (editTitle.trim()) {
      await renameSession(id, editTitle.trim());
      await loadSessions();
    }
    setEditingId(null);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-3 border-b border-gray-700">
        <button
          onClick={createNewSession}
          className="w-full py-2 px-3 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium transition-colors"
        >
          + 新建会话
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {sessions.map((s) => (
          <div
            key={s.id}
            className={`group flex items-center gap-1 px-3 py-2 cursor-pointer hover:bg-gray-700/50 text-sm ${
              currentSession?.id === s.id ? "bg-gray-700" : ""
            }`}
            onClick={() => editingId !== s.id && selectSession(s)}
          >
            {editingId === s.id ? (
              <input
                className="flex-1 bg-gray-600 px-2 py-0.5 rounded text-sm outline-none"
                value={editTitle}
                onChange={(e) => setEditTitle(e.target.value)}
                onBlur={() => handleRename(s.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleRename(s.id);
                  if (e.key === "Escape") setEditingId(null);
                }}
                autoFocus
              />
            ) : (
              <>
                <span className="flex-1 truncate">{s.title}</span>
                <div className="hidden group-hover:flex gap-1">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setEditingId(s.id);
                      setEditTitle(s.title);
                    }}
                    className="p-0.5 hover:text-blue-400"
                    title="重命名"
                  >
                    ✏
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      removeSession(s.id);
                    }}
                    className="p-0.5 hover:text-red-400"
                    title="删除"
                  >
                    ✕
                  </button>
                </div>
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
