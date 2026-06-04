import { useState, useEffect, useCallback } from "react";
import { fetchSystemPrompt, updateSystemPrompt } from "../../lib/api";

export function SystemPromptEditor() {
  const [content, setContent] = useState("");
  const [saved, setSaved] = useState("");
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");
  const [expanded, setExpanded] = useState(false);

  const load = useCallback(async () => {
    try {
      const text = await fetchSystemPrompt();
      setContent(text);
      setSaved(text);
      setDirty(false);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleSave = async () => {
    setSaving(true);
    try {
      await updateSystemPrompt(content);
      setSaved(content);
      setDirty(false);
      setMsg("已保存并生效");
    } catch {
      setMsg("保存失败");
    } finally {
      setSaving(false);
    }
    setTimeout(() => setMsg(""), 3000);
  };

  const handleReset = () => {
    setContent(saved);
    setDirty(false);
  };

  return (
    <div>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left py-1 text-xs text-gray-400 hover:text-gray-200 flex items-center justify-between"
      >
        <span>⚙️ 系统提示词</span>
        <span className="text-[10px]">{expanded ? "▼" : "▶"}</span>
      </button>

      {expanded && (
        <div className="mt-1 flex flex-col gap-1.5">
          <textarea
            className="w-full h-28 bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs resize-none outline-none focus:border-blue-500"
            placeholder="输入自定义系统提示词..."
            value={content}
            onChange={(e) => {
              setContent(e.target.value);
              setDirty(e.target.value !== saved);
            }}
          />
          <div className="flex gap-1.5">
            <button
              onClick={handleSave}
              disabled={saving || !dirty}
              className="flex-1 py-1 px-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 rounded text-xs transition-colors"
            >
              {saving ? "保存中..." : "保存"}
            </button>
            {dirty && (
              <button
                onClick={handleReset}
                className="py-1 px-2 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
              >
                撤销
              </button>
            )}
          </div>
          {msg && <div className="text-xs text-green-400">{msg}</div>}
        </div>
      )}
    </div>
  );
}
