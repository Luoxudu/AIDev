import { useState, useEffect, useRef, useCallback } from "react";
import {
  uploadDocument,
  triggerRebuild,
  fetchBuildStatus,
  listDocuments,
  deleteDocument,
  type BuildStatus,
} from "../../lib/api";

export function DocumentPanel() {
  const [status, setStatus] = useState<BuildStatus>({
    status: "idle",
    error: null,
  });
  const [files, setFiles] = useState<string[]>([]);
  const [uploadMsg, setUploadMsg] = useState("");
  const [uploadError, setUploadError] = useState(false);
  const [newFiles, setNewFiles] = useState<Set<string>>(new Set());
  const fileRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined);

  const refreshFiles = useCallback(async () => {
    try {
      const data = await listDocuments();
      setFiles(data.files);
    } catch {
      // ignore
    }
  }, []);

  const refreshStatus = useCallback(async () => {
    try {
      const s = await fetchBuildStatus();
      setStatus(s);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    const init = async () => {
      try {
        const data = await listDocuments();
        if (!cancelled) setFiles(data.files);
      } catch { /* ignore */ }
      try {
        const s = await fetchBuildStatus();
        if (!cancelled) setStatus(s);
      } catch { /* ignore */ }
    };
    init();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (status.status === "building") {
      pollRef.current = setInterval(refreshStatus, 1500);
    } else if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = undefined;
      if (status.status === "done" || status.status === "idle") {
        queueMicrotask(() => setNewFiles(new Set()));
      }
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [status.status, refreshStatus]);

  const showMsg = (text: string, isError = false) => {
    setUploadMsg(text);
    setUploadError(isError);
    setTimeout(() => setUploadMsg(""), 4000);
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const res = await uploadDocument(file);
      if (res.ok) {
        setNewFiles((prev) => new Set(prev).add(res.filename));
        await refreshFiles();
        showMsg("已上传，需手动重建索引");
      } else {
        showMsg(res.message || "上传失败", true);
      }
    } catch {
      showMsg("上传失败", true);
    }
    if (fileRef.current) fileRef.current.value = "";
  };

  const handleRebuild = async () => {
    try {
      setStatus({ status: "building", error: null });
      await triggerRebuild();
      await refreshStatus();
    } catch {
      setStatus({ status: "idle", error: null });
      showMsg("重建索引失败", true);
    }
  };

  const handleDelete = async (filename: string) => {
    try {
      await deleteDocument(filename);
      setNewFiles((prev) => {
        const next = new Set(prev);
        next.delete(filename);
        return next;
      });
      await refreshFiles();
    } catch {
      showMsg("删除文件失败", true);
    }
  };

  const statusConfig: Record<string, { label: string; color: string }> = {
    idle: { label: "已同步", color: "text-green-400" },
    done: { label: "构建完成", color: "text-green-400" },
    pending: { label: "待更新", color: "text-yellow-400" },
    building: { label: "构建中...", color: "text-yellow-400" },
    error: { label: "构建失败", color: "text-red-400" },
  };

  const st = statusConfig[status.status] || statusConfig.idle;

  return (
    <div className="flex flex-col gap-2">
      <button
        onClick={() => fileRef.current?.click()}
        disabled={status.status === "building"}
        className="w-full py-1.5 px-3 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 rounded text-sm transition-colors"
      >
        📁 上传文档
      </button>
      <input
        ref={fileRef}
        type="file"
        accept=".md,.txt,.pdf,.docx"
        className="hidden"
        onChange={handleUpload}
      />

      <button
        onClick={handleRebuild}
        disabled={status.status === "building"}
        className="w-full py-1.5 px-3 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 rounded text-sm transition-colors"
      >
        🔄 重建索引
      </button>

      <div className="text-xs text-gray-400">
        索引状态：<span className={st.color}>{st.label}</span>
        {status.error && (
          <span className="block text-red-400 mt-1 truncate" title={status.error}>
            {status.error}
          </span>
        )}
      </div>

      {uploadMsg && (
        <div className={`text-xs ${uploadError ? "text-red-400" : "text-blue-400"}`}>
          {uploadMsg}
        </div>
      )}

      {files.length > 0 && (
        <div className="text-xs max-h-24 overflow-y-auto flex flex-col gap-0.5">
          {files.map((f) => {
            const isNew = newFiles.has(f);
            return (
              <div key={f} className="flex items-center gap-1 group">
                <span
                  className={`truncate flex-1 ${isNew ? "text-green-400" : "text-gray-500"}`}
                  title={f}
                >
                  {f}
                </span>
                <button
                  onClick={() => handleDelete(f)}
                  className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-400 transition-opacity flex-shrink-0"
                  title="删除"
                >
                  ✕
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
