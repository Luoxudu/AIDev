import { useState, useEffect, useCallback, useRef } from "react";
import type { Session, TreeNode } from "../../lib/types";
import { fetchSessions, fetchTree, copyMessage } from "../../lib/api";

interface Props {
  sourceHumanId: string;
  sourceContent: string;
  currentSessionId: string;
  onClose: () => void;
  onCopied: (targetSessionId: string, newHumanId: string) => Promise<void>;
}

function MiniTree({
  nodes,
  selectedId,
  onSelect,
}: {
  nodes: TreeNode[];
  selectedId: string | null;
  onSelect: (humanId: string) => void;
}) {
  return (
    <div className="max-h-60 overflow-y-auto text-sm font-mono space-y-0.5">
      {nodes.map((node) => (
        <MiniTreeNode
          key={node.human.id}
          node={node}
          depth={0}
          selectedId={selectedId}
          onSelect={onSelect}
        />
      ))}
    </div>
  );
}

function MiniTreeNode({
  node,
  depth,
  selectedId,
  onSelect,
}: {
  node: TreeNode;
  depth: number;
  selectedId: string | null;
  onSelect: (humanId: string) => void;
}) {
  const isSelected = node.human.id === selectedId;
  return (
    <>
      <div
        className={`cursor-pointer px-2 py-1 rounded hover:bg-gray-600 truncate ${
          isSelected ? "bg-blue-700 text-white" : "text-gray-300"
        }`}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={() => onSelect(node.human.id)}
        title={node.human.content}
      >
        <span className="text-gray-500 mr-1">Q:</span>
        {node.human.content.slice(0, 40)}
      </div>
      {node.children.map((child) => (
        <MiniTreeNode
          key={child.human.id}
          node={child}
          depth={depth + 1}
          selectedId={selectedId}
          onSelect={onSelect}
        />
      ))}
    </>
  );
}

function findNode(nodes: TreeNode[], targetId: string): TreeNode | null {
  for (const node of nodes) {
    if (node.human.id === targetId) return node;
    const found = findNode(node.children, targetId);
    if (found) return found;
  }
  return null;
}

function findParentOf(
  nodes: TreeNode[],
  targetId: string
): TreeNode | null {
  for (const node of nodes) {
    for (const child of node.children) {
      if (child.human.id === targetId) return node;
    }
    const found = findParentOf(node.children, targetId);
    if (found) return found;
  }
  return null;
}

export function CopyNodeModal({
  sourceHumanId,
  sourceContent,
  currentSessionId,
  onClose,
  onCopied,
}: Props) {
  const [step, setStep] = useState<"pick-session" | "pick-node">("pick-session");
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedSession, setSelectedSession] = useState<Session | null>(null);
  const [treeNodes, setTreeNodes] = useState<TreeNode[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSeqRef = useRef(0);

  useEffect(() => {
    fetchSessions()
      .then((all) => {
        setSessions(all.filter((s) => s.id !== currentSessionId));
      })
      .catch(() => {
        setError("加载会话列表失败");
      });
  }, [currentSessionId]);

  const handleSelectSession = useCallback(async (session: Session) => {
    setLoading(true);
    setError(null);
    setSelectedNodeId(null);
    const seq = ++fetchSeqRef.current;
    try {
      const tree = await fetchTree(session.id);
      if (seq !== fetchSeqRef.current) return;
      if (tree.nodes.length === 0) {
        setError("该会话为空，无法作为复制目标");
        return;
      }
      setSelectedSession(session);
      setTreeNodes(tree.nodes);
      setStep("pick-node");
    } catch {
      if (seq !== fetchSeqRef.current) return;
      setError("加载会话树失败");
    } finally {
      if (seq === fetchSeqRef.current) setLoading(false);
    }
  }, []);

  const canAbove = useCallback((): boolean => {
    if (!selectedNodeId) return false;
    const node = findNode(treeNodes, selectedNodeId);
    if (!node) return false;
    const parent = findParentOf(treeNodes, selectedNodeId);
    if (!parent) return treeNodes.length < 3;
    return parent.children.length < 3;
  }, [selectedNodeId, treeNodes]);

  const canBelow = useCallback((): boolean => {
    if (!selectedNodeId) return false;
    const node = findNode(treeNodes, selectedNodeId);
    if (!node || !node.ai) return false;
    return node.children.length < 3;
  }, [selectedNodeId, treeNodes]);

  const canReplace = useCallback((): boolean => {
    if (!selectedNodeId) return false;
    const node = findNode(treeNodes, selectedNodeId);
    return node !== null && node.ai !== null;
  }, [selectedNodeId, treeNodes]);

  const handleCopy = useCallback(
    async (mode: "above" | "below" | "replace") => {
      if (!selectedSession || !selectedNodeId) return;
      setLoading(true);
      setError(null);
      try {
        const result = await copyMessage(
          sourceHumanId,
          selectedNodeId,
          selectedSession.id,
          mode
        );
        await onCopied(selectedSession.id, result.new_human_id);
        onClose();
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "复制失败");
      } finally {
        setLoading(false);
      }
    },
    [sourceHumanId, selectedSession, selectedNodeId, onCopied, onClose]
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-gray-800 rounded-lg w-[480px] max-h-[80vh] flex flex-col shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
          <h2 className="text-sm font-semibold text-gray-100">复制节点到其他会话</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200 text-lg leading-none"
          >
            ×
          </button>
        </div>

        {/* Source preview */}
        <div className="px-4 py-2 border-b border-gray-700 text-xs text-gray-400">
          复制内容：<span className="text-gray-200">{sourceContent.slice(0, 60)}</span>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-4 py-3 min-h-0">
          {error && (
            <div className="mb-2 text-xs text-red-400 bg-red-900/30 px-2 py-1 rounded">
              {error}
            </div>
          )}

          {step === "pick-session" && (
            <>
              <p className="text-xs text-gray-400 mb-2">选择目标会话：</p>
              {sessions.length === 0 ? (
                <p className="text-xs text-gray-500">没有其他会话可复制</p>
              ) : (
                <ul className="space-y-1">
                  {sessions.map((s) => (
                    <li key={s.id}>
                      <button
                        onClick={() => handleSelectSession(s)}
                        disabled={loading}
                        className="w-full text-left px-3 py-2 rounded bg-gray-700 hover:bg-gray-600 text-sm text-gray-200 disabled:opacity-50 truncate"
                      >
                        {s.title}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}

          {step === "pick-node" && (
            <>
              <p className="text-xs text-gray-400 mb-2">
                在「{selectedSession?.title}」中选择目标节点：
              </p>
              <MiniTree
                nodes={treeNodes}
                selectedId={selectedNodeId}
                onSelect={setSelectedNodeId}
              />
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-gray-700 flex items-center gap-2 justify-end">
          {step === "pick-node" && (
            <button
              onClick={() => {
                setStep("pick-session");
                setSelectedNodeId(null);
                setError(null);
              }}
              className="px-3 py-1.5 text-xs bg-gray-700 hover:bg-gray-600 rounded text-gray-300"
            >
              返回
            </button>
          )}
          {step === "pick-node" && (
            <>
              <button
                onClick={() => handleCopy("above")}
                disabled={!canAbove() || loading}
                title={canAbove() ? "成为目标节点的父节点" : "上方分支数已达上限"}
                className="px-3 py-1.5 text-xs bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 rounded text-white"
              >
                插入上方
              </button>
              <button
                onClick={() => handleCopy("below")}
                disabled={!canBelow() || loading}
                title={canBelow() ? "成为目标节点的子节点" : "下方分支数已达上限或目标无 AI 回复"}
                className="px-3 py-1.5 text-xs bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 rounded text-white"
              >
                插入下方
              </button>
              <button
                onClick={() => handleCopy("replace")}
                disabled={!canReplace() || loading}
                title={canReplace() ? "替换目标节点" : "目标节点缺少 AI 回复"}
                className="px-3 py-1.5 text-xs bg-orange-600 hover:bg-orange-500 disabled:bg-gray-700 disabled:text-gray-500 rounded text-white"
              >
                替换
              </button>
            </>
          )}
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-xs bg-gray-700 hover:bg-gray-600 rounded text-gray-300"
          >
            取消
          </button>
        </div>
      </div>
    </div>
  );
}
