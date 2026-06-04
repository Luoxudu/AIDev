import {
  createContext,
  useContext,
  useState,
  useCallback,
  useRef,
  type ReactNode,
} from "react";
import type { ChatMessage, Sibling, ToolCall, TreeNode } from "./types";
import {
  streamChat,
  fetchSessions,
  createSession,
  deleteSession,
  deleteMessage,
  fetchTree,
} from "./api";
import type { Session } from "./types";

interface AppState {
  sessions: Session[];
  currentSession: Session | null;
  messages: ChatMessage[];
  treeData: TreeNode[];
  isStreaming: boolean;
  leafAiId: string | null;
  activePath: string[]; // human.id path marking active branch
  loadSessions: () => Promise<void>;
  selectSession: (session: Session, targetHumanId?: string) => Promise<void>;
  createNewSession: () => Promise<void>;
  removeSession: (id: string) => Promise<void>;
  sendMessage: (query: string, parentAiId?: string | null, editPosition?: number) => Promise<void>;
  stopGeneration: () => void;
  switchBranch: (humanId: string, aiId: string | null) => Promise<void>;
  deleteNode: (humanMsgId: string) => Promise<void>;
}

const AppContext = createContext<AppState | null>(null);

function _uuid(): string {
  try {
    return crypto.randomUUID();
  } catch {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  }
}

/** Convert tree to flat ChatMessage[] along the active path. */
function treeToMessages(
  nodes: TreeNode[],
  activePath: string[]
): ChatMessage[] {
  const msgs: ChatMessage[] = [];

  function walk(nodeList: TreeNode[], depth: number) {
    for (const node of nodeList) {
      // Compute siblings: all children of the same parent
      const siblings: Sibling[] = nodeList.map((n) => ({
        humanId: n.human.id,
        aiId: n.ai?.id ?? null,
        content: n.human.content,
      }));
      const siblingIndex = nodeList.findIndex(
        (n) => n.human.id === node.human.id
      );

      // Check if this node is on the active path
      const isActive =
        activePath.length === 0 || activePath[depth] === node.human.id;

      if (!isActive) continue;

      msgs.push({
        id: node.human.id,
        role: "human",
        content: node.human.content,
        toolCalls: [],
        interrupted: false,
        siblings,
        siblingIndex,
      });
      if (node.ai) {
        msgs.push({
          id: node.ai.id,
          role: "ai",
          content: node.ai.content,
          toolCalls: node.ai.tool_calls,
          interrupted: node.ai.interrupted,
          siblings: [],
          siblingIndex: 0,
        });
      }
      // Recurse into children
      if (node.children.length > 0) {
        walk(node.children, depth + 1);
      }
    }
  }

  walk(nodes, 0);
  return msgs;
}

/** Extract active path from tree by always picking the first child. */
function defaultActivePath(nodes: TreeNode[]): string[] {
  const path: string[] = [];
  let current = nodes;
  while (current.length > 0) {
    path.push(current[0].human.id);
    current = current[0].children;
  }
  return path;
}

/** Build active path to the node whose parent ai id matches `parentAiId`,
 *  extending to the latest child if a new message was just added under it.
 *  Falls back to defaultActivePath if the parent can't be found. */
function pathToParentAi(
  nodes: TreeNode[],
  parentAiId: string | null
): string[] {
  if (parentAiId === null) {
    // First message of a session — pick the latest root node
    if (nodes.length === 0) return [];
    const latest = nodes.reduce((a, b) =>
      a.human.created_at > b.human.created_at ? a : b
    );
    return latest.children.length > 0
      ? [latest.human.id, latest.children[latest.children.length - 1].human.id]
      : [latest.human.id];
  }

  function search(nodeList: TreeNode[], path: string[]): string[] | null {
    for (const node of nodeList) {
      if (node.ai?.id === parentAiId) {
        const newPath = [...path, node.human.id];
        if (node.children.length > 0) {
          const latestChild = node.children[node.children.length - 1];
          return [...newPath, latestChild.human.id];
        }
        return newPath;
      }
      const found = search(node.children, [...path, node.human.id]);
      if (found) return found;
    }
    return null;
  }

  return search(nodes, []) ?? defaultActivePath(nodes);
}

/** Get leaf ai id from active path. */
function getLeafAiId(
  nodes: TreeNode[],
  activePath: string[]
): string | null {
  let current = nodes;
  for (let i = 0; i < activePath.length; i++) {
    const node = current.find((n) => n.human.id === activePath[i]);
    if (!node) break;
    if (i === activePath.length - 1) {
      return node.ai?.id ?? null;
    }
    current = node.children;
  }
  return null;
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [currentSession, setCurrentSession] = useState<Session | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [treeData, setTreeData] = useState<TreeNode[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const streamingRef = useRef(false); // sync guard for TOCTOU race
  const [leafAiId, setLeafAiId] = useState<string | null>(null);
  const [activePath, setActivePath] = useState<string[]>([]);
  const [abortController, setAbortController] =
    useState<AbortController | null>(null);

  const refreshTree = useCallback(
    async (sessionId: string) => {
      const tree = await fetchTree(sessionId);
      setTreeData(tree.nodes);
      return tree.nodes;
    },
    []
  );

  const refreshMessages = useCallback(
    (nodes: TreeNode[], path: string[]) => {
      const msgs = treeToMessages(nodes, path);
      setMessages(msgs);
      const leaf = getLeafAiId(nodes, path);
      setLeafAiId(leaf);
    },
    []
  );

  const loadSessions = useCallback(async () => {
    const list = await fetchSessions();
    setSessions(list);
  }, []);

  const selectSession = useCallback(
    async (session: Session, targetHumanId?: string) => {
      if (isStreaming) return;
      setCurrentSession(session);
      const nodes = await refreshTree(session.id);
      const path = targetHumanId
        ? (buildPathToNode(nodes, targetHumanId) || defaultActivePath(nodes))
        : defaultActivePath(nodes);
      setActivePath(path);
      refreshMessages(nodes, path);
    },
    [isStreaming, refreshTree, refreshMessages]
  );

  const createNewSession = useCallback(async () => {
    if (isStreaming) return;
    const session = await createSession();
    setSessions((prev) => [session, ...prev]);
    setCurrentSession(session);
    setMessages([]);
    setTreeData([]);
    setLeafAiId(null);
    setActivePath([]);
  }, [isStreaming]);

  const removeSession = useCallback(
    async (id: string) => {
      if (isStreaming) return;
      await deleteSession(id);
      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (currentSession?.id === id) {
        setCurrentSession(null);
        setMessages([]);
        setTreeData([]);
        setLeafAiId(null);
        setActivePath([]);
      }
    },
    [isStreaming, currentSession]
  );

  const sendMessage = useCallback(
    async (query: string, parentAiId?: string | null, editPosition?: number) => {
      if (!currentSession || isStreaming || streamingRef.current) return;
      streamingRef.current = true;

      const effectiveParentAi = parentAiId !== undefined ? parentAiId : leafAiId;
      const isEdit = editPosition !== undefined;

      const humanMsg: ChatMessage = {
        id: _uuid(),
        role: "human",
        content: query,
        toolCalls: [],
        interrupted: false,
        siblings: [],
        siblingIndex: 0,
      };
      const aiMsg: ChatMessage = {
        id: _uuid(),
        role: "ai",
        content: "",
        toolCalls: [],
        interrupted: false,
        streaming: true,
        siblings: [],
        siblingIndex: 0,
      };

      if (isEdit) {
        // 编辑模式：截断编辑位置之后的所有消息，原位插入新分支
        setMessages((prev) => [...prev.slice(0, editPosition), humanMsg, aiMsg]);
      } else {
        // 新建消息：追加到末尾
        setMessages((prev) => [...prev, humanMsg, aiMsg]);
      }
      setIsStreaming(true);

      const controller = new AbortController();
      setAbortController(controller);

      let accumulatedContent = "";
      let toolCallsLog: ToolCall[] = [];

      // 流式事件增量更新最后一条消息（编辑/新建都适用，都是数组末尾的 AI 占位）
      const updateLastMessage = (patch: Partial<ChatMessage>) => {
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            ...patch,
          };
          return updated;
        });
      };

      try {
        await streamChat(
          currentSession.id,
          query,
          effectiveParentAi,
          (event) => {
            switch (event.type) {
              case "token":
                accumulatedContent += event.text;
                updateLastMessage({ content: accumulatedContent });
                break;
              case "tool_start":
                toolCallsLog = [
                  ...toolCallsLog,
                  { name: event.name, input: event.input, output: "" },
                ];
                updateLastMessage({ toolCalls: [...toolCallsLog] });
                break;
              case "tool_end": {
                const idx = toolCallsLog.findLastIndex(
                  (tc) => tc.name === event.name && tc.output === ""
                );
                if (idx !== -1) {
                  toolCallsLog = toolCallsLog.map((tc, i) =>
                    i === idx ? { ...tc, output: event.output } : tc
                  );
                }
                updateLastMessage({ toolCalls: [...toolCallsLog] });
                break;
              }
              case "error":
                updateLastMessage({
                  content: accumulatedContent || `错误: ${event.message}`,
                  streaming: false,
                });
                break;
              case "done":
                updateLastMessage({ streaming: false });
                break;
            }
          },
          controller.signal
        );
      } catch {
        // streamChat 异常由内部 onEvent('error') 处理，这里静默吞咽
      } finally {
        setIsStreaming(false);
        setAbortController(null);
      }

      // 无论流成功或失败，始终刷新树和消息，确保 streamingRef 正确复位
      try {
        await loadSessions();
        const nodes = await refreshTree(currentSession.id);
        const newPath = pathToParentAi(nodes, effectiveParentAi);
        setActivePath(newPath);
        refreshMessages(nodes, newPath);
      } catch {
        // 刷新失败不阻断流式状态复位
      } finally {
        streamingRef.current = false;
      }
    },
    [currentSession, isStreaming, leafAiId, loadSessions, refreshTree, refreshMessages]
  );

  const switchBranch = useCallback(
    async (humanId: string, _aiId: string | null) => {
      if (isStreaming || streamingRef.current) return;
      const newPath = buildPathToNode(treeData, humanId);
      if (newPath.length === 0) return;
      setActivePath(newPath);
      refreshMessages(treeData, newPath);
    },
    [isStreaming, treeData, refreshMessages]
  );

  const deleteNodeAction = useCallback(
    async (humanMsgId: string) => {
      if (isStreaming || !currentSession) return;
      // Find grandparent before deletion to preserve branch position
      let anchorId: string | null = null;
      const idx = messages.findIndex((m) => m.id === humanMsgId);
      // The grandparent is the human message two positions before this one
      if (idx >= 2 && messages[idx - 2]?.role === "human") {
        anchorId = messages[idx - 2].id;
      }
      await deleteMessage(humanMsgId);
      const [nodes] = await Promise.all([
        refreshTree(currentSession.id),
        loadSessions(),
      ]);
      // Try to land on the anchor's subtree; fall back to default
      const newPath = anchorId
        ? (buildPathToNode(nodes, anchorId) ?? defaultActivePath(nodes))
        : defaultActivePath(nodes);
      setActivePath(newPath);
      refreshMessages(nodes, newPath);
    },
    [isStreaming, currentSession, messages, refreshTree, refreshMessages, loadSessions]
  );

  const stopGeneration = useCallback(() => {
    if (abortController) {
      abortController.abort();
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          streaming: false,
          interrupted: true,
        };
        return updated;
      });
    }
  }, [abortController]);

  return (
    <AppContext.Provider
      value={{
        sessions,
        currentSession,
        messages,
        treeData,
        isStreaming,
        leafAiId,
        activePath,
        loadSessions,
        selectSession,
        createNewSession,
        removeSession,
        sendMessage,
        stopGeneration,
        switchBranch,
        deleteNode: deleteNodeAction,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

function buildPathToNode(
  nodes: TreeNode[],
  targetHumanId: string
): string[] {
  function search(
    nodeList: TreeNode[],
    path: string[]
  ): string[] | null {
    for (const node of nodeList) {
      const newPath = [...path, node.human.id];
      if (node.human.id === targetHumanId) {
        return newPath;
      }
      const found = search(node.children, newPath);
      if (found) return found;
    }
    return null;
  }
  return search(nodes, []) ?? [];
}

export function useAppState() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useAppState must be used within AppProvider");
  return ctx;
}
