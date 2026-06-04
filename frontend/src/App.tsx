import { Component, type ReactNode, useEffect } from "react";
import { AppProvider, useAppState } from "./lib/AppContext";
import { SessionList } from "./components/sidebar/session-list";
import { DocumentPanel } from "./components/sidebar/document-panel";
import { SystemPromptEditor } from "./components/sidebar/system-prompt";
import { ChatArea } from "./components/chat/chat-area";
import { ChatInput } from "./components/chat/chat-input";
import { SessionTree } from "./components/tree/session-tree";

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div className="flex h-screen items-center justify-center bg-gray-900 text-gray-100">
          <div className="text-center max-w-md">
            <h1 className="text-xl font-bold text-red-400 mb-2">渲染错误</h1>
            <p className="text-sm text-gray-400 mb-4 break-all">{this.state.error.message}</p>
            <button
              onClick={() => this.setState({ error: null })}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm"
            >
              重试
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

function AppInner() {
  const { loadSessions, treeData, activePath, switchBranch } = useAppState();

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  return (
    <div className="flex h-screen bg-gray-800 text-gray-100">
      {/* 左侧栏 */}
      <aside className="w-60 flex-shrink-0 border-r border-gray-700 bg-gray-900 flex flex-col">
        <div className="p-3 text-sm font-bold border-b border-gray-700 text-gray-300">
          📋 历史会话
        </div>
        <SessionList />
        <div className="border-t border-gray-700 p-3 flex flex-col gap-3">
          <div className="text-sm font-bold text-gray-300">📁 文档管理</div>
          <DocumentPanel />
          <div className="border-t border-gray-700 pt-2">
            <SystemPromptEditor />
          </div>
        </div>
      </aside>

      {/* 主对话区域 */}
      <main className="flex-1 flex flex-col min-w-0">
        <ChatArea />
        <ChatInput />
      </main>

      {/* 右侧栏 — 会话树 */}
      <aside className="w-52 flex-shrink-0 border-l border-gray-700 bg-gray-900">
        <SessionTree
          nodes={treeData}
          activePath={activePath}
          onNodeClick={(humanId) => switchBranch(humanId, null)}
        />
      </aside>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <AppProvider>
        <AppInner />
      </AppProvider>
    </ErrorBoundary>
  );
}
