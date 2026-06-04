export interface Session {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ToolCall {
  name: string;
  input: string;
  output: string;
}

export interface TreeNode {
  human: {
    id: string;
    content: string;
    branch_index: number;
    created_at: string;
  };
  ai: {
    id: string;
    content: string;
    tool_calls: ToolCall[];
    interrupted: boolean;
    created_at: string;
  } | null;
  children: TreeNode[];
}

export interface TreeData {
  session_id: string;
  nodes: TreeNode[];
}

export interface ChatMessage {
  id: string;
  role: "human" | "ai";
  content: string;
  toolCalls: ToolCall[];
  interrupted: boolean;
  streaming?: boolean;
  /** siblings at the same parent level (parallel branches) */
  siblings: Sibling[];
  /** index into siblings[] for current message */
  siblingIndex: number;
}

export interface Sibling {
  humanId: string;
  aiId: string | null;
  content: string;
}

export type SSEEvent =
  | { type: "token"; text: string }
  | { type: "tool_start"; name: string; input: string }
  | { type: "tool_end"; name: string; output: string }
  | { type: "done" }
  | { type: "error"; message: string };
