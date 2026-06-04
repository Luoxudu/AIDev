import type { TreeNode } from "../../lib/types";

interface Props {
  nodes: TreeNode[];
  activePath: string[];
  onNodeClick: (humanId: string) => void;
}

export function SessionTree({ nodes, activePath, onNodeClick }: Props) {
  return (
    <div className="p-3 text-xs font-mono overflow-y-auto h-full">
      <div className="text-gray-400 mb-2 text-sm font-sans font-bold">
        会话树
      </div>
      {nodes.length === 0 && (
        <div className="text-gray-500 italic">暂无消息</div>
      )}
      <TreeLevel
        nodes={nodes}
        activePath={activePath}
        depth={0}
        onNodeClick={onNodeClick}
      />
    </div>
  );
}

function TreeLevel({
  nodes,
  activePath,
  depth,
  onNodeClick,
}: {
  nodes: TreeNode[];
  activePath: string[];
  depth: number;
  onNodeClick: (humanId: string) => void;
}) {
  return (
    <div className="ml-2">
      {nodes.map((node) => {
        const isActive = activePath[depth] === node.human.id;
        const hasBranches = node.children.length > 1;
        const label = node.human.content.slice(0, 15) + (node.human.content.length > 15 ? "…" : "");

        return (
          <div key={node.human.id}>
            <div
              className={`flex items-center gap-1 py-0.5 px-1 rounded cursor-pointer hover:bg-gray-700/50 ${
                isActive ? "bg-blue-900/40 text-blue-300" : "text-gray-400"
              }`}
              onClick={() => onNodeClick(node.human.id)}
              title={node.human.content}
            >
              <span>{isActive ? "●" : "○"}</span>
              <span className="truncate">{label}</span>
              {hasBranches && (
                <span className="text-yellow-400 text-[10px]">
                  ({node.children.length})
                </span>
              )}
            </div>
            {isActive && node.children.length > 0 && (
              <TreeLevel
                nodes={node.children}
                activePath={activePath}
                depth={depth + 1}
                onNodeClick={onNodeClick}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
