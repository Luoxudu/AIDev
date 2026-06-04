import type { Sibling } from "../../lib/types";

interface Props {
  siblings: Sibling[];
  currentIndex: number;
  onSwitch: (humanId: string) => void;
}

export function BranchIndicator({ siblings, currentIndex, onSwitch }: Props) {
  if (siblings.length <= 1) return null;

  return (
    <div className="flex items-center gap-1 mb-1 text-xs text-gray-400">
      <button
        onClick={() => {
          if (currentIndex > 0) onSwitch(siblings[currentIndex - 1].humanId);
        }}
        disabled={currentIndex === 0}
        className="px-1.5 py-0.5 rounded hover:bg-gray-600 disabled:opacity-30 disabled:cursor-not-allowed"
      >
        ◀
      </button>
      <span>
        {currentIndex + 1}/{siblings.length}
      </span>
      <button
        onClick={() => {
          if (currentIndex < siblings.length - 1)
            onSwitch(siblings[currentIndex + 1].humanId);
        }}
        disabled={currentIndex === siblings.length - 1}
        className="px-1.5 py-0.5 rounded hover:bg-gray-600 disabled:opacity-30 disabled:cursor-not-allowed"
      >
        ▶
      </button>
    </div>
  );
}
