"use client";

import React from "react";
import { HistoryItem } from "@/types";
import { History, ArrowRight } from "lucide-react";

interface QueryHistoryProps {
  history: HistoryItem[];
  activeId?: string;
  onSelect: (item: HistoryItem) => void;
}

export const QueryHistory: React.FC<QueryHistoryProps> = ({
  history,
  activeId,
  onSelect,
}) => {
  if (history.length === 0) return null;

  return (
    <div className="mt-8 pt-5 border-t border-editorial-border">
      <div className="flex items-center space-x-2 mb-3">
        <History className="w-3.5 h-3.5 text-charcoal-muted" />
        <span className="text-xs uppercase tracking-wider font-semibold text-charcoal-subtle">
          Recent Analyses ({history.length})
        </span>
      </div>

      <div className="flex flex-wrap gap-2">
        {history.map((item) => (
          <button
            key={item.id}
            onClick={() => onSelect(item)}
            className={`text-xs px-3 py-1.5 rounded-full border transition flex items-center space-x-1.5 ${
              activeId === item.id
                ? "border-editorial-accent bg-editorial-accent text-white"
                : "border-editorial-border bg-surface text-charcoal hover:border-charcoal-subtle"
            }`}
          >
            <span className="truncate max-w-[200px]">{item.question}</span>
            <ArrowRight className="w-3 h-3 opacity-60" />
          </button>
        ))}
      </div>
    </div>
  );
};
