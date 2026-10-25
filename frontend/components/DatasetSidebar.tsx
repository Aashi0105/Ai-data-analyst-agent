"use client";

import React from "react";
import { DatasetMetadata } from "@/types";
import { FileText, Database, Hash, AlertTriangle, Layers } from "lucide-react";

interface DatasetSidebarProps {
  dataset: DatasetMetadata;
  onOpenInspector: () => void;
}

export const DatasetSidebar: React.FC<DatasetSidebarProps> = ({
  dataset,
  onOpenInspector,
}) => {
  return (
    <aside className="w-full lg:w-64 shrink-0 border-r border-editorial-border bg-cream p-5 flex flex-col space-y-6 overflow-y-auto">
      {/* Dataset Identity */}
      <div>
        <span className="text-[10px] uppercase tracking-wider text-charcoal-subtle font-semibold block mb-1">
          Active Dataset
        </span>
        <h2 className="text-sm font-semibold text-charcoal truncate flex items-center space-x-1.5" title={dataset.filename}>
          <FileText className="w-4 h-4 text-editorial-accent shrink-0" />
          <span className="truncate">{dataset.filename}</span>
        </h2>
      </div>

      {/* High-level KPIs */}
      <div className="grid grid-cols-2 gap-2.5">
        <div className="p-2.5 rounded border border-editorial-border bg-surface">
          <span className="text-[11px] text-charcoal-muted block">Rows</span>
          <span className="text-base font-semibold text-charcoal">
            {dataset.num_rows.toLocaleString()}
          </span>
        </div>
        <div className="p-2.5 rounded border border-editorial-border bg-surface">
          <span className="text-[11px] text-charcoal-muted block">Columns</span>
          <span className="text-base font-semibold text-charcoal">
            {dataset.num_cols}
          </span>
        </div>
        <div className="p-2.5 rounded border border-editorial-border bg-surface">
          <span className="text-[11px] text-charcoal-muted block">Missing</span>
          <span className={`text-base font-semibold ${dataset.total_missing > 0 ? "text-amber-700" : "text-charcoal"}`}>
            {dataset.total_missing}
          </span>
        </div>
        <div className="p-2.5 rounded border border-editorial-border bg-surface">
          <span className="text-[11px] text-charcoal-muted block">Duplicates</span>
          <span className="text-base font-semibold text-charcoal">
            {dataset.duplicate_rows}
          </span>
        </div>
      </div>

      {/* Columns List with Types */}
      <div className="flex-1 flex flex-col min-h-0">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[11px] uppercase tracking-wider text-charcoal-subtle font-semibold">
            Columns ({dataset.columns.length})
          </span>
          <button
            onClick={onOpenInspector}
            className="text-[11px] text-editorial-accent hover:underline font-medium"
          >
            View All
          </button>
        </div>

        <div className="space-y-1.5 overflow-y-auto pr-1 flex-1 max-h-[420px]">
          {dataset.schema.map((item, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between text-xs py-1.5 px-2 rounded border border-editorial-border/60 bg-surface/70 hover:bg-surface transition"
            >
              <span className="font-mono text-[11px] text-charcoal truncate max-w-[130px]" title={item.column}>
                {item.column}
              </span>
              <span className="text-[10px] text-charcoal-subtle px-1.5 py-0.5 rounded bg-cream border border-editorial-border/70 font-mono">
                {item.type.replace("int64", "int").replace("float64", "float").replace("object", "str")}
              </span>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
};
