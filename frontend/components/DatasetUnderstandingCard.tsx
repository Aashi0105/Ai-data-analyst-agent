"use client";

import React, { useState } from "react";
import { DatasetMetadata } from "@/types";
import {
  Sparkles,
  Info,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  Lightbulb,
  CheckCircle2,
  Layers,
  ArrowRight,
} from "lucide-react";

interface DatasetUnderstandingCardProps {
  dataset: DatasetMetadata;
  onSelectQuestion: (question: string) => void;
  isLoading: boolean;
}

export const DatasetUnderstandingCard: React.FC<DatasetUnderstandingCardProps> = ({
  dataset,
  onSelectQuestion,
  isLoading,
}) => {
  const [isExpanded, setIsExpanded] = useState(true);

  const understanding = dataset.understanding;
  const profile = dataset.profile;

  if (!understanding) {
    return null;
  }

  const isFallback = understanding.status === "fallback";
  const warnings = profile?.warnings || [];

  return (
    <div className="mb-6 rounded-lg border border-editorial-border bg-surface shadow-sm overflow-hidden transition-all duration-200">
      {/* Header Bar */}
      <div className="px-5 py-3.5 bg-cream/70 border-b border-editorial-border flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="p-1 rounded bg-editorial-accent/10 text-editorial-accent">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-charcoal flex items-center space-x-2">
              <span>Dataset Understanding</span>
              {isFallback ? (
                <span className="text-[10px] normal-case font-normal px-2 py-0.5 rounded-full bg-stone-200/80 text-stone-700 border border-stone-300">
                  Deterministic Profile Mode
                </span>
              ) : (
                <span className="text-[10px] normal-case font-normal px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-200 flex items-center space-x-1">
                  <CheckCircle2 className="w-2.5 h-2.5" />
                  <span>AI Grounded</span>
                </span>
              )}
            </h2>
          </div>
        </div>

        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-xs text-charcoal-muted hover:text-charcoal flex items-center space-x-1 px-2 py-1 rounded hover:bg-stone-200/50 transition"
          aria-label={isExpanded ? "Collapse dataset understanding" : "Expand dataset understanding"}
        >
          <span className="text-[11px] font-medium">{isExpanded ? "Hide Details" : "Show Details"}</span>
          {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>
      </div>

      {/* Main Content Area */}
      <div className="p-5 space-y-4">
        {/* Plain-English Overview */}
        <div>
          <p className="text-sm text-charcoal leading-relaxed font-serif">
            {understanding.overview}
          </p>
        </div>

        {/* Collapsible Details */}
        {isExpanded && (
          <>
            {/* Factual Metrics Bar */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 pt-2 pb-1">
              <div className="p-2.5 rounded border border-editorial-border/70 bg-cream/30">
                <span className="text-[10px] uppercase tracking-wider text-charcoal-subtle font-medium block">
                  Records
                </span>
                <span className="text-sm font-semibold text-charcoal font-mono">
                  {dataset.num_rows.toLocaleString()}
                </span>
              </div>
              <div className="p-2.5 rounded border border-editorial-border/70 bg-cream/30">
                <span className="text-[10px] uppercase tracking-wider text-charcoal-subtle font-medium block">
                  Attributes
                </span>
                <span className="text-sm font-semibold text-charcoal font-mono">
                  {dataset.num_cols} columns
                </span>
              </div>
              <div className="p-2.5 rounded border border-editorial-border/70 bg-cream/30">
                <span className="text-[10px] uppercase tracking-wider text-charcoal-subtle font-medium block">
                  Missing Cells
                </span>
                <span
                  className={`text-sm font-semibold font-mono ${
                    dataset.total_missing > 0 ? "text-amber-700" : "text-charcoal"
                  }`}
                >
                  {dataset.total_missing} ({profile?.missing_cell_pct ?? 0}%)
                </span>
              </div>
              <div className="p-2.5 rounded border border-editorial-border/70 bg-cream/30">
                <span className="text-[10px] uppercase tracking-wider text-charcoal-subtle font-medium block">
                  Duplicate Rows
                </span>
                <span
                  className={`text-sm font-semibold font-mono ${
                    dataset.duplicate_rows > 0 ? "text-amber-700" : "text-charcoal"
                  }`}
                >
                  {dataset.duplicate_rows} ({profile?.duplicate_row_pct ?? 0}%)
                </span>
              </div>
            </div>

            {/* Inferred Column Meanings */}
            {understanding.column_insights && understanding.column_insights.length > 0 && (
              <div className="pt-2">
                <span className="text-[11px] uppercase tracking-wider text-charcoal-subtle font-semibold block mb-2">
                  Inferred Column Meanings
                </span>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                  {understanding.column_insights.slice(0, 6).map((item, idx) => (
                    <div
                      key={idx}
                      className="p-2.5 rounded border border-editorial-border/60 bg-cream/20 flex flex-col space-y-0.5"
                    >
                      <span className="font-mono text-[11px] font-medium text-editorial-accent">
                        {item.column}
                      </span>
                      <span className="text-charcoal-muted text-[11px] leading-snug">
                        {item.inferred_meaning}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Capabilities and Quality Notes */}
            <div className="pt-2 grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
              <div className="p-3 rounded border border-editorial-border/70 bg-cream/20">
                <div className="flex items-center space-x-1.5 text-editorial-accent font-semibold mb-1">
                  <Lightbulb className="w-3.5 h-3.5" />
                  <span className="text-[11px] uppercase tracking-wider">Analytical Potential</span>
                </div>
                <p className="text-charcoal-muted leading-relaxed text-[11px]">
                  {understanding.analytical_potential}
                </p>
              </div>

              <div className="p-3 rounded border border-editorial-border/70 bg-cream/20">
                <div className="flex items-center space-x-1.5 text-stone-700 font-semibold mb-1">
                  <Info className="w-3.5 h-3.5" />
                  <span className="text-[11px] uppercase tracking-wider">Data Quality & Constraints</span>
                </div>
                <p className="text-charcoal-muted leading-relaxed text-[11px]">
                  {understanding.data_quality_notes}
                </p>
              </div>
            </div>

            {/* Quality Warnings (if any) */}
            {warnings.length > 0 && (
              <div className="p-3 rounded border border-amber-200 bg-amber-50/70 text-amber-900 text-xs space-y-1.5">
                <div className="flex items-center space-x-1.5 font-semibold text-amber-800">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-700" />
                  <span>Quality Caveats & Observations</span>
                </div>
                <ul className="list-disc list-inside space-y-1 text-[11px] text-amber-900/90 pl-1">
                  {warnings.slice(0, 3).map((w, wIdx) => (
                    <li key={wIdx}>{w.message}</li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}

        {/* Five Selectable Grounded Questions */}
        <div className="pt-3 border-t border-editorial-border">
          <div className="flex items-center justify-between mb-2.5">
            <span className="text-[11px] uppercase tracking-wider text-charcoal-subtle font-semibold flex items-center space-x-1.5">
              <Sparkles className="w-3.5 h-3.5 text-editorial-accent" />
              <span>Suggested Analytical Questions (Click to Run)</span>
            </span>
          </div>

          <div className="flex flex-wrap gap-2">
            {understanding.suggested_questions.map((prompt, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => onSelectQuestion(prompt)}
                disabled={isLoading}
                className="group text-left text-xs text-charcoal hover:text-editorial-accent bg-cream/60 hover:bg-editorial-hover px-3 py-2 rounded-md border border-editorial-border hover:border-editorial-accent/60 transition flex items-center space-x-2 disabled:opacity-50"
              >
                <span className="text-[11px] font-mono text-charcoal-subtle group-hover:text-editorial-accent">
                  {idx + 1}.
                </span>
                <span className="text-xs font-normal leading-snug">{prompt}</span>
                <ArrowRight className="w-3 h-3 text-charcoal-subtle group-hover:text-editorial-accent shrink-0 opacity-0 group-hover:opacity-100 transition" />
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
