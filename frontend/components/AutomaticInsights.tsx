"use client";

import React, { useState } from "react";
import { Insight, VisualizationData } from "@/types";
import {
  TrendingUp,
  BarChart2,
  AlertCircle,
  Activity,
  ShieldAlert,
  Lightbulb,
  Info,
  ArrowRight,
  ChevronRight,
  Sparkles,
  Layers,
} from "lucide-react";
import { ResultVisualization } from "./ResultVisualization";

interface AutomaticInsightsProps {
  insights?: Insight[];
  onSelectQuery?: (query: string) => void;
  isLoading?: boolean;
}

export const AutomaticInsights: React.FC<AutomaticInsightsProps> = ({
  insights,
  onSelectQuery,
  isLoading = false,
}) => {
  const [selectedIndex, setSelectedIndex] = useState<number>(0);

  if (!insights || insights.length === 0) {
    return (
      <div className="mb-6 p-4 rounded-lg border border-editorial-border bg-surface text-charcoal-muted text-xs flex items-center space-x-2">
        <Info className="w-4 h-4 text-editorial-accent shrink-0" />
        <span>No proactive statistical anomalies or dominant trends detected in this dataset.</span>
      </div>
    );
  }

  const activeInsight = insights[Math.min(selectedIndex, insights.length - 1)];

  const getTypeMeta = (type: string) => {
    switch (type) {
      case "segmentation":
        return {
          label: "Segment Dominance",
          icon: <BarChart2 className="w-3.5 h-3.5" />,
          colorClass: "bg-emerald-50 text-emerald-800 border-emerald-200",
        };
      case "trend":
        return {
          label: "Temporal Trajectory",
          icon: <TrendingUp className="w-3.5 h-3.5" />,
          colorClass: "bg-blue-50 text-blue-800 border-blue-200",
        };
      case "outlier":
        return {
          label: "Statistical Outlier",
          icon: <AlertCircle className="w-3.5 h-3.5" />,
          colorClass: "bg-amber-50 text-amber-900 border-amber-200",
        };
      case "correlation":
        return {
          label: "Correlation",
          icon: <Activity className="w-3.5 h-3.5" />,
          colorClass: "bg-purple-50 text-purple-900 border-purple-200",
        };
      case "data_quality":
        return {
          label: "Data Hygiene",
          icon: <ShieldAlert className="w-3.5 h-3.5" />,
          colorClass: "bg-rose-50 text-rose-900 border-rose-200",
        };
      default:
        return {
          label: "Pattern Finding",
          icon: <Sparkles className="w-3.5 h-3.5" />,
          colorClass: "bg-stone-50 text-stone-800 border-stone-200",
        };
    }
  };

  const activeMeta = getTypeMeta(activeInsight.type);

  return (
    <div className="mb-6 rounded-lg border border-editorial-border bg-surface shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-5 py-3.5 bg-cream/70 border-b border-editorial-border flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="p-1 rounded bg-editorial-accent/10 text-editorial-accent">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-charcoal flex items-center space-x-2">
              <span>Automatic Insights Discovery</span>
              <span className="text-[10px] normal-case font-normal px-2 py-0.5 rounded-full bg-editorial-accent/10 text-editorial-accent border border-editorial-accent/20">
                {insights.length} Evidence-Backed Finding{insights.length > 1 ? "s" : ""}
              </span>
            </h3>
          </div>
        </div>
        <span className="text-[11px] text-charcoal-subtle font-mono">100% Deterministic</span>
      </div>

      {/* Main Body with Insight Selector and Detailed Active View */}
      <div className="grid grid-cols-1 lg:grid-cols-12 divide-y lg:divide-y-0 lg:divide-x divide-editorial-border">
        {/* Left Column: Insight Navigator / List */}
        <div className="lg:col-span-5 p-3 space-y-2 bg-cream/20">
          <span className="text-[10px] uppercase font-semibold text-charcoal-subtle tracking-wider px-2 block pt-1">
            Ranked Discoveries
          </span>
          <div className="space-y-1.5">
            {insights.map((item, idx) => {
              const meta = getTypeMeta(item.type);
              const isSelected = idx === selectedIndex;
              return (
                <button
                  key={item.id || idx}
                  type="button"
                  onClick={() => setSelectedIndex(idx)}
                  className={`w-full text-left p-3 rounded-md border transition flex flex-col space-y-1.5 ${
                    isSelected
                      ? "bg-surface border-editorial-accent shadow-sm ring-1 ring-editorial-accent/20"
                      : "bg-surface/60 border-editorial-border/70 hover:bg-surface hover:border-editorial-border"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span
                      className={`inline-flex items-center space-x-1 text-[10px] px-1.5 py-0.5 rounded border font-medium ${meta.colorClass}`}
                    >
                      {meta.icon}
                      <span>{meta.label}</span>
                    </span>
                    <span className="text-[10px] font-mono text-charcoal-subtle">
                      Rank #{idx + 1}
                    </span>
                  </div>
                  <h4 className="text-xs font-serif font-medium text-charcoal leading-snug line-clamp-2">
                    {item.headline}
                  </h4>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right Column: Active Insight Detail & Visuals */}
        <div className="lg:col-span-7 p-5 space-y-4 flex flex-col justify-between">
          <div className="space-y-4">
            {/* Headline and Tag */}
            <div>
              <div className="flex items-center space-x-2 mb-1.5">
                <span
                  className={`inline-flex items-center space-x-1.5 text-[11px] px-2 py-0.5 rounded border font-medium ${activeMeta.colorClass}`}
                >
                  {activeMeta.icon}
                  <span>{activeMeta.label}</span>
                </span>
                <span className="text-[11px] text-charcoal-subtle font-mono">
                  Rank #{selectedIndex + 1} of {insights.length}
                </span>
              </div>
              <h3 className="text-base font-serif font-semibold text-charcoal leading-snug">
                {activeInsight.headline}
              </h3>
            </div>

            {/* Explanation */}
            <p className="text-xs text-charcoal/90 leading-relaxed font-sans">
              {activeInsight.explanation}
            </p>

            {/* Why It Matters Callout */}
            <div className="p-3 rounded border border-editorial-border/80 bg-cream/30 flex items-start space-x-2 text-xs">
              <Lightbulb className="w-4 h-4 text-editorial-accent shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold text-charcoal block mb-0.5 text-[11px] uppercase tracking-wider">
                  Why This Matters
                </span>
                <p className="text-charcoal-muted leading-relaxed text-[11px]">
                  {activeInsight.why_it_matters}
                </p>
              </div>
            </div>

            {/* Structured Evidence Card */}
            {activeInsight.evidence && (
              <div className="p-3 rounded border border-editorial-border/60 bg-surface text-xs space-y-2">
                <span className="font-semibold text-charcoal-subtle block text-[10px] uppercase tracking-wider flex items-center space-x-1">
                  <Layers className="w-3 h-3 text-editorial-accent" />
                  <span>Observed Empirical Evidence</span>
                </span>

                <div className="flex flex-wrap gap-2 pt-1">
                  {Object.entries(activeInsight.evidence.key_figures || {}).map(([k, v]) => (
                    <div
                      key={k}
                      className="px-2.5 py-1.5 rounded bg-cream/60 border border-editorial-border/70 text-[11px]"
                    >
                      <span className="text-charcoal-subtle block text-[9px] uppercase tracking-wider">
                        {k.replace(/_/g, " ")}
                      </span>
                      <span className="font-mono font-medium text-charcoal">
                        {typeof v === "number" ? v.toLocaleString() : String(v)}
                      </span>
                    </div>
                  ))}
                  {activeInsight.evidence.sample_size !== undefined && (
                    <div className="px-2.5 py-1.5 rounded bg-cream/60 border border-editorial-border/70 text-[11px]">
                      <span className="text-charcoal-subtle block text-[9px] uppercase tracking-wider">
                        Sample Size
                      </span>
                      <span className="font-mono font-medium text-charcoal">
                        N={activeInsight.evidence.sample_size}
                      </span>
                    </div>
                  )}
                </div>

                {activeInsight.evidence.comparison_basis && (
                  <p className="text-[10px] text-charcoal-subtle italic pt-1">
                    Baseline: {activeInsight.evidence.comparison_basis}
                  </p>
                )}
              </div>
            )}

            {/* Embedded Chart if present */}
            {activeInsight.visualization && (
              <div className="pt-2">
                <ResultVisualization visualization={activeInsight.visualization} />
              </div>
            )}

            {/* Uncertainty or Limitations Disclaimer */}
            {activeInsight.uncertainty_or_limitations && (
              <div className="text-[11px] text-charcoal-subtle italic bg-stone-50 border border-stone-200 p-2.5 rounded flex items-start space-x-1.5">
                <Info className="w-3.5 h-3.5 text-stone-500 shrink-0 mt-0.5" />
                <span>{activeInsight.uncertainty_or_limitations}</span>
              </div>
            )}
          </div>

          {/* Action Button: Run Analytical Query */}
          {activeInsight.suggested_query && onSelectQuery && (
            <div className="pt-3 border-t border-editorial-border/60 flex justify-end">
              <button
                type="button"
                onClick={() => onSelectQuery(activeInsight.suggested_query!)}
                disabled={isLoading}
                className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded bg-cream hover:bg-editorial-hover border border-editorial-border hover:border-editorial-accent text-charcoal hover:text-editorial-accent text-xs font-medium transition disabled:opacity-50"
              >
                <span>Investigate Further: &ldquo;{activeInsight.suggested_query}&rdquo;</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
