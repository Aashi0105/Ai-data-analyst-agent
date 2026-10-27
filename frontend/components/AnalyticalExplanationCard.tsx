"use client";

import React from "react";
import { AnswerExplanation } from "@/types";
import {
  Lightbulb,
  Compass,
  Info,
  ArrowRight,
  TrendingUp,
  BarChart2,
  PieChart,
  Hash,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

interface AnalyticalExplanationCardProps {
  explanation?: AnswerExplanation | null;
  onSelectNextStep?: (question: string) => void;
}

export const AnalyticalExplanationCard: React.FC<AnalyticalExplanationCardProps> = ({
  explanation,
  onSelectNextStep,
}) => {
  if (!explanation) return null;

  const { finding, evidence, interpretation, uncertainty_or_limitations, next_steps, status } = explanation;

  // Format numbers nicely
  const formatNum = (num?: number | null) => {
    if (num === null || num === undefined) return null;
    if (Math.abs(num) >= 1_000_000) return `${(num / 1_000_000).toFixed(2)}M`;
    if (Math.abs(num) >= 1_000) return `${(num / 1_000).toFixed(1)}k`;
    return num.toLocaleString(undefined, { maximumFractionDigits: 2 });
  };

  return (
    <div className="mt-5 pt-5 border-t border-editorial-border/80 space-y-4">
      {/* Header Eyebrow & Status Badge */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Sparkles className="w-3.5 h-3.5 text-editorial-accent" />
          <span className="text-[11px] uppercase tracking-wider text-charcoal-subtle font-semibold">
            Analytical Significance & Evidence
          </span>
        </div>
        <span
          className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-mono border ${
            status === "synthesized"
              ? "bg-emerald-50 text-emerald-800 border-emerald-200"
              : "bg-stone-50 text-stone-700 border-stone-200"
          }`}
          title={
            status === "synthesized"
              ? "Synthesized by AI from verified numerical evidence"
              : "Formulated deterministically from verified numerical evidence"
          }
        >
          <ShieldCheck className="w-3 h-3" />
          <span>{status === "synthesized" ? "AI Grounded" : "Deterministic Facts"}</span>
        </span>
      </div>

      {/* Primary Finding */}
      <div className="p-4 rounded-lg bg-surface border border-editorial-border/70 shadow-sm">
        <span className="text-[10px] font-mono uppercase tracking-wider text-charcoal-muted block mb-1">
          Observed Finding
        </span>
        <p className="text-sm text-charcoal font-medium leading-relaxed">
          {finding}
        </p>
      </div>

      {/* Verified Empirical Evidence Badges Strip */}
      <div className="space-y-1.5">
        <span className="text-[10px] font-mono uppercase tracking-wider text-charcoal-muted block">
          Verified Empirical Evidence
        </span>
        <div className="flex flex-wrap gap-2">
          {evidence.metric_name && (
            <div className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded bg-cream/70 border border-editorial-border/70 text-[11px] text-charcoal">
              <Hash className="w-3 h-3 text-charcoal-muted" />
              <span className="text-charcoal-muted">Metric:</span>
              <span className="font-medium font-mono">{evidence.metric_name}</span>
            </div>
          )}

          {evidence.is_additive && evidence.total_value !== null && evidence.total_value !== undefined && (
            <div className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded bg-cream/70 border border-editorial-border/70 text-[11px] text-charcoal">
              <PieChart className="w-3 h-3 text-charcoal-muted" />
              <span className="text-charcoal-muted">Total Volume:</span>
              <span className="font-semibold font-mono">{formatNum(evidence.total_value)}</span>
            </div>
          )}

          {evidence.top_contributor && (
            <div className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded bg-emerald-50/80 border border-emerald-200/80 text-[11px] text-emerald-900">
              <BarChart2 className="w-3 h-3 text-emerald-700" />
              <span className="text-emerald-700">Top Contributor:</span>
              <span className="font-medium">{evidence.top_contributor}</span>
              {evidence.top_value !== null && evidence.top_value !== undefined && (
                <span className="font-mono text-[10px] text-emerald-800">
                  ({formatNum(evidence.top_value)})
                </span>
              )}
            </div>
          )}

          {evidence.top_share_pct !== null && evidence.top_share_pct !== undefined && (
            <div className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded bg-emerald-50/80 border border-emerald-200/80 text-[11px] text-emerald-900">
              <span className="text-emerald-700">Share of Total:</span>
              <span className="font-bold font-mono">{evidence.top_share_pct.toFixed(1)}%</span>
            </div>
          )}

          {evidence.disparity_ratio !== null && evidence.disparity_ratio !== undefined && (
            <div className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded bg-blue-50/80 border border-blue-200/80 text-[11px] text-blue-900">
              <TrendingUp className="w-3 h-3 text-blue-700" />
              <span className="text-blue-700">Disparity Ratio:</span>
              <span className="font-bold font-mono">{evidence.disparity_ratio.toFixed(1)}x</span>
              {evidence.runner_up && (
                <span className="text-[10px] text-blue-800">vs {evidence.runner_up}</span>
              )}
            </div>
          )}

          {evidence.sample_size !== null && evidence.sample_size !== undefined && (
            <div className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded bg-stone-50 border border-stone-200 text-[11px] text-stone-700">
              <span className="text-stone-500">Dataset Scope:</span>
              <span className="font-mono">{evidence.sample_size.toLocaleString()} rows</span>
            </div>
          )}

          {evidence.baseline_value !== null && evidence.baseline_value !== undefined && (
            <div className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded bg-stone-50 border border-stone-200 text-[11px] text-stone-700">
              <span className="text-stone-500">
                {evidence.baseline_description ? `${evidence.baseline_description}:` : "Baseline:"}
              </span>
              <span className="font-mono">{formatNum(evidence.baseline_value)}</span>
            </div>
          )}
        </div>
      </div>

      {/* Analytical Interpretation (Demarcated Hypothesis) */}
      <div className="p-3.5 rounded-lg bg-amber-50/50 border border-amber-200/70 text-xs">
        <div className="flex items-center space-x-2 mb-1.5">
          <Lightbulb className="w-3.5 h-3.5 text-amber-700 shrink-0" />
          <span className="font-semibold text-amber-900 uppercase tracking-wider text-[10px]">
            Analytical Interpretation (Hypothesis)
          </span>
        </div>
        <p className="text-amber-950/90 leading-relaxed font-sans pl-5">
          {interpretation}
        </p>
      </div>

      {/* Epistemic Modesty & Limitations */}
      {uncertainty_or_limitations && (
        <div className="p-3 rounded-md bg-stone-50 border border-stone-200/80 text-[11px] text-stone-600 flex items-start space-x-2">
          <Info className="w-3.5 h-3.5 text-stone-500 shrink-0 mt-0.5" />
          <div className="leading-relaxed">
            <span className="font-semibold text-stone-700">Methodology & Limitations: </span>
            <span>{uncertainty_or_limitations}</span>
          </div>
        </div>
      )}

      {/* Suggested Follow-up Investigations */}
      {next_steps && next_steps.length > 0 && (
        <div className="pt-2">
          <div className="flex items-center space-x-1.5 mb-2 text-charcoal-subtle">
            <Compass className="w-3.5 h-3.5 text-editorial-accent" />
            <span className="text-[11px] font-semibold uppercase tracking-wider">
              Suggested Next Investigations
            </span>
          </div>
          <div className="flex flex-col sm:flex-row sm:flex-wrap gap-2">
            {next_steps.map((step, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => onSelectNextStep && onSelectNextStep(step)}
                className="group flex items-center justify-between sm:justify-start space-x-2 text-left text-xs text-charcoal bg-surface hover:bg-cream/70 border border-editorial-border hover:border-editorial-accent/60 rounded px-3 py-2 transition shadow-xs"
                title="Click to populate query input"
              >
                <span className="group-hover:text-editorial-accent transition leading-snug">
                  {step}
                </span>
                <ArrowRight className="w-3 h-3 text-charcoal-muted group-hover:text-editorial-accent group-hover:translate-x-0.5 transition shrink-0 ml-1" />
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
