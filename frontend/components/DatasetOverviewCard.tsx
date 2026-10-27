"use client";

import React, { useState, useEffect } from "react";
import { DatasetMetadata, DatasetOverview } from "@/types";
import { getDatasetOverview } from "@/lib/api";
import {
  Compass,
  Sparkles,
  ChevronDown,
  ChevronUp,
  TrendingUp,
  Target,
  AlertCircle,
  HelpCircle,
  CheckCircle2,
  Calendar,
  Layers,
  ArrowRight,
  ShieldAlert,
  Loader2,
  RefreshCw,
  Info,
} from "lucide-react";

interface DatasetOverviewCardProps {
  dataset: DatasetMetadata;
  onSelectQuery?: (query: string) => void;
  isLoading?: boolean;
}

export const DatasetOverviewCard: React.FC<DatasetOverviewCardProps> = ({
  dataset,
  onSelectQuery,
  isLoading = false,
}) => {
  const [overview, setOverview] = useState<DatasetOverview | null>(
    dataset.overview || null
  );
  const [isFetching, setIsFetching] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState<boolean>(true);

  // Synchronize when dataset changes or when overview is provided
  useEffect(() => {
    let isCancelled = false;

    if (dataset.overview) {
      setOverview(dataset.overview);
      setError(null);
      setIsFetching(false);
      return;
    }

    // If overview is not in metadata, fetch automatically
    setIsFetching(true);
    setError(null);

    getDatasetOverview(dataset.dataset_id)
      .then((data) => {
        if (!isCancelled) {
          setOverview(data);
          setError(null);
        }
      })
      .catch((err) => {
        if (!isCancelled) {
          setError(err.message || "Failed to load dataset overview.");
        }
      })
      .finally(() => {
        if (!isCancelled) {
          setIsFetching(false);
        }
      });

    return () => {
      isCancelled = true;
    };
  }, [dataset.dataset_id, dataset.overview]);

  const handleRetry = () => {
    setIsFetching(true);
    setError(null);
    getDatasetOverview(dataset.dataset_id)
      .then((data) => {
        setOverview(data);
        setError(null);
      })
      .catch((err) => {
        setError(err.message || "Failed to load dataset overview.");
      })
      .finally(() => {
        setIsFetching(false);
      });
  };

  // Loading skeleton state
  if (isFetching && !overview) {
    return (
      <div className="mb-6 rounded-lg border border-editorial-border bg-surface p-6 shadow-sm animate-pulse">
        <div className="flex items-center justify-between pb-3 border-b border-editorial-border/60">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded bg-cream/70 flex items-center justify-center">
              <Loader2 className="w-4 h-4 animate-spin text-editorial-accent" />
            </div>
            <div className="space-y-1.5">
              <div className="h-4 w-48 bg-stone-200/80 rounded" />
              <div className="h-3 w-28 bg-stone-200/50 rounded" />
            </div>
          </div>
        </div>
        <div className="mt-5 space-y-3">
          <div className="h-4 w-full bg-stone-200/70 rounded" />
          <div className="h-4 w-5/6 bg-stone-200/60 rounded" />
          <div className="h-4 w-2/3 bg-stone-200/50 rounded" />
        </div>
        <div className="mt-6 grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-14 bg-cream/50 rounded border border-editorial-border/40" />
          ))}
        </div>
      </div>
    );
  }

  // Error state with retry
  if (error && !overview) {
    return (
      <div className="mb-6 rounded-lg border border-red-200 bg-red-50/60 p-5 text-red-900 shadow-sm">
        <div className="flex items-start justify-between">
          <div className="flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <h3 className="text-sm font-semibold">Unable to Generate Dataset Overview</h3>
              <p className="text-xs text-red-800/90 leading-relaxed">{error}</p>
            </div>
          </div>
          <button
            type="button"
            onClick={handleRetry}
            className="px-3 py-1.5 rounded border border-red-300 bg-surface hover:bg-red-50 text-xs font-medium text-red-800 transition flex items-center space-x-1.5 shrink-0"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Retry</span>
          </button>
        </div>
      </div>
    );
  }

  if (!overview) {
    return null;
  }

  const isSynthesized = overview.status === "synthesized";

  const getConfidenceBadge = (confidence: string) => {
    switch (confidence.toLowerCase()) {
      case "high":
        return (
          <span className="text-[10px] font-normal px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-200 flex items-center space-x-1">
            <CheckCircle2 className="w-2.5 h-2.5" />
            <span>High Confidence Domain</span>
          </span>
        );
      case "medium":
        return (
          <span className="text-[10px] font-normal px-2 py-0.5 rounded-full bg-blue-100 text-blue-800 border border-blue-200 flex items-center space-x-1">
            <Info className="w-2.5 h-2.5" />
            <span>Inferred Domain</span>
          </span>
        );
      default:
        return (
          <span className="text-[10px] font-normal px-2 py-0.5 rounded-full bg-stone-200/80 text-stone-700 border border-stone-300 flex items-center space-x-1">
            <HelpCircle className="w-2.5 h-2.5" />
            <span>Unspecified Domain</span>
          </span>
        );
    }
  };

  const getFeasibilityBadge = (feasibility: string) => {
    const fLower = feasibility.toLowerCase();
    if (fLower.includes("exploration")) {
      return (
        <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-200">
          Feasible for exploration
        </span>
      );
    } else if (fLower.includes("caveats")) {
      return (
        <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-blue-100 text-blue-800 border border-blue-200">
          Feasible with caveats
        </span>
      );
    } else if (fLower.includes("additional")) {
      return (
        <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 border border-amber-200">
          Requires additional data
        </span>
      );
    }
    return (
      <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-stone-200 text-stone-700 border border-stone-300">
        Not currently supported by the data
      </span>
    );
  };

  const getRoleBadge = (role?: string | null) => {
    if (!role) return null;
    const r = role.toLowerCase();
    let bg = "bg-stone-100 text-stone-600 border-stone-200";
    if (r === "metric") bg = "bg-emerald-50 text-emerald-700 border-emerald-200";
    if (r === "dimension") bg = "bg-blue-50 text-blue-700 border-blue-200";
    if (r === "temporal") bg = "bg-amber-50 text-amber-700 border-amber-200";
    if (r === "identifier") bg = "bg-purple-50 text-purple-700 border-purple-200";

    return (
      <span className={`text-[9px] uppercase tracking-wider font-mono px-1.5 py-0.5 rounded border ${bg}`}>
        {role}
      </span>
    );
  };

  return (
    <div className="mb-6 rounded-lg border border-editorial-border bg-surface shadow-sm overflow-hidden transition-all duration-200">
      {/* Top Header Bar */}
      <div className="px-5 py-3.5 bg-cream/70 border-b border-editorial-border flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-1.5 rounded bg-editorial-accent/10 text-editorial-accent">
            <Compass className="w-4 h-4" />
          </div>
          <div className="flex items-center flex-wrap gap-2">
            <h2 className="text-sm font-serif font-semibold text-charcoal tracking-tight">
              {overview.domain}
            </h2>
            {getConfidenceBadge(overview.domain_confidence)}
            {isSynthesized ? (
              <span className="text-[10px] font-normal px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 flex items-center space-x-1">
                <Sparkles className="w-2.5 h-2.5 text-editorial-accent" />
                <span>AI Grounded Overview</span>
              </span>
            ) : (
              <span className="text-[10px] font-normal px-2 py-0.5 rounded-full bg-stone-100 text-stone-600 border border-stone-200">
                Deterministic Profile
              </span>
            )}
          </div>
        </div>

        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-xs text-charcoal-muted hover:text-charcoal flex items-center space-x-1 px-2.5 py-1 rounded hover:bg-stone-200/50 transition shrink-0"
          aria-label={isExpanded ? "Collapse overview details" : "Expand overview details"}
        >
          <span className="text-[11px] font-medium">{isExpanded ? "Hide Details" : "Show Details"}</span>
          {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>
      </div>

      <div className="p-5 sm:p-6 space-y-6">
        {/* Usability Blocker Banner for Entirely Empty Datasets */}
        {dataset.profile?.missing_cell_pct === 100 && (
          <div className="p-4 rounded-lg border border-amber-300 bg-amber-50/90 text-amber-900 text-xs flex items-start space-x-3">
            <ShieldAlert className="w-5 h-5 text-amber-700 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-semibold block text-amber-950 text-xs">
                Zero Usable Observations (100% Missing Cells)
              </span>
              <p className="text-[11px] text-amber-900/90 leading-relaxed font-sans">
                This dataset defines {dataset.num_cols} column headers, but all {dataset.num_rows.toLocaleString()} rows contain zero recorded values ({dataset.total_missing.toLocaleString()} missing cells). Statistical aggregations, group comparisons, and machine-learning predictions are disabled until populated data records are supplied.
              </p>
            </div>
          </div>
        )}

        {/* Central Summary Paragraph: Central to user comprehension */}
        <div className="p-4 sm:p-5 rounded-lg bg-cream/40 border border-editorial-border/80">
          <span className="text-[10px] uppercase tracking-wider text-charcoal-subtle font-semibold block mb-1.5">
            Dataset Summary & Scope
          </span>
          <p className="text-sm sm:text-base font-serif text-charcoal leading-relaxed">
            {overview.summary_paragraph}
          </p>

          <div className="mt-3 pt-2.5 border-t border-editorial-border/60 flex items-start space-x-2 text-xs text-charcoal-muted">
            <span className="font-semibold text-charcoal shrink-0 text-[11px] uppercase tracking-wider">
              Row Representation:
            </span>
            <span className="text-stone-700 text-[11px] leading-snug">
              {overview.row_meaning}
            </span>
          </div>
        </div>

        {/* Factual Metrics & Coverage Bar */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          <div className="p-2.5 rounded border border-editorial-border/70 bg-cream/20">
            <span className="text-[10px] uppercase tracking-wider text-charcoal-subtle font-medium block">
              Records
            </span>
            <span className="text-sm font-semibold text-charcoal font-mono">
              {dataset.num_rows.toLocaleString()}
            </span>
          </div>

          <div className="p-2.5 rounded border border-editorial-border/70 bg-cream/20">
            <span className="text-[10px] uppercase tracking-wider text-charcoal-subtle font-medium block">
              Columns
            </span>
            <span className="text-sm font-semibold text-charcoal font-mono">
              {dataset.num_cols} attributes
            </span>
          </div>

          <div className="p-2.5 rounded border border-editorial-border/70 bg-cream/20">
            <span className="text-[10px] uppercase tracking-wider text-charcoal-subtle font-medium block">
              Data Completeness
            </span>
            <span
              className={`text-sm font-semibold font-mono ${
                dataset.total_missing > 0 ? "text-amber-700" : "text-emerald-700"
              }`}
            >
              {dataset.total_missing === 0
                ? "100% complete"
                : `${dataset.total_missing.toLocaleString()} missing (${dataset.profile?.missing_cell_pct ?? 0}%)`}
            </span>
          </div>

          <div className="p-2.5 rounded border border-editorial-border/70 bg-cream/20">
            <span className="text-[10px] uppercase tracking-wider text-charcoal-subtle font-medium block">
              Temporal Scope
            </span>
            <span className="text-xs font-medium text-charcoal truncate block" title={overview.size_and_coverage}>
              {overview.size_and_coverage.split(";")[1] || "Cross-sectional snapshot"}
            </span>
          </div>
        </div>

        {/* Collapsible Deep-Dive Sections */}
        {isExpanded && (
          <>
            {/* Key Fields & Meanings */}
            {overview.key_fields && overview.key_fields.length > 0 && (
              <div>
                <div className="flex items-center space-x-2 mb-2.5">
                  <Layers className="w-3.5 h-3.5 text-editorial-accent" />
                  <span className="text-[11px] uppercase tracking-wider text-charcoal-subtle font-semibold">
                    Important Columns & Meanings
                  </span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
                  {overview.key_fields.map((field, idx) => (
                    <div
                      key={idx}
                      className="p-3 rounded border border-editorial-border/70 bg-surface flex flex-col justify-between space-y-1 text-xs"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-xs font-semibold text-editorial-accent">
                          {field.name}
                        </span>
                        {getRoleBadge(field.role)}
                      </div>
                      <p className="text-[11px] text-charcoal-muted leading-snug">
                        {field.meaning}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Section: What You Can Learn */}
            {overview.what_we_can_learn && overview.what_we_can_learn.length > 0 && (
              <div>
                <div className="flex items-center space-x-2 mb-2.5">
                  <TrendingUp className="w-3.5 h-3.5 text-editorial-accent" />
                  <span className="text-[11px] uppercase tracking-wider text-charcoal-subtle font-semibold">
                    What You Can Learn From This Data
                  </span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {overview.what_we_can_learn.map((item, idx) => (
                    <div
                      key={idx}
                      className="p-3.5 rounded border border-editorial-border/80 bg-cream/20 flex flex-col justify-between space-y-2 text-xs"
                    >
                      <div className="space-y-1">
                        <h4 className="font-semibold text-charcoal text-xs">
                          {item.title}
                        </h4>
                        <p className="text-[11px] text-charcoal-muted leading-relaxed">
                          {item.description}
                        </p>
                      </div>

                      <div className="pt-2 border-t border-editorial-border/50 space-y-2">
                        <div className="flex items-start space-x-1.5 text-[10px] text-stone-600">
                          <span className="font-semibold text-charcoal shrink-0">Business Value:</span>
                          <span className="leading-snug">{item.business_value}</span>
                        </div>

                        {item.sample_question && onSelectQuery && (
                          <button
                            type="button"
                            onClick={() => onSelectQuery(item.sample_question!)}
                            disabled={isLoading}
                            className="group w-full text-left text-[11px] text-editorial-accent hover:text-[#153427] font-medium flex items-center justify-between px-2 py-1 rounded bg-surface hover:bg-editorial-hover border border-editorial-border transition"
                          >
                            <span className="truncate">Ask: &ldquo;{item.sample_question}&rdquo;</span>
                            <ArrowRight className="w-3 h-3 text-editorial-accent group-hover:translate-x-0.5 transition shrink-0 ml-1" />
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Section: Potential Predictions & Limitations */}
            {overview.potential_predictions && overview.potential_predictions.length > 0 && (
              <div>
                <div className="flex items-center space-x-2 mb-2.5">
                  <Target className="w-3.5 h-3.5 text-editorial-accent" />
                  <span className="text-[11px] uppercase tracking-wider text-charcoal-subtle font-semibold">
                    Potential Predictive Tasks & Prerequisites
                  </span>
                </div>

                {/* Analytical Boundary Disclaimer */}
                <div className="mb-3 p-3 rounded border border-amber-200/80 bg-amber-50/50 text-[11px] text-amber-900/90 flex items-start space-x-2">
                  <ShieldAlert className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
                  <p className="leading-relaxed">
                    <strong className="font-semibold text-amber-950">Exploratory vs. Predictive Boundary: </strong>
                    {overview.prediction_limitations}
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {overview.potential_predictions.map((pred, idx) => (
                    <div
                      key={idx}
                      className="p-3.5 rounded border border-editorial-border/70 bg-surface flex flex-col justify-between space-y-2 text-xs"
                    >
                      <div className="space-y-1.5">
                        <div className="flex items-start justify-between gap-1.5">
                          <h4 className="font-semibold text-charcoal text-xs leading-snug">
                            {pred.task}
                          </h4>
                        </div>
                        <div>{getFeasibilityBadge(pred.feasibility)}</div>
                        {pred.target_variable && (
                          <div className="text-[10px] text-stone-500 font-mono">
                            Target: <span className="font-medium text-charcoal">{pred.target_variable}</span>
                          </div>
                        )}
                        <p className="text-[11px] text-stone-600 leading-relaxed pt-1">
                          {pred.prerequisites}
                        </p>
                      </div>

                      {pred.missing_elements && pred.missing_elements.length > 0 && (
                        <div className="pt-2 border-t border-editorial-border/40">
                          <span className="text-[10px] font-semibold text-charcoal-subtle uppercase tracking-wider block mb-1">
                            Missing Prerequisites:
                          </span>
                          <div className="flex flex-wrap gap-1">
                            {pred.missing_elements.map((m, mIdx) => (
                              <span
                                key={mIdx}
                                className="text-[9px] px-1.5 py-0.5 rounded bg-stone-100 text-stone-600 border border-stone-200"
                              >
                                {m}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};
