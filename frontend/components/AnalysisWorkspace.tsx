"use client";

import React, { useState } from "react";
import { DatasetMetadata, AnalyzeResponse, HistoryItem } from "@/types";
import { analyzeQuery } from "@/lib/api";
import { ResultVisualization } from "./ResultVisualization";
import { CodeAuditPanel } from "./CodeAuditPanel";
import { QueryHistory } from "./QueryHistory";
import { DatasetUnderstandingCard } from "./DatasetUnderstandingCard";
import { AutomaticInsights } from "./AutomaticInsights";
import { AnalyticalExplanationCard } from "./AnalyticalExplanationCard";
import { Send, Sparkles, AlertCircle, Loader2, Table2 } from "lucide-react";


interface AnalysisWorkspaceProps {
  dataset: DatasetMetadata;
}

export const AnalysisWorkspace: React.FC<AnalysisWorkspaceProps> = ({ dataset }) => {
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [currentResult, setCurrentResult] = useState<AnalyzeResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isQuotaExceeded, setIsQuotaExceeded] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [activeHistoryId, setActiveHistoryId] = useState<string | undefined>();

  const handleAnalyze = async (queryText?: string) => {
    const q = (queryText || question).trim();
    if (!q) return;

    setErrorMessage(null);
    setIsQuotaExceeded(false);
    setIsLoading(true);

    try {
      const response = await analyzeQuery(dataset.dataset_id, q);

      if (response.execution_status === "failed" || response.validation_status === "failed") {
        setErrorMessage(response.error_message || response.validation_message || "Analysis could not be completed.");
      }

      setCurrentResult(response);

      // Add to session history
      const historyEntry: HistoryItem = {
        id: Math.random().toString(36).substring(2, 9),
        question: q,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        response,
      };

      setHistory((prev) => [historyEntry, ...prev.filter((h) => h.question !== q)]);
      setActiveHistoryId(historyEntry.id);
    } catch (err: any) {
      if (err.errorType === "quota_exceeded" || err.isQuota || err.message?.includes("usage limit") || err.message?.includes("quota")) {
        setIsQuotaExceeded(true);
      } else {
        setErrorMessage(err.message || "Failed to communicate with analysis engine.");
      }
      setCurrentResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectHistory = (item: HistoryItem) => {
    setQuestion(item.question);
    setCurrentResult(item.response);
    setActiveHistoryId(item.id);
    setIsQuotaExceeded(false);
    setErrorMessage(item.response.error_message || null);
  };

  const quickSuggestions = dataset.understanding?.suggested_questions || [
    "Which region generated the most revenue?",
    "What are the top 5 products by revenue?",
    "What is the average customer rating?",
  ];

  return (
    <div className="flex-1 flex flex-col p-6 sm:p-10 max-w-5xl mx-auto w-full overflow-y-auto">
      {/* Header */}
      <div className="mb-6">
        <span className="text-xs uppercase tracking-wider text-charcoal-subtle font-medium block mb-1">
          Analysis Workspace
        </span>
        <h1 className="text-2xl sm:text-3xl font-serif text-charcoal tracking-tight">
          Explore {dataset.filename.replace(" (Sample)", "")}
        </h1>
      </div>

      {/* Dataset Understanding Card */}
      {dataset.understanding && (
        <DatasetUnderstandingCard
          dataset={dataset}
          onSelectQuestion={(q) => {
            setQuestion(q);
            handleAnalyze(q);
          }}
          isLoading={isLoading}
        />
      )}

      {/* Automatic Insights Discovery Section */}
      {dataset.insights && dataset.insights.length > 0 && (
        <AutomaticInsights
          insights={dataset.insights}
          onSelectQuery={(q) => {
            setQuestion(q);
            handleAnalyze(q);
          }}
          isLoading={isLoading}
        />
      )}


      {/* Query Input Card */}
      <div className="border border-editorial-border rounded-lg bg-surface p-4 shadow-sm">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleAnalyze();
          }}
          className="flex flex-col sm:flex-row gap-2.5 items-stretch"
        >
          <div className="relative flex-1">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask anything about your data... (e.g. Which region generated the most revenue?)"
              disabled={isLoading}
              className="w-full px-4 py-3 text-sm text-charcoal bg-cream/40 border border-editorial-border rounded focus:outline-none focus:border-editorial-accent focus:ring-1 focus:ring-editorial-accent transition font-sans"
            />
          </div>

          <button
            type="submit"
            disabled={isLoading || !question.trim()}
            className="px-6 py-3 bg-editorial-accent hover:bg-[#153427] text-white text-xs font-medium tracking-wide uppercase rounded transition flex items-center justify-center space-x-2 disabled:opacity-50 shrink-0"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Analyzing...</span>
              </>
            ) : (
              <>
                <span>Analyze</span>
                <Send className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </form>

        {/* Quick Suggestion Chips */}
        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
          <span className="text-charcoal-subtle font-medium text-[11px] flex items-center space-x-1">
            <Sparkles className="w-3 h-3 text-editorial-accent" />
            <span>Suggestions:</span>
          </span>
          {quickSuggestions.slice(0, 3).map((prompt, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => {
                setQuestion(prompt);
                handleAnalyze(prompt);
              }}
              disabled={isLoading}
              className="text-[11px] text-charcoal-muted hover:text-charcoal hover:bg-editorial-hover px-2.5 py-1 rounded border border-editorial-border/80 transition"
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>

      {/* Honest Loading Indicator */}
      {isLoading && (
        <div className="mt-8 p-8 border border-editorial-border rounded-lg bg-surface flex flex-col items-center justify-center space-y-3">
          <Loader2 className="w-6 h-6 animate-spin text-editorial-accent" />
          <p className="text-sm font-medium text-charcoal">Analyzing your dataset…</p>
          <p className="text-xs text-charcoal-muted font-light">
            Generating Pandas query, verifying AST safety rules, and executing in sandbox.
          </p>
        </div>
      )}

      {/* Polished Quota Error State */}
      {isQuotaExceeded && !isLoading && (
        <div className="mt-6 p-5 rounded-lg border border-amber-200/90 bg-[#FFFDF5] text-stone-800 shadow-sm">
          <div className="flex items-start space-x-3.5">
            <div className="p-2 rounded-md bg-amber-100/80 text-amber-800 shrink-0 mt-0.5">
              <AlertCircle className="w-5 h-5 text-amber-700" />
            </div>
            <div className="space-y-1">
              <h3 className="font-serif text-base font-medium text-stone-900">
                Analysis temporarily unavailable
              </h3>
              <p className="text-xs text-stone-600 leading-relaxed font-sans">
                AI analysis has reached its current usage limit. Please try again later.
              </p>
              <div className="mt-2.5 pt-2 border-t border-amber-200/60 flex items-center space-x-2 text-[11px] text-stone-500">
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-500"></span>
                <span>Gemini API Free Tier daily quota allocation reached (20 requests/day). Rate limits reset automatically.</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Standard Error Message Display (non-quota) */}
      {errorMessage && !isQuotaExceeded && !isLoading && (
        <div className="mt-6 p-4 rounded-lg border border-red-200 bg-red-50/70 text-red-800 text-xs">
          <div className="flex items-start space-x-2.5">
            <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-semibold block">Analysis Issue Encountered</span>
              <p className="leading-relaxed">{errorMessage}</p>
            </div>
          </div>
        </div>
      )}

      {/* Dominant Answer Area */}
      {currentResult && !isLoading && currentResult.execution_status === "passed" && (
        <div className="mt-6 border border-editorial-border rounded-lg bg-surface p-6">
          <div className="flex items-center justify-between pb-3 mb-4 border-b border-editorial-border">
            <span className="text-[11px] uppercase tracking-wider text-charcoal-subtle font-semibold">
              Analysis Result
            </span>
            <span className="text-[11px] text-charcoal-muted font-mono">
              {currentResult.execution_time_ms}ms execution
            </span>
          </div>

          {/* Prominent Answer Card */}
          <div className="text-2xl sm:text-3xl font-serif text-charcoal font-medium leading-snug">
            {currentResult.formatted_result}
          </div>

          {/* Analytical Significance & Empirical Evidence (Feature A) */}
          {currentResult.explanation && (
            <AnalyticalExplanationCard
              explanation={currentResult.explanation}
              onSelectNextStep={(nextQuestion) => {
                setQuestion(nextQuestion);
                window.scrollTo({ top: 0, behavior: "smooth" });
              }}
            />
          )}

          {/* Tabular Output (if DataFrame or Series) */}
          {currentResult.table_data && currentResult.table_data.rows.length > 0 && (
            <div className="mt-6 pt-5 border-t border-editorial-border">
              <div className="flex items-center space-x-2 mb-3">
                <Table2 className="w-4 h-4 text-charcoal-muted" />
                <span className="text-xs font-semibold uppercase tracking-wider text-charcoal-subtle">
                  Tabular Breakdown
                </span>
              </div>
              <div className="overflow-x-auto border border-editorial-border rounded max-h-64">
                <table className="w-full text-xs text-left">
                  <thead className="bg-cream border-b border-editorial-border text-charcoal font-semibold">
                    <tr>
                      {currentResult.table_data.columns.map((c, i) => (
                        <th key={i} className="py-2 px-3 font-mono text-[11px]">
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-editorial-border/60 font-mono text-[11px]">
                    {currentResult.table_data.rows.slice(0, 15).map((row, rIdx) => (
                      <tr key={rIdx} className="hover:bg-stone-50/70">
                        {currentResult.table_data?.columns.map((c, cIdx) => (
                          <td key={cIdx} className="py-1.5 px-3 text-charcoal-muted">
                            {row[c] !== null && row[c] !== undefined ? String(row[c]) : "-"}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Structured Recharts Visualization */}
          {currentResult.visualization && (
            <ResultVisualization visualization={currentResult.visualization} />
          )}

          {/* Collapsible Code & Security Audit Panel */}
          <CodeAuditPanel
            code={currentResult.code}
            validationStatus={currentResult.validation_status}
            executionStatus={currentResult.execution_status}
            validationMessage={currentResult.validation_message}
            executionTimeMs={currentResult.execution_time_ms}
          />
        </div>
      )}

      {/* Query History */}
      <QueryHistory
        history={history}
        activeId={activeHistoryId}
        onSelect={handleSelectHistory}
      />
    </div>
  );
};
