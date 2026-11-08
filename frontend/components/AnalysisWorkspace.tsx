"use client";

import React, { useState, useEffect } from "react";
import { DatasetMetadata, AnalyzeResponse, HistoryItem, SavedExample } from "@/types";
import { analyzeQuery, getExamples, deleteDataset } from "@/lib/api";
import { ResultVisualization } from "./ResultVisualization";
import { CodeAuditPanel } from "./CodeAuditPanel";
import { QueryHistory } from "./QueryHistory";
import { DatasetUnderstandingCard } from "./DatasetUnderstandingCard";
import { DatasetOverviewCard } from "./DatasetOverviewCard";
import { AutomaticInsights } from "./AutomaticInsights";
import { AnalyticalExplanationCard } from "./AnalyticalExplanationCard";
import { Send, Sparkles, AlertCircle, Loader2, Table2 } from "lucide-react";


interface AnalysisWorkspaceProps {
  dataset: DatasetMetadata;
  initialQuestion?: string;
  onDeleteDataset?: () => void;
}

const LOADING_MESSAGES = [
  "Reading your question...",
  "Generating analysis code...",
  "Running the calculation safely...",
  "Almost done — complex questions take 15-20 seconds...",
];

export const AnalysisWorkspace: React.FC<AnalysisWorkspaceProps> = ({
  dataset,
  initialQuestion,
  onDeleteDataset,
}) => {
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [currentResult, setCurrentResult] = useState<AnalyzeResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isQuotaExceeded, setIsQuotaExceeded] = useState(false);
  const [quotaMessage, setQuotaMessage] = useState<string | null>(null);
  const [savedExamples, setSavedExamples] = useState<SavedExample[]>([]);
  const [activeExampleIndex, setActiveExampleIndex] = useState(0);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [activeHistoryId, setActiveHistoryId] = useState<string | undefined>();

  const handleDeleteData = async () => {
    try {
      await deleteDataset(dataset.dataset_id);
    } catch (e) {
      console.error("Failed to delete dataset", e);
    }
    if (onDeleteDataset) {
      onDeleteDataset();
    }
  };

  // Auto-display saved example when initialQuestion is provided (e.g. from sample dataset or example cards)
  useEffect(() => {
    if (initialQuestion) {
      getExamples()
        .then((examples) => {
          setSavedExamples(examples);
          const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
          const targetNorm = norm(initialQuestion);

          let matched = examples.find((ex) => {
            const exNorm = norm(ex.question);
            return exNorm.includes(targetNorm) || targetNorm.includes(exNorm);
          });

          if (!matched) {
            const keywords = ["revenue", "category", "profit", "trend"];
            for (const kw of keywords) {
              if (targetNorm.includes(kw)) {
                matched = examples.find((ex) => norm(ex.question).includes(kw));
                if (matched) break;
              }
            }
          }

          const exampleToDisplay = matched || examples[0];
          if (exampleToDisplay) {
            const exIdx = examples.indexOf(exampleToDisplay);
            setActiveExampleIndex(exIdx >= 0 ? exIdx : 0);
            setQuestion(exampleToDisplay.question);
            setCurrentResult({
              question: exampleToDisplay.question,
              code: `# Pre-calculated verified result for ${dataset.filename}\n# Question: ${exampleToDisplay.question}`,
              validation_status: "passed",
              validation_message: "Pre-calculated verified result",
              execution_status: "passed",
              formatted_result: exampleToDisplay.answer_text,
              visualization: exampleToDisplay.chart_data,
              execution_time_ms: 1,
            });
          }
        })
        .catch((err) => {
          console.error("Failed to load initial example", err);
        });
    }
  }, [initialQuestion, dataset.filename]);

  // Cycle through loading messages while query is running
  useEffect(() => {
    if (!isLoading) {
      setLoadingStep(0);
      return;
    }

    setLoadingStep(0);
    const t1 = setTimeout(() => setLoadingStep(1), 3000);
    const t2 = setTimeout(() => setLoadingStep(2), 8000);
    const t3 = setTimeout(() => setLoadingStep(3), 15000);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, [isLoading]);

  // Load saved examples when quota is exceeded or rate limit reached
  useEffect(() => {
    if (isQuotaExceeded && savedExamples.length === 0) {
      getExamples().then(setSavedExamples).catch(console.error);
    }
  }, [isQuotaExceeded, savedExamples.length]);

  const handleAnalyze = async (queryText?: string) => {
    const q = (queryText || question).trim();
    if (!q) return;

    setErrorMessage(null);
    setIsQuotaExceeded(false);
    setIsLoading(true);

    try {
      const response = await analyzeQuery(dataset.dataset_id, q);

      if (response.quota_exhausted || response.session_limit_reached) {
        setIsQuotaExceeded(true);
        setQuotaMessage(
          response.message ||
            "Daily analysis limit reached. Explore the saved examples below to see what this tool can do."
        );
        try {
          const examples = await getExamples();
          setSavedExamples(examples);
        } catch (e) {
          console.error("Failed to fetch saved examples", e);
        }
        setCurrentResult(null);
        return;
      }

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
      if (err.errorType === "quota_exceeded" || err.isQuota || err.message?.includes("usage limit") || err.message?.includes("quota") || err.status === 429) {
        setIsQuotaExceeded(true);
        setQuotaMessage(err.message || "Live analysis unavailable — showing saved examples from the sample dataset.");
        try {
          const examples = await getExamples();
          setSavedExamples(examples);
        } catch (e) {
          console.error("Failed to fetch saved examples", e);
        }
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
      <div className="mb-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
        <div>
          <span className="text-xs uppercase tracking-wider text-charcoal-subtle font-medium block mb-1">
            Analysis Workspace
          </span>
          <h1 className="text-2xl sm:text-3xl font-serif text-charcoal tracking-tight">
            Explore {dataset.filename.replace(" (Sample)", "")}
          </h1>
        </div>
        <button
          type="button"
          onClick={handleDeleteData}
          className="text-xs text-charcoal-muted hover:text-red-700 underline underline-offset-2 transition self-start sm:self-auto cursor-pointer"
          title="Delete this dataset and clear session data"
        >
          Delete my data
        </button>
      </div>

      {/* Dedicated Dataset Overview Card */}
      <DatasetOverviewCard
        dataset={dataset}
        onSelectQuery={(q) => {
          setQuestion(q);
          handleAnalyze(q);
        }}
        isLoading={isLoading}
      />

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
              placeholder={
                isQuotaExceeded
                  ? "Analysis paused — quota reached. Come back tomorrow for live analysis."
                  : "Ask anything about your data... (e.g. Which region generated the most revenue?)"
              }
              title={
                isQuotaExceeded
                  ? "Analysis paused — quota reached. Come back tomorrow for live analysis."
                  : undefined
              }
              disabled={isLoading || isQuotaExceeded}
              className={`w-full px-4 py-3 text-sm text-charcoal bg-cream/40 border border-editorial-border rounded focus:outline-none focus:border-editorial-accent focus:ring-1 focus:ring-editorial-accent transition font-sans ${
                isQuotaExceeded ? "opacity-60 bg-stone-100 cursor-not-allowed" : ""
              }`}
            />
          </div>

          <button
            type="submit"
            disabled={isLoading || !question.trim() || isQuotaExceeded}
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
              disabled={isLoading || isQuotaExceeded}
              className="text-[11px] text-charcoal-muted hover:text-charcoal hover:bg-editorial-hover px-2.5 py-1 rounded border border-editorial-border/80 transition disabled:opacity-40 disabled:cursor-not-allowed"
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
          <p className="text-sm font-medium text-charcoal">{LOADING_MESSAGES[loadingStep]}</p>
          <p className="text-xs text-charcoal-muted font-light">
            Generating Pandas query, verifying AST safety rules, and executing in sandbox.
          </p>
        </div>
      )}

      {/* Polished Quota Error State with Saved Examples */}
      {isQuotaExceeded && !isLoading && (
        <div className="mt-6 space-y-6">
          {/* Yellow / Amber Banner */}
          <div className="p-4 rounded-lg border border-amber-300 bg-amber-50/90 text-amber-950 shadow-sm flex items-start space-x-3.5">
            <AlertCircle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <h3 className="font-serif text-base font-semibold text-amber-950">
                Live analysis unavailable — showing saved examples from the sample dataset
              </h3>
              <p className="text-xs text-amber-800 leading-relaxed font-sans">
                {quotaMessage ||
                  "Daily analysis limit reached. Gemini's free tier resets daily. Explore the saved examples below to see what this tool can do."}
              </p>
            </div>
          </div>

          {/* Saved Examples Section */}
          {savedExamples.length > 0 && (
            <div className="space-y-4">
              {/* Click-through Header & Tabs */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-editorial-border pb-3">
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-semibold uppercase tracking-wider text-charcoal-subtle">
                    Saved Examples ({activeExampleIndex + 1} of {savedExamples.length})
                  </span>
                  <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-stone-100 text-stone-600 border border-stone-200">
                    Saved Example
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <button
                    type="button"
                    onClick={() =>
                      setActiveExampleIndex((prev) => (prev > 0 ? prev - 1 : savedExamples.length - 1))
                    }
                    className="px-2.5 py-1 text-xs border border-editorial-border rounded bg-surface hover:bg-editorial-hover text-charcoal transition"
                  >
                    ← Prev
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      setActiveExampleIndex((prev) => (prev < savedExamples.length - 1 ? prev + 1 : 0))
                    }
                    className="px-2.5 py-1 text-xs border border-editorial-border rounded bg-surface hover:bg-editorial-hover text-charcoal transition"
                  >
                    Next →
                  </button>
                </div>
              </div>

              {/* Pills for the 5 questions */}
              <div className="flex flex-wrap gap-2">
                {savedExamples.map((ex, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => setActiveExampleIndex(idx)}
                    className={`px-3 py-1.5 rounded-full text-xs font-medium transition text-left ${
                      activeExampleIndex === idx
                        ? "bg-editorial-accent text-white shadow-sm"
                        : "bg-surface border border-editorial-border text-charcoal-muted hover:text-charcoal hover:bg-stone-50"
                    }`}
                  >
                    {ex.question}
                  </button>
                ))}
              </div>

              {/* Render Active Example (matches real result card) */}
              {savedExamples[activeExampleIndex] && (
                <div className="border border-editorial-border rounded-lg bg-surface p-6 shadow-sm">
                  <div className="flex items-center justify-between pb-3 mb-4 border-b border-editorial-border">
                    <div className="flex items-center space-x-2.5">
                      <span className="text-xs uppercase tracking-wider text-charcoal-subtle font-semibold">
                        {savedExamples[activeExampleIndex].question}
                      </span>
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-stone-100 text-stone-600 border border-stone-200">
                        Saved Example
                      </span>
                    </div>
                    <span className="text-[11px] text-charcoal-muted font-mono">
                      {savedExamples[activeExampleIndex].label}
                    </span>
                  </div>

                  {/* Prominent Answer Card */}
                  <div className="text-2xl sm:text-3xl font-serif text-charcoal font-medium leading-snug">
                    {savedExamples[activeExampleIndex].answer_text}
                  </div>

                  {/* Structured Recharts Visualization */}
                  {savedExamples[activeExampleIndex].chart_data && (
                    <ResultVisualization visualization={savedExamples[activeExampleIndex].chart_data} />
                  )}
                </div>
              )}
            </div>
          )}
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
        <div className="mt-6 border border-editorial-border rounded-lg bg-surface p-6 shadow-sm">
          <div className="flex items-center justify-between pb-3 mb-4 border-b border-editorial-border">
            <div className="flex items-center space-x-2">
              <span className="text-[11px] uppercase tracking-wider text-charcoal-subtle font-semibold">
                {currentResult.question ? currentResult.question : "Analysis Result"}
              </span>
              {currentResult.validation_message?.includes("Pre-calculated") && (
                <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-stone-100 text-stone-600 border border-stone-200">
                  Saved Example
                </span>
              )}
            </div>
            <span className="text-[11px] text-charcoal-muted font-mono">
              {currentResult.validation_message?.includes("Pre-calculated")
                ? "Pre-calculated • 1ms"
                : `${currentResult.execution_time_ms}ms execution`}
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

      {/* Workspace Privacy & Data Deletion Footer (Task 6) */}
      <div className="mt-12 pt-6 border-t border-editorial-border flex flex-col sm:flex-row items-center justify-between text-xs text-charcoal-muted gap-2">
        <span>Session data stored in-memory (auto-expires in 1 hour). No sensitive personal data retained.</span>
        <button
          type="button"
          onClick={handleDeleteData}
          className="hover:text-red-700 underline underline-offset-2 transition cursor-pointer"
        >
          Delete my data
        </button>
      </div>
    </div>
  );
};
