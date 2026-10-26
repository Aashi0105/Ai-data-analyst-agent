"use client";

import React, { useState } from "react";
import { X, Table, FileSpreadsheet, Layers, BarChart2 } from "lucide-react";
import { DatasetMetadata } from "@/types";

interface DatasetInspectorProps {
  dataset: DatasetMetadata;
  isOpen: boolean;
  onClose: () => void;
}

export const DatasetInspector: React.FC<DatasetInspectorProps> = ({
  dataset,
  isOpen,
  onClose,
}) => {
  const [activeTab, setActiveTab] = useState<"preview" | "schema" | "stats">("preview");

  if (!isOpen) return null;

  const profile = dataset.profile;
  const colProfiles = profile?.column_profiles || {};

  return (
    <div className="fixed inset-0 z-50 bg-charcoal/40 backdrop-blur-sm flex items-center justify-center p-4 sm:p-6">
      <div className="bg-surface border border-editorial-border rounded-xl shadow-xl w-full max-w-4xl max-h-[85vh] flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-editorial-border flex items-center justify-between bg-cream">
          <div className="flex items-center space-x-2">
            <FileSpreadsheet className="w-5 h-5 text-editorial-accent" />
            <h3 className="text-base font-medium text-charcoal">
              Dataset Inspector: <span className="font-mono text-sm">{dataset.filename}</span>
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-full hover:bg-editorial-hover text-charcoal-muted hover:text-charcoal transition"
            aria-label="Close inspector"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab Controls */}
        <div className="px-6 border-b border-editorial-border bg-surface flex space-x-6 text-xs">
          <button
            onClick={() => setActiveTab("preview")}
            className={`py-3 font-medium border-b-2 transition flex items-center space-x-1.5 ${
              activeTab === "preview"
                ? "border-editorial-accent text-editorial-accent"
                : "border-transparent text-charcoal-muted hover:text-charcoal"
            }`}
          >
            <Table className="w-4 h-4" />
            <span>Data Preview (First 10 Rows)</span>
          </button>
          <button
            onClick={() => setActiveTab("schema")}
            className={`py-3 font-medium border-b-2 transition flex items-center space-x-1.5 ${
              activeTab === "schema"
                ? "border-editorial-accent text-editorial-accent"
                : "border-transparent text-charcoal-muted hover:text-charcoal"
            }`}
          >
            <Layers className="w-4 h-4" />
            <span>Column Schema & Nulls ({dataset.columns.length})</span>
          </button>
          <button
            onClick={() => setActiveTab("stats")}
            className={`py-3 font-medium border-b-2 transition flex items-center space-x-1.5 ${
              activeTab === "stats"
                ? "border-editorial-accent text-editorial-accent"
                : "border-transparent text-charcoal-muted hover:text-charcoal"
            }`}
          >
            <BarChart2 className="w-4 h-4" />
            <span>Descriptive Statistics</span>
          </button>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-auto p-6">
          {activeTab === "preview" && (
            <div className="overflow-x-auto border border-editorial-border rounded">
              <table className="w-full text-xs text-left">
                <thead className="bg-cream border-b border-editorial-border text-charcoal font-semibold">
                  <tr>
                    {dataset.columns.map((col, idx) => (
                      <th key={idx} className="py-2.5 px-3 whitespace-nowrap font-mono text-[11px]">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-editorial-border/60 font-mono text-[11px] text-charcoal">
                  {dataset.preview.map((row, rIdx) => (
                    <tr key={rIdx} className="hover:bg-stone-50/70">
                      {dataset.columns.map((col, cIdx) => (
                        <td key={cIdx} className="py-2 px-3 whitespace-nowrap text-charcoal-muted">
                          {row[col] !== null && row[col] !== undefined ? String(row[col]) : <span className="italic text-charcoal-subtle">null</span>}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {activeTab === "schema" && (
            <div className="overflow-x-auto border border-editorial-border rounded">
              <table className="w-full text-xs text-left">
                <thead className="bg-cream border-b border-editorial-border text-charcoal font-semibold">
                  <tr>
                    <th className="py-2.5 px-4">Column Name</th>
                    <th className="py-2.5 px-4">Data Type</th>
                    <th className="py-2.5 px-4">Missing Count</th>
                    <th className="py-2.5 px-4">Missing %</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-editorial-border/60 text-charcoal">
                  {dataset.schema.map((item, idx) => (
                    <tr key={idx} className="hover:bg-stone-50/70">
                      <td className="py-2 px-4 font-mono font-medium">{item.column}</td>
                      <td className="py-2 px-4 font-mono text-charcoal-muted">{item.type}</td>
                      <td className="py-2 px-4">
                        <span className={item.missing_count > 0 ? "text-amber-700 font-medium" : "text-charcoal-muted"}>
                          {item.missing_count}
                        </span>
                      </td>
                      <td className="py-2 px-4 text-charcoal-muted">{item.missing_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {activeTab === "stats" && (
            <div className="space-y-4">
              <div className="overflow-x-auto border border-editorial-border rounded">
                <table className="w-full text-xs text-left">
                  <thead className="bg-cream border-b border-editorial-border text-charcoal font-semibold">
                    <tr>
                      <th className="py-2.5 px-4">Column</th>
                      <th className="py-2.5 px-3">Type</th>
                      <th className="py-2.5 px-4">Summary Statistics / Distributions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-editorial-border/60 text-charcoal">
                    {dataset.columns.map((col, idx) => {
                      const p = colProfiles[col];
                      return (
                        <tr key={idx} className="hover:bg-stone-50/70">
                          <td className="py-3 px-4 font-mono font-medium align-top whitespace-nowrap">
                            {col}
                          </td>
                          <td className="py-3 px-3 align-top whitespace-nowrap">
                            <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-cream border border-editorial-border">
                              {p?.semantic_type || "column"}
                            </span>
                          </td>
                          <td className="py-3 px-4 text-charcoal-muted text-[11px] align-top">
                            {p?.semantic_type === "numeric" && p.numeric_stats && (
                              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono">
                                <div><span className="text-charcoal-subtle">Min:</span> {p.numeric_stats.min ?? "-"}</div>
                                <div><span className="text-charcoal-subtle">Max:</span> {p.numeric_stats.max ?? "-"}</div>
                                <div><span className="text-charcoal-subtle">Mean:</span> {p.numeric_stats.mean ?? "-"}</div>
                                <div><span className="text-charcoal-subtle">Median:</span> {p.numeric_stats.median ?? "-"}</div>
                                <div><span className="text-charcoal-subtle">Std:</span> {p.numeric_stats.std ?? "-"}</div>
                                <div><span className="text-charcoal-subtle">Q25:</span> {p.numeric_stats.q25 ?? "-"}</div>
                                <div><span className="text-charcoal-subtle">Q75:</span> {p.numeric_stats.q75 ?? "-"}</div>
                              </div>
                            )}

                            {p?.semantic_type in { categorical: 1, boolean: 1, text: 1 } && p?.categorical_stats && (
                              <div>
                                <div className="mb-1">
                                  <span className="text-charcoal-subtle">Unique categories:</span>{" "}
                                  <span className="font-mono font-medium text-charcoal">{p.categorical_stats.unique_count}</span>
                                </div>
                                {p.categorical_stats.top_categories.length > 0 && (
                                  <div className="flex flex-wrap gap-1.5 mt-1">
                                    {p.categorical_stats.top_categories.map((c, cIdx) => (
                                      <span
                                        key={cIdx}
                                        className="inline-block px-1.5 py-0.5 rounded bg-stone-100 border border-stone-200 text-[10px]"
                                      >
                                        <span className="font-medium text-charcoal">{c.value}</span>: {c.count} ({c.pct}%)
                                      </span>
                                    ))}
                                  </div>
                                )}
                              </div>
                            )}

                            {p?.semantic_type === "datetime" && p.date_stats && (
                              <div className="font-mono">
                                <div><span className="text-charcoal-subtle">Earliest:</span> {p.date_stats.min_date ?? "-"}</div>
                                <div><span className="text-charcoal-subtle">Latest:</span> {p.date_stats.max_date ?? "-"}</div>
                              </div>
                            )}

                            {(!p || p.semantic_type === "empty") && (
                              <span className="italic text-charcoal-subtle">Empty or unprofiled column</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 border-t border-editorial-border bg-cream flex justify-between items-center text-xs text-charcoal-muted">
          <span>{dataset.num_rows.toLocaleString()} total records • {dataset.num_cols} columns</span>
          <button
            onClick={onClose}
            className="px-3.5 py-1.5 rounded border border-editorial-border bg-surface text-charcoal hover:bg-editorial-hover transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
