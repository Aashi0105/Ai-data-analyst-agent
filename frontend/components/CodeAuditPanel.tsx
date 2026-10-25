"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronUp, ShieldCheck, CheckCircle2, XCircle, Code2 } from "lucide-react";

interface CodeAuditPanelProps {
  code: string;
  validationStatus: "passed" | "failed";
  executionStatus: "passed" | "failed";
  validationMessage?: string;
  executionTimeMs?: number;
}

export const CodeAuditPanel: React.FC<CodeAuditPanelProps> = ({
  code,
  validationStatus,
  executionStatus,
  validationMessage,
  executionTimeMs,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  if (!code) return null;

  const isAllPassed = validationStatus === "passed" && executionStatus === "passed";

  return (
    <div className="mt-5 border border-editorial-border rounded-lg bg-surface overflow-hidden">
      {/* Collapsible Header */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-editorial-hover transition text-left"
      >
        <div className="flex items-center space-x-2">
          <Code2 className="w-4 h-4 text-editorial-accent" />
          <span className="text-xs font-medium text-charcoal">
            View Generated Pandas Code
          </span>
          <span className="text-[10px] text-charcoal-muted font-mono">
            ({executionTimeMs || 0}ms)
          </span>
        </div>

        <div className="flex items-center space-x-3">
          {/* Status Badges */}
          <div className="flex items-center space-x-1.5">
            {validationStatus === "passed" ? (
              <span className="inline-flex items-center space-x-1 text-[11px] text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                <CheckCircle2 className="w-3 h-3" />
                <span>AST Validated</span>
              </span>
            ) : (
              <span className="inline-flex items-center space-x-1 text-[11px] text-red-800 bg-red-50 px-2 py-0.5 rounded border border-red-200">
                <XCircle className="w-3 h-3" />
                <span>AST Rejected</span>
              </span>
            )}

            {executionStatus === "passed" ? (
              <span className="inline-flex items-center space-x-1 text-[11px] text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                <ShieldCheck className="w-3 h-3" />
                <span>Sandbox Executed</span>
              </span>
            ) : (
              <span className="inline-flex items-center space-x-1 text-[11px] text-red-800 bg-red-50 px-2 py-0.5 rounded border border-red-200">
                <XCircle className="w-3 h-3" />
                <span>Execution Failed</span>
              </span>
            )}
          </div>

          {isOpen ? (
            <ChevronUp className="w-4 h-4 text-charcoal-muted" />
          ) : (
            <ChevronDown className="w-4 h-4 text-charcoal-muted" />
          )}
        </div>
      </button>

      {/* Expandable Code Area */}
      {isOpen && (
        <div className="border-t border-editorial-border p-4 bg-stone-900 text-stone-100 text-xs font-mono">
          <pre className="overflow-x-auto leading-relaxed">
            <code>{code}</code>
          </pre>

          <div className="mt-3 pt-3 border-t border-stone-700/80 flex flex-wrap items-center justify-between text-[11px] text-stone-400">
            <span>Security: Whitelisted AST nodes • No arbitrary imports • Deep-copied DataFrame</span>
            {validationMessage && <span>Status: {validationMessage}</span>}
          </div>
        </div>
      )}
    </div>
  );
};
