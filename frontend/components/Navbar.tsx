"use client";

import React from "react";
import { Table, RotateCcw } from "lucide-react";

interface NavbarProps {
  hasDataset: boolean;
  onReset?: () => void;
  onOpenInspector?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  hasDataset,
  onReset,
  onOpenInspector,
}) => {
  return (
    <header className="border-b border-editorial-border bg-cream/90 backdrop-blur-sm sticky top-0 z-30 px-6 py-3.5 flex items-center justify-between">
      <div className="flex items-center space-x-3">
        <div className="w-2.5 h-2.5 rounded-full bg-editorial-accent" />
        <span className="font-medium tracking-tight text-charcoal text-sm">
          AI DATA ANALYST
        </span>
        <span className="hidden sm:inline-block text-xs text-charcoal-muted border border-editorial-border rounded px-2 py-0.5 bg-surface">
          Sandboxed AST Execution
        </span>
      </div>

      <div className="flex items-center space-x-3 text-sm">
        {hasDataset && (
          <>
            {onOpenInspector && (
              <button
                onClick={onOpenInspector}
                className="flex items-center space-x-1.5 px-3 py-1.5 text-xs text-charcoal font-medium border border-editorial-border rounded hover:bg-editorial-hover transition"
              >
                <Table className="w-3.5 h-3.5 text-charcoal-muted" />
                <span>Inspect Data</span>
              </button>
            )}
            {onReset && (
              <button
                onClick={onReset}
                className="flex items-center space-x-1.5 px-3 py-1.5 text-xs text-charcoal-muted hover:text-charcoal border border-editorial-border rounded hover:bg-editorial-hover transition"
                title="Change or upload another dataset"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>New Dataset</span>
              </button>
            )}
          </>
        )}
      </div>
    </header>
  );
};
