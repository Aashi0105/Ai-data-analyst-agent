"use client";

import React, { useState } from "react";
import { DatasetMetadata } from "@/types";
import { deleteDataset } from "@/lib/api";
import { Navbar } from "@/components/Navbar";
import { LandingUpload } from "@/components/LandingUpload";
import { DatasetSidebar } from "@/components/DatasetSidebar";
import { AnalysisWorkspace } from "@/components/AnalysisWorkspace";
import { DatasetInspector } from "@/components/DatasetInspector";

export default function Home() {
  const [dataset, setDataset] = useState<DatasetMetadata | null>(null);
  const [isInspectorOpen, setIsInspectorOpen] = useState(false);
  const [isResetting, setIsResetting] = useState(false);

  const handleReset = async () => {
    if (isResetting) return;
    const currentId = dataset?.dataset_id;
    setIsResetting(true);
    setDataset(null);
    setIsInspectorOpen(false);

    if (currentId) {
      try {
        await deleteDataset(currentId);
      } catch {
        // Silently tolerate deletion errors; client state is already cleanly reset
      } finally {
        setIsResetting(false);
      }
    } else {
      setIsResetting(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-cream text-charcoal">
      {/* Top Navigation */}
      <Navbar
        hasDataset={!!dataset}
        onReset={handleReset}
        onOpenInspector={() => setIsInspectorOpen(true)}
      />

      {/* Main Screen Layout */}
      {!dataset ? (
        <LandingUpload onDatasetLoaded={(meta) => setDataset(meta)} />
      ) : (
        <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
          {/* Left Dataset Sidebar */}
          <DatasetSidebar
            dataset={dataset}
            onOpenInspector={() => setIsInspectorOpen(true)}
          />

          {/* Main Analysis Workspace */}
          <AnalysisWorkspace dataset={dataset} />
        </div>
      )}

      {/* Dataset Inspector Modal */}
      {dataset && (
        <DatasetInspector
          dataset={dataset}
          isOpen={isInspectorOpen}
          onClose={() => setIsInspectorOpen(false)}
        />
      )}
    </div>
  );
}
