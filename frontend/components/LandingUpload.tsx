"use client";

import React, { useRef, useState } from "react";
import { UploadCloud, FileText, Sparkles, AlertCircle } from "lucide-react";
import { uploadDataset, loadSampleDataset } from "@/lib/api";
import { DatasetMetadata } from "@/types";

const MAX_FILE_SIZE_MB = 25;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024; // 25 MiB

interface LandingUploadProps {
  onDatasetLoaded: (dataset: DatasetMetadata) => void;
}

export const LandingUpload: React.FC<LandingUploadProps> = ({ onDatasetLoaded }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setErrorMessage("Please upload a CSV (.csv) file.");
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      const fileSizeMB = (file.size / (1024 * 1024)).toFixed(1);
      setErrorMessage(
        `File is too large (${fileSizeMB} MB). Maximum allowed CSV upload size is ${MAX_FILE_SIZE_MB} MB.`
      );
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    setErrorMessage(null);
    setIsLoading(true);

    try {
      const metadata = await uploadDataset(file);
      onDatasetLoaded(metadata);
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to upload and parse CSV.");
    } finally {
      setIsLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleSampleClick = async () => {
    setErrorMessage(null);
    setIsLoading(true);
    try {
      const metadata = await loadSampleDataset();
      onDatasetLoaded(metadata);
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to load sample dataset from backend.");
    } finally {
      setIsLoading(false);
    }
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => {
    setIsDragging(false);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <main className="flex-1 flex flex-col items-center justify-center px-6 py-16 max-w-3xl mx-auto w-full">
      {/* Editorial Headline */}
      <div className="text-center mb-10">
        <span className="text-xs uppercase tracking-widest text-editorial-accent font-medium mb-3 block">
          AI-Powered Data Exploration
        </span>
        <h1 className="text-4xl sm:text-5xl font-serif text-charcoal font-normal tracking-tight mb-4">
          Ask your data anything.
        </h1>
        <p className="text-charcoal-muted text-base max-w-md mx-auto font-light leading-relaxed">
          Upload a CSV and explore it using natural language.
        </p>
      </div>

      {/* Central Upload Area */}
      <div
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`w-full border border-dashed rounded-lg p-10 text-center cursor-pointer transition-all duration-200 bg-surface ${
          isDragging
            ? "border-editorial-accent bg-editorial-hover"
            : "border-editorial-border hover:border-charcoal-subtle hover:bg-stone-50/60"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              handleFile(e.target.files[0]);
            }
          }}
        />

        <div className="flex flex-col items-center justify-center space-y-3">
          <div className="w-12 h-12 rounded-full bg-cream border border-editorial-border flex items-center justify-center text-charcoal-muted">
            <UploadCloud className="w-6 h-6 stroke-[1.5]" />
          </div>

          {isLoading ? (
            <div className="space-y-1">
              <p className="text-sm font-medium text-charcoal">Processing dataset...</p>
              <p className="text-xs text-charcoal-muted">Validating schema & profiling columns</p>
            </div>
          ) : (
            <div className="space-y-1">
              <p className="text-sm font-medium text-charcoal">
                Drop your CSV here <span className="text-charcoal-subtle font-normal">or</span> Choose a file
              </p>
              <p className="text-xs text-charcoal-muted font-light">
                Supports UTF-8, Latin1, and standard comma-delimited tabular files (up to 25 MB)
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Error Message */}
      {errorMessage && (
        <div className="mt-4 w-full p-3 rounded border border-red-200 bg-red-50 text-xs text-red-700 flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* One-click Sample Dataset */}
      <div className="mt-6 flex items-center space-x-2 text-xs text-charcoal-muted">
        <span>Need data to test?</span>
        <button
          onClick={handleSampleClick}
          disabled={isLoading}
          className="font-medium text-editorial-accent hover:underline flex items-center space-x-1"
        >
          <Sparkles className="w-3 h-3" />
          <span>Try with sample e-commerce data (100 rows)</span>
        </button>
      </div>

      {/* Sample Query Prompts */}
      <div className="mt-14 w-full pt-8 border-t border-editorial-border">
        <p className="text-xs uppercase tracking-wider text-charcoal-subtle font-medium text-center mb-4">
          Example questions you can ask
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {[
            "Which region generated the most revenue?",
            "What are my top 5 products?",
            "How did revenue change over time?",
          ].map((prompt, idx) => (
            <div
              key={idx}
              className="p-3.5 rounded border border-editorial-border bg-surface/80 text-xs text-charcoal font-normal text-left"
            >
              <span className="text-charcoal-subtle font-serif italic mr-1.5">“</span>
              {prompt}
              <span className="text-charcoal-subtle font-serif italic ml-1.5">”</span>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
};
