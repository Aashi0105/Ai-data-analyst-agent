"use client";

import React, { useRef, useState } from "react";
import { UploadCloud, Sparkles, AlertCircle } from "lucide-react";
import { uploadDataset, loadSampleDataset } from "@/lib/api";
import { DatasetMetadata } from "@/types";

const MAX_FILE_SIZE_MB = 25;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024; // 25 MiB

interface LandingUploadProps {
  onDatasetLoaded: (dataset: DatasetMetadata, initialQuestion?: string) => void;
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

  const handleSampleClick = async (questionToDisplay?: string) => {
    setErrorMessage(null);
    setIsLoading(true);
    try {
      const metadata = await loadSampleDataset();
      onDatasetLoaded(metadata, questionToDisplay || "Which region has the highest total revenue?");
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
      <div className="text-center mb-8">
        <span className="text-xs uppercase tracking-widest text-editorial-accent font-medium mb-3 block">
          AI-Powered Data Exploration
        </span>
        <h1 className="text-4xl sm:text-5xl font-serif text-charcoal font-normal tracking-tight mb-4">
          Ask questions about your data. Get answers instantly.
        </h1>
        <p className="text-charcoal-muted text-base max-w-xl mx-auto font-light leading-relaxed">
          Upload a CSV and ask anything in plain English — the agent writes the code, runs it safely, and explains what it found. No Python needed.
        </p>
      </div>

      {/* Privacy Notice (Task 6) */}
      <p className="text-xs text-charcoal-muted max-w-xl text-center mb-3 font-light leading-relaxed">
        Your data is processed securely: only column names, data types, and 3 sample rows are sent to Google Gemini for analysis. Data is held in memory for up to 1 hour, then automatically deleted. Do not upload files containing sensitive personal information.
      </p>

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

      {/* Action Buttons: Primary Upload & Secondary Sample Dataset CTA (Task 3) */}
      <div className="mt-6 flex flex-col sm:flex-row items-center justify-center gap-3.5 w-full">
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={isLoading}
          className="w-full sm:w-auto px-6 py-3 bg-editorial-accent hover:bg-[#153427] text-white text-xs sm:text-sm font-medium tracking-wide uppercase rounded shadow-sm transition flex items-center justify-center space-x-2 disabled:opacity-50 shrink-0 cursor-pointer"
        >
          <UploadCloud className="w-4 h-4" />
          <span>Upload your CSV</span>
        </button>

        <button
          type="button"
          onClick={() => handleSampleClick("Which region has the highest total revenue?")}
          disabled={isLoading}
          className="w-full sm:w-auto px-6 py-3 bg-surface hover:bg-stone-100 text-editorial-accent border-2 border-editorial-accent text-xs sm:text-sm font-medium tracking-wide rounded shadow-sm transition flex items-center justify-center space-x-2 disabled:opacity-50 shrink-0 cursor-pointer"
        >
          <Sparkles className="w-4 h-4 text-editorial-accent" />
          <span>Try with a sample dataset →</span>
        </button>
      </div>

      {/* Error Message */}
      {errorMessage && (
        <div className="mt-4 w-full p-3 rounded border border-red-200 bg-red-50 text-xs text-red-700 flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Who It's For Section (Task 5) */}
      <div className="mt-8 text-center">
        <p className="text-xs font-medium text-charcoal-subtle uppercase tracking-wider mb-2">
          Built for people who work with data but not code:
        </p>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-2 sm:gap-6 text-xs text-charcoal-muted">
          <span>• HR managers analyzing workforce trends</span>
          <span>• Operations teams tracking performance</span>
          <span>• Anyone who wants answers from spreadsheets</span>
        </div>
      </div>

      {/* Sample Query Prompts - Clickable Question Cards (Task 4) */}
      <div className="mt-10 w-full pt-6 border-t border-editorial-border">
        <p className="text-xs uppercase tracking-wider text-charcoal-subtle font-medium text-center mb-4">
          Example questions you can ask
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {[
            "Which region has the highest revenue?",
            "What is the top-selling product category?",
            "Which product has the highest profit?",
          ].map((prompt, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => handleSampleClick(prompt)}
              disabled={isLoading}
              className="p-3.5 rounded border border-editorial-border bg-surface/90 hover:bg-editorial-hover hover:border-editorial-accent text-xs text-charcoal font-normal text-left cursor-pointer transition-all shadow-sm hover:shadow flex items-start justify-between group disabled:opacity-50"
            >
              <span>
                <span className="text-charcoal-subtle font-serif italic mr-1.5">“</span>
                {prompt}
                <span className="text-charcoal-subtle font-serif italic ml-1.5">”</span>
              </span>
              <span className="text-editorial-accent opacity-0 group-hover:opacity-100 transition-opacity ml-1.5 font-bold text-xs shrink-0">
                →
              </span>
            </button>
          ))}
        </div>
      </div>
    </main>
  );
};
