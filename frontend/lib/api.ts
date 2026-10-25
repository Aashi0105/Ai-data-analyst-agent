import { DatasetMetadata, AnalyzeResponse } from "@/types";

// Centralized API base URL configuration
// Accepts: http://localhost:8000, http://localhost:8000/, http://localhost:8000/api
const RAW_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const NORMALIZED_BASE = RAW_URL.replace(/\/+$/, "").replace(/\/api$/, "");
const API_BASE = `${NORMALIZED_BASE}/api`;

/**
 * Wraps fetch to provide user-friendly error messages when the backend server
 * is unreachable or down, while preserving HTTP error details.
 */
async function safeFetch(url: string, options?: RequestInit): Promise<Response> {
  try {
    return await fetch(url, options);
  } catch (err: any) {
    if (err instanceof TypeError || err?.name === "TypeError" || err?.message?.includes("fetch")) {
      throw new Error(
        "Unable to connect to the analysis server. Make sure the FastAPI server is running on port 8000."
      );
    }
    throw err;
  }
}

export async function uploadDataset(file: File): Promise<DatasetMetadata> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await safeFetch(`${API_BASE}/datasets/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Failed to upload dataset" }));
    throw new Error(err.detail || `Upload failed with status ${res.status}`);
  }

  return res.json();
}

export async function loadSampleDataset(): Promise<DatasetMetadata> {
  const res = await safeFetch(`${API_BASE}/datasets/sample`, {
    method: "POST",
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Failed to load sample dataset" }));
    throw new Error(err.detail || `Failed to load sample dataset (HTTP ${res.status})`);
  }

  return res.json();
}

export async function getDataset(datasetId: string): Promise<DatasetMetadata> {
  const res = await safeFetch(`${API_BASE}/datasets/${datasetId}`, {
    method: "GET",
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Dataset not found" }));
    throw new Error(err.detail || "Dataset not found");
  }

  return res.json();
}

export async function analyzeQuery(datasetId: string, question: string): Promise<AnalyzeResponse> {
  const res = await safeFetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      dataset_id: datasetId,
      question: question.trim(),
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    if (res.status === 429 || err.error === "quota_exceeded") {
      const quotaErr: any = new Error(
        err.message || "AI analysis is temporarily unavailable because the current API usage limit has been reached."
      );
      quotaErr.isQuota = true;
      quotaErr.errorType = "quota_exceeded";
      throw quotaErr;
    }
    throw new Error(err.detail || err.message || `Analysis request failed (HTTP ${res.status})`);
  }

  return res.json();
}

export async function deleteDataset(datasetId: string): Promise<boolean> {
  try {
    const res = await safeFetch(`${API_BASE}/datasets/${datasetId}`, {
      method: "DELETE",
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
