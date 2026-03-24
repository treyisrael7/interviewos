"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useLibrary } from "@/contexts/LibraryContext";
import { GradientShell } from "@/components/GradientShell";
import { ApiError, type DocumentSummary } from "@/lib/api";
import { formatDocumentsListError } from "@/lib/query-error";
import {
  useDocuments,
  useUploadJobDescriptionMutation,
  useIngestDocumentMutation,
  useDeleteDocumentMutation,
  useDeleteAllDocumentsMutation,
} from "@/hooks/use-documents";
import { AccountResumeSection } from "@/components/dashboard/AccountResumeSection";

const STATUS_LABELS: Record<string, string> = {
  pending: "Pending",
  uploaded: "Uploaded",
  processing: "Processing",
  ready: "Ready",
  failed: "Failed",
};

const DOMAIN_LABELS: Record<string, string> = {
  technical: "Technical",
  finance: "Finance",
  healthcare_social_work: "Healthcare & Social Work",
  sales_marketing: "Sales & Marketing",
  operations: "Operations",
  education: "Education",
  general_business: "General Business",
};

const SENIORITY_LABELS: Record<string, string> = {
  entry: "Entry",
  mid: "Mid",
  senior: "Senior",
};

const STATUS_STYLES: Record<string, string> = {
  pending: "text-amber-700 bg-amber-100/80",
  uploaded: "text-blue-700 bg-blue-100/80",
  processing:
    "text-indigo-700 bg-indigo-100/80 animate-pulse",
  ready: "text-emerald-700 bg-emerald-100/80",
  failed: "text-red-700 bg-red-100/80",
};

function formatUploadedAt(createdAt: string | undefined): string {
  if (!createdAt) return "";
  try {
    const d = new Date(createdAt);
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    return d.toLocaleDateString();
  } catch {
    return "";
  }
}

export default function DashboardPage() {
  const router = useRouter();
  const {
    data: docs = [],
    isPending: documentsLoading,
    isError: documentsQueryError,
    error: documentsError,
  } = useDocuments();
  const [error, setError] = useState<string | null>(null);
  const [processingId, setProcessingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const uploadMutation = useUploadJobDescriptionMutation();
  const ingestMutation = useIngestDocumentMutation();
  const deleteMutation = useDeleteDocumentMutation();
  const deleteAllMutation = useDeleteAllDocumentsMutation();

  const listError =
    documentsQueryError && documentsError
      ? formatDocumentsListError(documentsError)
      : null;

  // Redirect to Interview Setup when processing completes successfully
  useEffect(() => {
    if (!processingId) return;
    const doc = docs.find((d) => d.id === processingId);
    if (doc?.status === "ready") {
      const docId = doc.id;
      setProcessingId(null);
      router.replace(`/interview/setup/${docId}`);
    }
  }, [docs, processingId, router]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || file.type !== "application/pdf") {
      setError("Please select a PDF file.");
      return;
    }
    setError(null);
    uploadMutation.mutate(file, {
      onSuccess: (documentId) => {
        setProcessingId(documentId);
      },
      onError: (err) => {
        setError(
          err instanceof ApiError
            ? String(err.detail || err.message)
            : "Upload failed"
        );
      },
    });
    e.target.value = "";
  };

  const { openLibrary } = useLibrary();

  const handleProcess = (doc: DocumentSummary) => {
    if (doc.status !== "uploaded") return;
    setProcessingId(doc.id);
    setError(null);
    ingestMutation.mutate(doc.id, {
      onError: (e) => {
        setError(
          e instanceof ApiError
            ? String(e.detail || e.message)
            : "Failed to start processing"
        );
        setProcessingId(null);
      },
    });
  };

  const handleDelete = (doc: DocumentSummary) => {
    if (!confirm(`Delete "${doc.filename}"? This cannot be undone.`)) return;
    setDeletingId(doc.id);
    setError(null);
    deleteMutation.mutate(doc.id, {
      onSettled: () => setDeletingId(null),
      onError: (e) => {
        setError(
          e instanceof ApiError
            ? String(e.detail || e.message)
            : "Failed to delete document"
        );
      },
    });
  };

  const handleClearAll = () => {
    if (!confirm(`Delete all ${docs.length} documents? This cannot be undone.`)) return;
    setError(null);
    deleteAllMutation.mutate(undefined, {
      onError: (e) => {
        setError(
          e instanceof ApiError
            ? String(e.detail || e.message)
            : "Failed to clear documents"
        );
      },
    });
  };

  const displayError = error ?? listError;

  return (
    <GradientShell>
      {/* Hero: product landing style - generous spacing, accent on title, glass for interactive only */}
      <section className="mx-auto w-full max-w-[1160px] pb-6 pt-8 sm:pt-12">
        <div className="grid grid-cols-1 gap-12 lg:grid-cols-[1fr_auto] lg:items-start lg:gap-16">
          {/* Left: product identity - no card, typography + badges */}
          <div className="space-y-10">
            <div>
              <h1 className="relative inline-block pb-4 text-[clamp(2.75rem,5vw,4rem)] font-bold leading-[1.1] tracking-tighter text-zenodrift-text-strong">
                InterviewOS
                <span
                  className="absolute bottom-0 left-0 h-1 w-16 rounded-full bg-gradient-to-r from-zenodrift-accent to-orange-400"
                  aria-hidden
                />
              </h1>
            </div>
            <p className="max-w-[36ch] text-xl leading-relaxed text-zenodrift-text sm:text-2xl">
              Job description–grounded interview practice with evidence-cited feedback.
            </p>
            <div className="flex flex-wrap gap-3">
              <span className="rounded-full border border-white/25 bg-white/20 px-4 py-2 text-sm font-medium text-zenodrift-text">
                Job description–grounded
              </span>
              <span className="rounded-full border border-white/25 bg-white/20 px-4 py-2 text-sm font-medium text-zenodrift-text">
                Evidence-cited
              </span>
              <span className="rounded-full border border-white/25 bg-white/20 px-4 py-2 text-sm font-medium text-zenodrift-text">
                Fast practice
              </span>
            </div>
            <Link
              href="/dashboard/analytics"
              className="inline-flex text-sm font-semibold text-zenodrift-accent transition-colors hover:text-zenodrift-accent-hover"
            >
              View practice analytics →
            </Link>
          </div>

          {/* Right: glass cards for interactive elements only */}
          <div className="flex flex-col gap-4">
            <div className="hero-glass-card overflow-hidden p-6 sm:min-w-[320px] sm:p-8">
              <label className="group relative flex min-h-[200px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-white/40 px-6 py-12 transition-all duration-200 hover:border-zenodrift-accent/50 hover:bg-white/30 focus-within:ring-2 focus-within:ring-zenodrift-accent/25 focus-within:ring-offset-2 focus-within:ring-offset-transparent">
                <input
                  type="file"
                  accept="application/pdf"
                  onChange={handleFileSelect}
                  disabled={uploadMutation.isPending}
                  className="sr-only"
                  aria-label="Upload PDF file"
                />
                {uploadMutation.isPending ? (
                  <div className="flex flex-col items-center gap-4">
                    <div
                      className="h-10 w-10 animate-spin rounded-full border-2 border-white/50 border-t-zenodrift-accent"
                      aria-hidden
                    />
                    <span className="text-sm font-medium text-zenodrift-text-muted">
                      Uploading and processing…
                    </span>
                  </div>
                ) : (
                  <>
                    <div className="rounded-full bg-white/50 p-5 shadow-sm transition-all duration-200 group-hover:scale-105 group-hover:bg-white/70 group-hover:shadow-md">
                      <svg
                        className="h-14 w-14 text-zenodrift-accent"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        aria-hidden
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={1.5}
                          d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                        />
                      </svg>
                    </div>
                    <span className="mt-5 text-base font-semibold text-zenodrift-text">
                      Drop a job description PDF or click to upload
                    </span>
                    <span className="mt-1.5 text-xs text-zenodrift-text-muted">
                      PDF only · processed automatically
                    </span>
                  </>
                )}
              </label>
            </div>
            <button
              onClick={openLibrary}
              className="rounded-2xl bg-gradient-to-r from-orange-500 to-orange-600 px-6 py-3 text-sm font-semibold text-white shadow-lg transition-all duration-200 hover:-translate-y-0.5 hover:shadow-xl focus:outline-none focus:ring-2 focus:ring-zenodrift-accent focus:ring-offset-2 focus:ring-offset-transparent"
            >
              Library
            </button>
          </div>
        </div>
      </section>

      {/* Error alert */}
      {displayError && (
        <div
          className="rounded-2xl border border-red-200/60 bg-red-50/80 px-5 py-4 text-sm text-red-700 shadow-sm"
          role="alert"
        >
          {displayError}
        </div>
      )}

      {/* Dashboard card: Account Resume + Recent Documents */}
      <section className="dashboard-card px-6 py-6">
        <AccountResumeSection />
        <div className="mb-3 mt-4 flex items-center justify-between">
          <h2 className="text-xs font-medium uppercase tracking-wider text-zenodrift-text-muted">
            Recent Documents
          </h2>
          {docs.length > 0 && (
            <button
              onClick={handleClearAll}
              disabled={deleteAllMutation.isPending}
              className="text-xs font-medium text-red-600 hover:text-red-700 disabled:opacity-50"
            >
              {deleteAllMutation.isPending ? "Clearing…" : "Clear all"}
            </button>
          )}
        </div>
        {documentsLoading ? (
          <div className="flex items-center justify-center py-10">
            <div
              className="h-8 w-8 animate-spin rounded-full border-2 border-neutral-200 border-t-zenodrift-accent"
              aria-label="Loading documents"
            />
          </div>
        ) : docs.length === 0 ? (
          <p className="py-6 text-center text-sm text-zenodrift-text-muted">
            No documents yet. Upload a job description PDF above to get started.
          </p>
        ) : (
          <ul className="divide-y divide-neutral-100">
            {docs.map((doc) => (
              <li
                key={doc.id}
                className="flex flex-col gap-2 py-4 first:pt-0 last:pb-0 sm:flex-row sm:items-center sm:justify-between sm:gap-4"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="truncate font-medium text-zenodrift-text-strong">
                      {doc.filename}
                    </span>
                    <span
                      className={`inline-flex shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ${
                        STATUS_STYLES[doc.status] ??
                        "text-zenodrift-text bg-slate-100"
                      }`}
                    >
                      {STATUS_LABELS[doc.status] ?? doc.status}
                    </span>
                    {doc.error_message && (
                      <span
                        className="text-xs text-red-600"
                        title={doc.error_message}
                      >
                        Error
                      </span>
                    )}
                  </div>
                  <div className="mt-0.5 flex flex-wrap gap-x-3 text-xs text-zenodrift-text-muted">
                    {doc.page_count != null && (
                      <span>
                        {doc.page_count} page{doc.page_count !== 1 ? "s" : ""}
                      </span>
                    )}
                    {doc.created_at && formatUploadedAt(doc.created_at) && (
                      <span>{formatUploadedAt(doc.created_at)}</span>
                    )}
                  </div>
                  {doc.status === "ready" && (doc.role_profile || doc.competencies?.length) && (
                    <div className="mt-2 space-y-1.5">
                      {(doc.role_profile || (doc.coverage_total ?? 0) > 0) && (
                        <div className="text-xs text-zenodrift-text-muted">
                          {doc.role_profile && (
                            <>
                              Detected role:{" "}
                              {DOMAIN_LABELS[doc.role_profile.domain] ??
                                doc.role_profile.domain}
                              {" • "}
                              Level:{" "}
                              {SENIORITY_LABELS[doc.role_profile.seniority] ??
                                doc.role_profile.seniority}
                              {(doc.coverage_total ?? 0) > 0 && " • "}
                            </>
                          )}
                          {(doc.coverage_total ?? 0) > 0 && (
                            <span className="font-medium">Coverage: {(doc.coverage_practiced ?? 0)}/{(doc.coverage_total ?? 0)}</span>
                          )}
                        </div>
                      )}
                      {(doc.competencies?.length ?? 0) > 0 ? (
                        <div className="flex flex-wrap gap-1.5">
                          {doc.competencies!.slice(0, 8).map((c) => (
                            <span
                              key={c.id}
                              className="rounded-md bg-white/25 px-2 py-0.5 text-xs font-medium text-zenodrift-text"
                              title={c.attempts_count > 0 ? `Attempts: ${c.attempts_count}, Avg: ${c.avg_score ?? "—"}` : undefined}
                            >
                              {c.label}
                            </span>
                          ))}
                        </div>
                      ) : (
                        doc.role_profile?.focusAreas?.length ? (
                          <div className="flex flex-wrap gap-1.5">
                            {doc.role_profile.focusAreas.slice(0, 8).map((area) => (
                              <span
                                key={area}
                                className="rounded-md bg-white/25 px-2 py-0.5 text-xs font-medium text-zenodrift-text"
                              >
                                {area}
                              </span>
                            ))}
                          </div>
                        ) : null
                      )}
                    </div>
                  )}
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  {doc.status === "uploaded" && (
                    <button
                      onClick={() => handleProcess(doc)}
                      disabled={
                        processingId === doc.id ||
                        (ingestMutation.isPending &&
                          ingestMutation.variables === doc.id)
                      }
                      className="rounded-lg bg-zenodrift-accent px-3 py-2 text-sm font-medium text-white shadow-zenodrift-soft transition-all duration-200 hover:bg-zenodrift-accent-hover hover:shadow-md focus:outline-none focus:ring-2 focus:ring-zenodrift-accent focus:ring-offset-2 disabled:opacity-50"
                    >
                      {processingId === doc.id ||
                      (ingestMutation.isPending &&
                        ingestMutation.variables === doc.id)
                        ? "Processing…"
                        : "Process"}
                    </button>
                  )}
                  {doc.status === "ready" && (
                    <>
                      {doc.doc_domain === "job_description" ? (
                        <>
                          <Link
                            href={`/interview/setup/${doc.id}`}
                            className="inline-flex items-center rounded-lg bg-zenodrift-accent px-3 py-2 text-sm font-medium text-white shadow-zenodrift-soft transition-all duration-200 hover:bg-zenodrift-accent-hover hover:shadow-md focus:outline-none focus:ring-2 focus:ring-zenodrift-accent focus:ring-offset-2"
                          >
                            Start Interview
                          </Link>
                          <Link
                            href={`/documents/${doc.id}`}
                            className="inline-flex items-center rounded-lg bg-neutral-100 px-3 py-2 text-sm font-medium text-zenodrift-text transition-all duration-200 hover:bg-neutral-200 focus:outline-none focus:ring-2 focus:ring-neutral-300 focus:ring-offset-2"
                          >
                            Ask
                          </Link>
                        </>
                      ) : (
                        <Link
                          href={`/documents/${doc.id}`}
                          className="inline-flex items-center rounded-lg bg-zenodrift-accent px-3 py-2 text-sm font-medium text-white shadow-zenodrift-soft transition-all duration-200 hover:bg-zenodrift-accent-hover hover:shadow-md focus:outline-none focus:ring-2 focus:ring-zenodrift-accent focus:ring-offset-2"
                        >
                          Open
                        </Link>
                      )}
                    </>
                  )}
                  <button
                    onClick={() => handleDelete(doc)}
                    disabled={deletingId === doc.id}
                    className="inline-flex items-center rounded-lg bg-neutral-100 px-3 py-2 text-sm font-medium text-red-600 transition-all duration-200 hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-red-200 focus:ring-offset-2 disabled:opacity-50"
                    title="Delete document"
                  >
                    {deletingId === doc.id ? "Deleting…" : "Delete"}
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </GradientShell>
  );
}
