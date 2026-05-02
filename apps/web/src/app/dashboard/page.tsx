"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion, useReducedMotion } from "framer-motion";
import { useLibrary } from "@/contexts/LibraryContext";
import { GradientShell } from "@/components/GradientShell";
import { useToast } from "@/components/ui/ToastProvider";
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
import {
  DocumentListSkeleton,
  LoadingSpinner,
} from "@/components/ui/loading";
import { useDelayedBusy } from "@/hooks/use-delayed-busy";
import { useUserResume } from "@/hooks/use-user-resume";

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

interface JobDescriptionUploadTargetProps {
  busy: boolean;
  disabled: boolean;
  onFileSelect: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

function JobDescriptionUploadTarget({
  busy,
  disabled,
  onFileSelect,
}: JobDescriptionUploadTargetProps) {
  return (
    <label className="group relative flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-orange-200/80 bg-white/60 px-5 py-8 text-center shadow-sm transition-all duration-200 hover:border-zenodrift-accent/60 hover:bg-orange-50/70 focus-within:ring-2 focus-within:ring-zenodrift-accent/25 focus-within:ring-offset-2 sm:min-h-[180px]">
      <input
        type="file"
        accept="application/pdf"
        onChange={onFileSelect}
        disabled={disabled}
        className="sr-only"
        aria-label="Upload job description PDF"
      />
      {busy ? (
        <div className="flex flex-col items-center gap-3">
          <LoadingSpinner size="lg" variant="warm" label="Uploading" />
          <span className="text-sm font-medium text-zenodrift-text-muted">
            Uploading and starting processing...
          </span>
        </div>
      ) : (
        <>
          <div className="rounded-full bg-orange-50 p-4 text-zenodrift-accent shadow-sm transition-all duration-200 group-hover:scale-105 group-hover:bg-white">
            <svg
              className="h-10 w-10"
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
          <span className="mt-4 text-base font-semibold text-zenodrift-text-strong">
            Add a job description PDF
          </span>
          <span className="mt-1.5 max-w-sm text-sm leading-relaxed text-zenodrift-text-muted">
            Drop a JD here or click to upload. We will process it here and open setup when it is ready.
          </span>
        </>
      )}
    </label>
  );
}

export default function DashboardPage() {
  const reduceMotion = useReducedMotion();
  const router = useRouter();
  const { showToast } = useToast();
  const {
    data: docs = [],
    isPending: documentsLoading,
    isError: documentsQueryError,
    error: documentsError,
  } = useDocuments();
  const { data: resumeStatus } = useUserResume();

  /** Hide profile resume (by id or domain) so it never shows as a job row. */
  const jobDescriptionDocs = useMemo(() => {
    const rid = resumeStatus?.document_id?.trim();
    return docs.filter((d) => {
      if (rid && d.id === rid) return false;
      if (d.doc_domain === "user_resume") return false;
      return true;
    });
  }, [docs, resumeStatus?.document_id]);

  const [error, setError] = useState<string | null>(null);
  const [processingId, setProcessingId] = useState<string | null>(null);
  const [pendingUploadName, setPendingUploadName] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const uploadMutation = useUploadJobDescriptionMutation();
  const ingestMutation = useIngestDocumentMutation();
  const deleteMutation = useDeleteDocumentMutation();
  const deleteAllMutation = useDeleteAllDocumentsMutation();
  const uploadBusy = useDelayedBusy(uploadMutation.isPending);
  const clearAllBusy = useDelayedBusy(deleteAllMutation.isPending);

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
      showToast({
        tone: "info",
        message: "Please select a PDF before uploading.",
      });
      return;
    }
    setError(null);
    setPendingUploadName(file.name);
    uploadMutation.mutate(file, {
      onSuccess: (documentId) => {
        setProcessingId(documentId);
        showToast({
          tone: "success",
          message: "Upload complete. Starting document processing.",
        });
      },
      onError: (err) => {
        const message =
          err instanceof ApiError
            ? String(err.detail || err.message)
            : "Upload failed";
        setError(message);
        setPendingUploadName(null);
        showToast({ tone: "error", message });
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
      onSuccess: () => {
        showToast({
          tone: "success",
          message: "Processing started. We will take you to setup when ready.",
        });
      },
      onError: (e) => {
        const message =
          e instanceof ApiError
            ? String(e.detail || e.message)
            : "Failed to start processing";
        setError(message);
        showToast({ tone: "error", message });
        setProcessingId(null);
      },
    });
  };

  const handleDelete = (doc: DocumentSummary) => {
    if (!confirm(`Delete "${doc.filename}"? This cannot be undone.`)) return;
    setDeletingId(doc.id);
    setError(null);
    deleteMutation.mutate(doc.id, {
      onSuccess: () => {
        showToast({
          tone: "success",
          message: `Deleted "${doc.filename}".`,
        });
      },
      onSettled: () => setDeletingId(null),
      onError: (e) => {
        const message =
          e instanceof ApiError
            ? String(e.detail || e.message)
            : "Failed to delete document";
        setError(message);
        showToast({ tone: "error", message });
      },
    });
  };

  const handleClearAll = () => {
    if (
      !confirm(
        `Delete all ${jobDescriptionDocs.length} job description${jobDescriptionDocs.length === 1 ? "" : "s"}? Your profile resume stays put. This cannot be undone.`
      )
    )
      return;
    setError(null);
    deleteAllMutation.mutate(undefined, {
      onSuccess: (result) => {
        showToast({
          tone: "success",
          message: `Removed ${result.count} job description${result.count === 1 ? "" : "s"}.`,
        });
      },
      onError: (e) => {
        const message =
          e instanceof ApiError
            ? String(e.detail || e.message)
            : "Failed to clear documents";
        setError(message);
        showToast({ tone: "error", message });
      },
    });
  };

  const displayError = error ?? listError;
  const processingDoc = processingId
    ? jobDescriptionDocs.find((doc) => doc.id === processingId)
    : null;
  const showPendingUploadState =
    Boolean(pendingUploadName) && (uploadBusy || (Boolean(processingId) && !processingDoc));

  return (
    <GradientShell>
      {/* Hero: product landing style - generous spacing, accent on title */}
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
          </div>

          <div className="hero-glass-card flex flex-col gap-3 p-5 sm:min-w-[260px]">
            <p className="text-sm font-medium text-zenodrift-text-strong">
              Already uploaded a JD?
            </p>
            <p className="text-sm leading-relaxed text-zenodrift-text-muted">
              Open your library to pick up an existing role or switch interview context.
            </p>
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

      {/* Dashboard card: profile resume + job descriptions */}
      <section className="dashboard-card px-6 py-6">
        <AccountResumeSection />
        <div className="mb-5 mt-8 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-xs font-medium uppercase tracking-wider text-zenodrift-text-muted">
              Job descriptions
            </p>
            <h2 className="mt-1 text-2xl font-semibold tracking-tight text-zenodrift-text-strong">
              Add and manage JDs
            </h2>
            <p className="mt-2 max-w-2xl text-sm leading-relaxed text-zenodrift-text-muted">
              Upload roles here, watch processing, and start interview setup from the same workspace. The profile resume above goes with every job you add.
            </p>
          </div>
          {jobDescriptionDocs.length > 0 && (
            <button
              onClick={handleClearAll}
              disabled={clearAllBusy}
              className="shrink-0 text-xs font-medium text-red-600 hover:text-red-700 disabled:opacity-50"
            >
              {clearAllBusy ? "Clearing…" : "Clear all JDs"}
            </button>
          )}
        </div>

        <div className="mb-5">
          <JobDescriptionUploadTarget
            busy={uploadBusy}
            disabled={uploadMutation.isPending}
            onFileSelect={handleFileSelect}
          />
        </div>

        {showPendingUploadState && pendingUploadName && (
          <div className="mb-4 flex flex-col gap-3 rounded-2xl border border-indigo-100 bg-indigo-50/70 px-4 py-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold text-zenodrift-text-strong">
                {pendingUploadName}
              </p>
              <p className="mt-1 text-sm text-indigo-700">
                Processing this JD. Setup will open when ready.
              </p>
            </div>
            <div className="flex shrink-0 items-center gap-2 text-sm font-medium text-indigo-700">
              <LoadingSpinner size="sm" variant="warm" decorative />
              Processing
            </div>
          </div>
        )}

        {documentsLoading ? (
          <div className="py-6">
            <DocumentListSkeleton />
          </div>
        ) : jobDescriptionDocs.length === 0 ? (
          <div className="rounded-2xl bg-white/40 px-5 py-6 text-center">
            <p className="text-sm font-medium text-zenodrift-text-strong">
              No job descriptions yet
            </p>
            <p className="mx-auto mt-1 max-w-md text-sm leading-relaxed text-zenodrift-text-muted">
              Add a JD above and it will show up here with processing status, detected role details, and interview actions.
            </p>
          </div>
        ) : (
          <motion.ul
            className="space-y-3"
            {...(reduceMotion
              ? {}
              : {
                  initial: { opacity: 0, y: 10 },
                  animate: { opacity: 1, y: 0 },
                  transition: { duration: 0.42, ease: [0.22, 1, 0.36, 1] },
                })}
          >
            {jobDescriptionDocs.map((doc) => (
              <li
                key={doc.id}
                className="flex flex-col gap-4 rounded-2xl border border-white/50 bg-white/55 p-4 shadow-sm sm:flex-row sm:items-center sm:justify-between sm:gap-6"
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
                  {doc.status === "processing" && (
                    <p className="mt-2 text-sm text-indigo-700">
                      Processing this JD. Setup will open when ready.
                    </p>
                  )}
                  {doc.status === "pending" && (
                    <p className="mt-2 text-sm text-amber-700">
                      Waiting for upload confirmation.
                    </p>
                  )}
                  {doc.status === "uploaded" && (
                    <p className="mt-2 text-sm text-blue-700">
                      Upload complete. Start processing here if it did not begin automatically.
                    </p>
                  )}
                  {doc.status === "failed" && doc.error_message && (
                    <p className="mt-2 text-sm text-red-600">
                      {doc.error_message}
                    </p>
                  )}
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
                              title={c.attempts_count > 0 ? `Attempts: ${c.attempts_count}, avg: ${c.avg_score ?? "n/a"}` : undefined}
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
                <div className="flex shrink-0 flex-wrap items-center gap-2 sm:justify-end">
                  {doc.status === "processing" && (
                    <button
                      disabled
                      className="inline-flex items-center gap-2 rounded-lg bg-indigo-100 px-3 py-2 text-sm font-medium text-indigo-700 disabled:opacity-80"
                    >
                      <LoadingSpinner size="sm" variant="warm" decorative />
                      Processing
                    </button>
                  )}
                  {doc.status === "pending" && (
                    <button
                      disabled
                      className="rounded-lg bg-amber-100 px-3 py-2 text-sm font-medium text-amber-700 disabled:opacity-80"
                    >
                      Pending
                    </button>
                  )}
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
                            href={`/documents/${doc.id}`}
                            className="inline-flex items-center rounded-lg bg-white/80 px-3 py-2 text-sm font-medium text-zenodrift-text shadow-sm ring-1 ring-neutral-200/80 transition-all duration-200 hover:bg-white hover:text-zenodrift-text-strong focus:outline-none focus:ring-2 focus:ring-zenodrift-accent focus:ring-offset-2"
                          >
                            View JD details
                          </Link>
                          <Link
                            href={`/interview/setup/${doc.id}`}
                            className="inline-flex items-center rounded-lg bg-zenodrift-accent px-3 py-2 text-sm font-medium text-white shadow-zenodrift-soft transition-all duration-200 hover:bg-zenodrift-accent-hover hover:shadow-md focus:outline-none focus:ring-2 focus:ring-zenodrift-accent focus:ring-offset-2"
                          >
                            Start Interview
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
          </motion.ul>
        )}
      </section>
    </GradientShell>
  );
}
