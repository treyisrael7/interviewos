"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { GradientShell } from "@/components/GradientShell";
import { LoadingCenter } from "@/components/ui/loading";
import { useToast } from "@/components/ui/ToastProvider";
import { InterviewFocusMode } from "@/components/interview/InterviewFocusMode";
import {
  ApiError,
  parseAskStructuredAnswer,
  type AskResponse,
} from "@/lib/api";
import { AskAnswerDisplay } from "@/components/AskAnswerDisplay";
import { formatQueryError } from "@/lib/query-error";
import { useDocument, useDeleteDocumentMutation } from "@/hooks/use-documents";
import { useAskQuestionMutation } from "@/hooks/use-ask-question";
import { useDelayedBusy } from "@/hooks/use-delayed-busy";

const STATUS_LABELS: Record<string, string> = {
  pending: "Pending",
  uploaded: "Uploaded",
  processing: "Processing",
  ready: "Ready",
  failed: "Failed",
};

const STATUS_STYLES: Record<string, string> = {
  pending: "text-amber-700 bg-amber-100/80 ring-1 ring-amber-200/60",
  uploaded: "text-blue-700 bg-blue-100/80 ring-1 ring-blue-200/60",
  processing:
    "text-indigo-700 bg-indigo-100/80 ring-1 ring-indigo-200/60 animate-pulse",
  ready: "text-emerald-700 bg-emerald-100/80 ring-1 ring-emerald-200/60",
  failed: "text-red-700 bg-red-100/80 ring-1 ring-red-200/60",
};

type Tab = "chat" | "interview";

const TAB_BASE =
  "rounded-t-lg px-4 py-2.5 text-sm font-medium transition-colors duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-zenodrift-accent focus-visible:ring-offset-2 focus-visible:ring-offset-white";

function DocumentPageContent() {
  const params = useParams();
  const router = useRouter();
  const searchParams = useSearchParams();
  const id = params.id as string;
  const [tab, setTab] = useState<Tab>("chat");
  const [actionError, setActionError] = useState<string | null>(null);

  const {
    data: doc,
    isPending: loading,
    isError,
    error: loadError,
  } = useDocument(id);

  const deleteMutation = useDeleteDocumentMutation();
  const { showToast } = useToast();
  const deleteBusy = useDelayedBusy(deleteMutation.isPending);

  const queryError =
    isError && loadError ? formatQueryError(loadError) : null;
  const error = actionError ?? queryError;

  const showInterviewPrep =
    doc?.status === "ready" && doc?.doc_domain === "job_description";

  const tabSyncRef = useRef<{ docId: string; tabQuery: string | null } | null>(
    null
  );

  useEffect(() => {
    setTab("chat");
    tabSyncRef.current = null;
  }, [id]);

  useEffect(() => {
    if (!showInterviewPrep || !doc || doc.id !== id) return;
    const tabQuery = searchParams.get("tab");
    const prev = tabSyncRef.current;
    if (
      prev &&
      prev.docId === doc.id &&
      prev.tabQuery === tabQuery
    ) {
      return;
    }
    tabSyncRef.current = { docId: doc.id, tabQuery };
    const t = (tabQuery || "").toLowerCase();
    if (t === "interview") setTab("interview");
    else if (t === "chat" || t === "ask") setTab("chat");
  }, [
    id,
    showInterviewPrep,
    doc,
    searchParams,
  ]);

  const handleDelete = () => {
    if (!doc || !confirm(`Delete "${doc.filename}"? This cannot be undone.`)) return;
    setActionError(null);
    deleteMutation.mutate(id, {
      onSuccess: () => {
        showToast({ tone: "success", message: `Deleted "${doc.filename}".` });
        router.push("/dashboard");
      },
      onError: (e) => {
        const message =
          e instanceof ApiError
            ? String(e.detail || e.message)
            : "Failed to delete document";
        setActionError(message);
        showToast({ tone: "error", message });
      },
    });
  };

  return (
    <GradientShell>
      {loading && (
        <div className="flex justify-center py-20">
          <LoadingCenter message="Loading document…" />
        </div>
      )}

      {error && !doc && !loading && (
        <div
          className="dashboard-card w-full max-w-2xl border-red-200/50 bg-red-50/60 p-5 text-red-700"
          role="alert"
        >
          {error}
          <Link
            href="/dashboard"
            className="mt-4 block text-sm font-medium text-orange-600 hover:text-orange-700 focus:outline-none focus-visible:underline"
          >
            ← Back to dashboard
          </Link>
        </div>
      )}

      {doc && !loading && (
        <div className="dashboard-card flex flex-col overflow-hidden">
          {actionError && (
            <div
              className="border-b border-red-100 bg-red-50/80 px-6 py-3 text-sm text-red-700 sm:px-8"
              role="alert"
            >
              {actionError}
            </div>
          )}

          {/* Header row */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-100/80 px-6 py-4 sm:px-8">
            <div className="flex min-w-0 flex-1 items-center gap-3">
              <Link
                href="/dashboard"
                className="shrink-0 text-sm font-medium text-zenodrift-text-muted transition-colors duration-200 hover:text-zenodrift-text focus:outline-none focus-visible:ring-2 focus-visible:ring-zenodrift-accent focus-visible:ring-offset-2 focus-visible:ring-offset-white focus-visible:rounded"
              >
                ← Dashboard
              </Link>
              <h1
                className="min-w-0 truncate text-lg font-semibold text-zenodrift-text-strong"
                title={doc.filename}
              >
                {doc.filename}
              </h1>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleDelete}
                disabled={deleteBusy}
                className="rounded-lg border border-red-200 px-3 py-1.5 text-sm font-medium text-red-600 transition-colors hover:bg-red-50 disabled:opacity-50"
              >
                {deleteBusy ? "Deleting…" : "Delete"}
              </button>
              <span
                className={`shrink-0 rounded-full px-2.5 py-1 text-xs font-medium ${
                  STATUS_STYLES[doc.status] ??
                  "bg-slate-100 text-zenodrift-text ring-1 ring-slate-200/60"
                }`}
              >
                {STATUS_LABELS[doc.status] ?? doc.status}
              </span>
              {doc.error_message && (
                <span
                  className="max-w-[180px] truncate text-xs text-red-600"
                  title={doc.error_message}
                >
                  {doc.error_message}
                </span>
              )}
            </div>
          </div>

          {/* Tabs */}
          <div className="mt-4 flex gap-0 border-b border-neutral-100/80 px-6 sm:mt-5 sm:px-8">
            <button
              onClick={() => setTab("chat")}
              role="tab"
              aria-selected={tab === "chat"}
              aria-controls="chat-panel"
              id="chat-tab"
              className={`${TAB_BASE} -mb-px ${
                tab === "chat"
                  ? "border-b-2 border-zenodrift-accent text-zenodrift-text-strong"
                  : "border-b-2 border-transparent text-zenodrift-text-muted hover:text-zenodrift-text"
              }`}
            >
              Ask
            </button>
            {showInterviewPrep && (
              <button
                onClick={() => setTab("interview")}
                role="tab"
                aria-selected={tab === "interview"}
                aria-controls="interview-panel"
                id="interview-tab"
                className={`${TAB_BASE} -mb-px ${
                  tab === "interview"
                    ? "border-b-2 border-zenodrift-accent text-zenodrift-text-strong"
                    : "border-b-2 border-transparent text-zenodrift-text-muted hover:text-zenodrift-text"
                }`}
              >
                Interview Prep
              </button>
            )}
          </div>

          {/* Tab content area */}
          <div className="px-6 pt-6 sm:px-8">
            {doc.status !== "ready" && (
              <div className="mb-6 rounded-xl bg-amber-50/80 px-4 py-3 text-sm text-amber-800 shadow-sm ring-1 ring-amber-200/50">
                {doc.status === "processing" && (
                  <p>Document is being processed. This page will update automatically.</p>
                )}
                {doc.status === "failed" && doc.error_message && (
                  <p>{doc.error_message}</p>
                )}
                {(doc.status === "uploaded" || doc.status === "pending") && (
                  <p>Go to the dashboard and click Process.</p>
                )}
              </div>
            )}

            {tab === "chat" && doc.status === "ready" && (
              <div id="chat-panel" role="tabpanel" aria-labelledby="chat-tab">
                <ChatTab documentId={id} />
              </div>
            )}
            {tab === "interview" && showInterviewPrep && doc && (
              <div
                id="interview-panel"
                role="tabpanel"
                aria-labelledby="interview-tab"
              >
                <InterviewFocusMode
                  documentId={id}
                  documentFilename={doc.filename}
                />
              </div>
            )}
          </div>
        </div>
      )}
    </GradientShell>
  );
}

export default function DocumentPage() {
  return (
    <Suspense
      fallback={
        <GradientShell>
          <div className="flex justify-center py-20">
            <LoadingCenter message="Loading…" label="Loading page" />
          </div>
        </GradientShell>
      }
    >
      <DocumentPageContent />
    </Suspense>
  );
}

function ChatTab({ documentId }: { documentId: string }) {
  const [question, setQuestion] = useState("");
  const [answerData, setAnswerData] = useState<AskResponse | null>(null);
  const [askError, setAskError] = useState<string | null>(null);
  const askMutation = useAskQuestionMutation(documentId);

  const handleAsk = (e: React.FormEvent) => {
    e.preventDefault();
    const q = question.trim();
    if (!q || askMutation.isPending) return;
    setAskError(null);
    setAnswerData(null);
    askMutation.mutate(q, {
      onSuccess: (res) => setAnswerData(res),
      onError: (err) => {
        setAskError(
          err instanceof ApiError
            ? String(err.detail || err.message)
            : "Request failed"
        );
      },
    });
  };

  const structuredAnswer = answerData
    ? parseAskStructuredAnswer(answerData.answer)
    : null;
  const citationsByLabel = new Map(
    (answerData?.citations ?? [])
      .filter((c) => c.label)
      .map((c) => [String(c.label), c])
  );

  return (
    <div className="space-y-6">
      <form onSubmit={handleAsk}>
        <div className="flex gap-3">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Try: salary range, must-have skills, day-to-day work, remote policy…"
            disabled={askMutation.isPending}
            className="flex-1 rounded-xl border border-slate-200 bg-white/80 px-4 py-3 text-zenodrift-text-strong shadow-sm ring-1 ring-slate-200/60 placeholder-zenodrift-text-muted transition-colors focus:border-orange-400 focus:outline-none focus:ring-2 focus:ring-orange-400/20 disabled:opacity-70"
          />
          <button
            type="submit"
            disabled={askMutation.isPending || !question.trim()}
            className="rounded-xl bg-slate-900 px-6 py-3 font-medium text-white shadow-sm transition-colors hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-orange-400 focus:ring-offset-2 disabled:opacity-50"
          >
            {askMutation.isPending ? "..." : "Ask"}
          </button>
        </div>
      </form>

      {askError && (
        <div
          className="rounded-xl bg-red-50/80 px-4 py-3 text-red-700 shadow-sm ring-1 ring-red-200/50"
          role="alert"
        >
          {askError}
        </div>
      )}

      {answerData && (
        <section className="pt-4">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-zenodrift-text-muted">
            Answer
          </h2>
          {structuredAnswer ? (
            <AskAnswerDisplay data={structuredAnswer} />
          ) : (
            <div className="prose prose-slate max-w-none whitespace-pre-wrap text-zenodrift-text">
              {answerData.answer.split(/(\[p?\d+-c\d+\])/g).map((part, i) => {
                const m = part.match(/^\[(p?\d+-c\d+)\]$/);
                if (m) {
                  const label = m[1].startsWith("p") ? m[1] : `p${m[1]}`;
                  const citation = citationsByLabel.get(label);
                  return (
                    <sup
                      key={i}
                      className="cursor-help font-medium text-orange-600"
                      title={citation ? `Page ${citation.page_number}: ${citation.snippet}` : undefined}
                    >
                      {part}
                    </sup>
                  );
                }
                return <span key={i}>{part}</span>;
              })}
            </div>
          )}
          {answerData.citations.length > 0 && (
            <div className="mt-6 border-t border-slate-200/80 pt-6">
              <h3 className="mb-2 text-sm font-semibold text-zenodrift-text">
                Citations
              </h3>
              <ul className="space-y-2">
                {answerData.citations.map((c) => (
                  <li
                    key={c.label ?? c.chunk_id}
                    className="rounded-lg bg-slate-50/80 px-3 py-2 text-sm text-zenodrift-text"
                  >
                    <span className="font-mono text-xs text-orange-600">
                      [{c.label ?? c.chunk_id}]
                    </span>{" "}
                    Page {c.page_number}: {c.snippet}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
