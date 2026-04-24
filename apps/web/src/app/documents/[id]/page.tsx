"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { GradientShell } from "@/components/GradientShell";
import { LoadingCenter, LoadingRow } from "@/components/ui/loading";
import { InterviewFocusMode } from "@/components/interview/InterviewFocusMode";
import {
  ApiError,
  getAnalyzeFitLatest,
  parseAskStructuredAnswer,
  type AskResponse,
  type StudyPlanResult,
} from "@/lib/api";
import { AnalyzeFitDisplay } from "@/components/AnalyzeFitDisplay";
import { AskAnswerDisplay } from "@/components/AskAnswerDisplay";
import { formatQueryError } from "@/lib/query-error";
import { useAnalyzeFitMutation } from "@/hooks/use-analyze-fit";
import { useStudyPlanMutation } from "@/hooks/use-study-plan";
import { queryKeys } from "@/lib/query-keys";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useDocument, useDeleteDocumentMutation } from "@/hooks/use-documents";
import { useAskQuestionMutation } from "@/hooks/use-ask-question";
import { useUserResume } from "@/hooks/use-user-resume";

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

type Tab = "chat" | "fit" | "study" | "interview";

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
  const { data: resumeStatus, isPending: resumeLoading } = useUserResume();

  const deleteMutation = useDeleteDocumentMutation();

  const queryError =
    isError && loadError ? formatQueryError(loadError) : null;
  const error = actionError ?? queryError;

  const showInterviewPrep =
    doc?.status === "ready" && doc?.doc_domain === "job_description";
  const hasAccountResume = Boolean(
    resumeStatus?.has_resume && resumeStatus.document_id
  );
  const showAnalyzeFit = showInterviewPrep && hasAccountResume;

  const tabSyncRef = useRef<{ docId: string; tabQuery: string | null } | null>(
    null
  );

  useEffect(() => {
    setTab("chat");
    tabSyncRef.current = null;
  }, [id]);

  useEffect(() => {
    if (!showInterviewPrep || !doc || doc.id !== id) return;
    if (resumeLoading) return;
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
    if (t === "fit") {
      setTab(hasAccountResume ? "fit" : "chat");
    } else if (t === "study") {
      setTab("study");
    } else if (t === "interview") setTab("interview");
    else if (t === "chat" || t === "ask") setTab("chat");
  }, [
    id,
    showInterviewPrep,
    doc,
    searchParams,
    resumeLoading,
    hasAccountResume,
  ]);

  useEffect(() => {
    if (tab === "fit" && !hasAccountResume && !resumeLoading) {
      setTab("chat");
    }
  }, [tab, hasAccountResume, resumeLoading]);

  const handleDelete = () => {
    if (!doc || !confirm(`Delete "${doc.filename}"? This cannot be undone.`)) return;
    setActionError(null);
    deleteMutation.mutate(id, {
      onSuccess: () => router.push("/dashboard"),
      onError: (e) => {
        setActionError(
          e instanceof ApiError
            ? String(e.detail || e.message)
            : "Failed to delete document"
        );
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
                disabled={deleteMutation.isPending}
                className="rounded-lg border border-red-200 px-3 py-1.5 text-sm font-medium text-red-600 transition-colors hover:bg-red-50 disabled:opacity-50"
              >
                {deleteMutation.isPending ? "Deleting…" : "Delete"}
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
            {showAnalyzeFit && (
              <button
                onClick={() => setTab("fit")}
                role="tab"
                aria-selected={tab === "fit"}
                aria-controls="fit-panel"
                id="fit-tab"
                className={`${TAB_BASE} -mb-px ${
                  tab === "fit"
                    ? "border-b-2 border-zenodrift-accent text-zenodrift-text-strong"
                    : "border-b-2 border-transparent text-zenodrift-text-muted hover:text-zenodrift-text"
                }`}
              >
                Analyze Fit
              </button>
            )}
            {showInterviewPrep && (
              <button
                onClick={() => setTab("study")}
                role="tab"
                aria-selected={tab === "study"}
                aria-controls="study-panel"
                id="study-tab"
                className={`${TAB_BASE} -mb-px ${
                  tab === "study"
                    ? "border-b-2 border-zenodrift-accent text-zenodrift-text-strong"
                    : "border-b-2 border-transparent text-zenodrift-text-muted hover:text-zenodrift-text"
                }`}
              >
                Study Plan
              </button>
            )}
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
            {tab === "fit" && showAnalyzeFit && doc.status === "ready" && (
              <div id="fit-panel" role="tabpanel" aria-labelledby="fit-tab">
                <AnalyzeFitTab documentId={id} />
              </div>
            )}
            {tab === "study" && showInterviewPrep && doc.status === "ready" && (
              <div id="study-panel" role="tabpanel" aria-labelledby="study-tab">
                <StudyPlanTab documentId={id} />
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

function AnalyzeFitTab({ documentId }: { documentId: string }) {
  const queryClient = useQueryClient();
  const { data: resume, isPending: resumeLoading } = useUserResume();
  const [focusQuestion, setFocusQuestion] = useState("");
  const [fitError, setFitError] = useState<string | null>(null);
  const fitMutation = useAnalyzeFitMutation(documentId);

  const hasResume = Boolean(resume?.has_resume && resume.document_id);
  const resumeId = resume?.document_id ?? "";

  const latestQuery = useQuery({
    queryKey: queryKeys.analyzeFitLatest(documentId, resumeId),
    queryFn: () => getAnalyzeFitLatest(documentId, resumeId),
    enabled: hasResume && Boolean(resumeId),
    staleTime: 0,
  });

  const saved = latestQuery.data;
  const displayAnalysis =
    saved?.has_analysis && saved.analysis ? saved.analysis : null;

  const handleAnalyze = () => {
    if (!hasResume || fitMutation.isPending) return;
    setFitError(null);
    fitMutation.mutate(
      { question: focusQuestion.trim() || undefined },
      {
        onSuccess: (data) => {
          queryClient.setQueryData(
            queryKeys.analyzeFitLatest(documentId, resumeId),
            {
              has_analysis: true,
              analysis: data,
              created_at: new Date().toISOString(),
              cache_hit_default_question: !focusQuestion.trim(),
            }
          );
        },
        onError: (err) => {
          if (err instanceof Error && err.message === "RESUME_REQUIRED") {
            setFitError(
              "Add your resume on the dashboard first. We need it to compare you to this job."
            );
            return;
          }
          setFitError(
            err instanceof ApiError
              ? String(err.detail || err.message)
              : "Could not analyze fit"
          );
        },
      }
    );
  };

  const savedLabel =
    saved?.created_at &&
    (() => {
      try {
        return new Date(saved.created_at).toLocaleString(undefined, {
          dateStyle: "medium",
          timeStyle: "short",
        });
      } catch {
        return saved.created_at;
      }
    })();

  return (
    <div className="space-y-6 pb-8">
      <p className="text-sm leading-relaxed text-zenodrift-text">
        We line this job up with your{" "}
        <strong>account resume</strong>: what matches, what&apos;s missing, a fit score, and
        specific tweaks you could make. Your last run stays put until the job or resume changes
        (no surprise API cost). Hit Refresh after you change something.
      </p>

      {resumeLoading ? (
        <LoadingRow message="Checking resume…" />
      ) : !hasResume ? (
        <div
          className="rounded-xl border border-amber-200/80 bg-amber-50/60 px-4 py-3 text-sm text-amber-950"
          role="status"
        >
          <p className="font-medium text-amber-950">No account resume yet</p>
          <p className="mt-1 text-amber-900/90">
            Upload a PDF on the dashboard so we can compare it to this job.
          </p>
          <Link
            href="/dashboard"
            className="mt-3 inline-block text-sm font-medium text-orange-700 underline-offset-2 hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-orange-400 focus-visible:rounded"
          >
            Go to dashboard →
          </Link>
        </div>
      ) : (
        <>
          {latestQuery.isPending && (
            <LoadingRow message="Loading saved analysis…" size="sm" />
          )}
          {displayAnalysis && savedLabel && !latestQuery.isPending && (
            <div
              className="rounded-xl border border-slate-200/80 bg-slate-50/80 px-4 py-3 text-sm text-zenodrift-text"
              role="status"
            >
              <p className="font-medium text-zenodrift-text-strong">Saved analysis</p>
              <p className="mt-1 text-xs text-zenodrift-text-muted">
                From {savedLabel}. Re-running uses tokens only if something changed or you
                add an optional focus below.
              </p>
            </div>
          )}
          <div className="space-y-2">
            <label
              htmlFor="analyze-fit-focus"
              className="block text-sm font-medium text-zenodrift-text-strong"
            >
              Optional focus{" "}
              <span className="font-normal text-zenodrift-text-muted">
                (narrows retrieval; leave blank for a full comparison)
              </span>
            </label>
            <textarea
              id="analyze-fit-focus"
              value={focusQuestion}
              onChange={(e) => setFocusQuestion(e.target.value)}
              rows={3}
              disabled={fitMutation.isPending}
              placeholder="e.g. Emphasize FP&A, budgeting, and executive reporting…"
              className="w-full rounded-xl border border-slate-200 bg-white/80 px-4 py-3 text-sm text-zenodrift-text shadow-sm ring-1 ring-slate-200/60 placeholder-zenodrift-text-muted focus:border-orange-400 focus:outline-none focus:ring-2 focus:ring-orange-400/20 disabled:opacity-70"
            />
          </div>
          <button
            type="button"
            onClick={handleAnalyze}
            disabled={fitMutation.isPending}
            className="rounded-xl bg-slate-900 px-6 py-3 text-sm font-medium text-white shadow-sm transition-colors hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-orange-400 focus:ring-offset-2 disabled:opacity-50"
          >
            {fitMutation.isPending
              ? "Working…"
              : displayAnalysis
                ? "Refresh analysis"
                : "Analyze fit"}
          </button>
        </>
      )}

      {fitError && (
        <div
          className="rounded-xl bg-red-50/80 px-4 py-3 text-sm text-red-700 shadow-sm ring-1 ring-red-200/50"
          role="alert"
        >
          {fitError}
        </div>
      )}

      {displayAnalysis && (
        <section className="border-t border-slate-200/80 pt-6">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-zenodrift-text-muted">
            Analysis
          </h2>
          <AnalyzeFitDisplay data={displayAnalysis} />
        </section>
      )}
    </div>
  );
}

function StudyPlanTab({ documentId }: { documentId: string }) {
  const [days, setDays] = useState(10);
  const [focus, setFocus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [plan, setPlan] = useState<StudyPlanResult | null>(null);
  const mutation = useStudyPlanMutation(documentId);

  const handleGenerate = () => {
    if (mutation.isPending) return;
    setError(null);
    mutation.mutate(
      { days, focus: focus.trim() || undefined },
      {
        onSuccess: (data) => setPlan(data),
        onError: (err) => {
          setError(
            err instanceof ApiError
              ? String(err.detail || err.message)
              : "Could not generate study plan"
          );
        },
      }
    );
  };

  return (
    <div className="space-y-6 pb-8">
      <p className="text-sm leading-relaxed text-zenodrift-text">
        Generate a role-specific prep plan from this JD. Pick a timeline between{" "}
        <strong>7 and 14 days</strong>, then get daily topics, drills, and mock
        interview targets.
      </p>

      <div className="grid gap-4 sm:grid-cols-[180px_1fr]">
        <div className="space-y-2">
          <label
            htmlFor="study-plan-days"
            className="block text-sm font-medium text-zenodrift-text-strong"
          >
            Plan length
          </label>
          <input
            id="study-plan-days"
            type="number"
            min={7}
            max={14}
            value={days}
            onChange={(e) => {
              const parsed = Number(e.target.value);
              if (!Number.isFinite(parsed)) return;
              setDays(Math.max(7, Math.min(14, Math.round(parsed))));
            }}
            disabled={mutation.isPending}
            className="w-full rounded-xl border border-slate-200 bg-white/80 px-3 py-2 text-sm text-zenodrift-text shadow-sm ring-1 ring-slate-200/60 focus:border-orange-400 focus:outline-none focus:ring-2 focus:ring-orange-400/20 disabled:opacity-70"
          />
        </div>
        <div className="space-y-2">
          <label
            htmlFor="study-plan-focus"
            className="block text-sm font-medium text-zenodrift-text-strong"
          >
            Optional focus{" "}
            <span className="font-normal text-zenodrift-text-muted">
              (e.g. system design, leadership stories)
            </span>
          </label>
          <textarea
            id="study-plan-focus"
            value={focus}
            onChange={(e) => setFocus(e.target.value)}
            rows={3}
            disabled={mutation.isPending}
            placeholder="Prioritize architecture and stakeholder communication..."
            className="w-full rounded-xl border border-slate-200 bg-white/80 px-4 py-3 text-sm text-zenodrift-text shadow-sm ring-1 ring-slate-200/60 placeholder-zenodrift-text-muted focus:border-orange-400 focus:outline-none focus:ring-2 focus:ring-orange-400/20 disabled:opacity-70"
          />
        </div>
      </div>

      <button
        type="button"
        onClick={handleGenerate}
        disabled={mutation.isPending}
        className="rounded-xl bg-slate-900 px-6 py-3 text-sm font-medium text-white shadow-sm transition-colors hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-orange-400 focus:ring-offset-2 disabled:opacity-50"
      >
        {mutation.isPending
          ? "Building plan…"
          : plan
            ? "Regenerate plan"
            : "Generate study plan"}
      </button>

      {error && (
        <div
          className="rounded-xl bg-red-50/80 px-4 py-3 text-sm text-red-700 shadow-sm ring-1 ring-red-200/50"
          role="alert"
        >
          {error}
        </div>
      )}

      {plan && (
        <section className="space-y-4 border-t border-slate-200/80 pt-6">
          <div>
            <h2 className="text-base font-semibold text-zenodrift-text-strong">
              {plan.title}
            </h2>
            <p className="mt-1 text-sm text-zenodrift-text">
              {plan.role_title} · {plan.duration_days} days
            </p>
            {plan.summary.trim() ? (
              <p className="mt-2 text-sm leading-relaxed text-zenodrift-text">
                {plan.summary}
              </p>
            ) : null}
          </div>

          <div className="space-y-3">
            {plan.daily_plan.map((day) => (
              <article
                key={day.day}
                className="rounded-xl border border-slate-200/80 bg-white/70 px-4 py-3 shadow-sm"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <p className="text-sm font-semibold text-zenodrift-text-strong">
                    Day {day.day}
                  </p>
                  {day.theme.trim() ? (
                    <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-zenodrift-text">
                      {day.theme}
                    </span>
                  ) : null}
                </div>
                {day.topics.length > 0 && (
                  <p className="mt-2 text-sm text-zenodrift-text">
                    <span className="font-medium text-zenodrift-text-strong">Topics: </span>
                    {day.topics.join(", ")}
                  </p>
                )}
                {day.drills.length > 0 && (
                  <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-zenodrift-text">
                    {day.drills.map((drill, idx) => (
                      <li key={`${day.day}-drill-${idx}`}>{drill}</li>
                    ))}
                  </ul>
                )}
                {day.mock_target.trim() ? (
                  <p className="mt-2 rounded-lg bg-amber-50/80 px-3 py-2 text-xs text-amber-950 ring-1 ring-amber-200/70">
                    <span className="font-semibold uppercase tracking-wide text-amber-900/90">
                      Mock target
                    </span>{" "}
                    {day.mock_target}
                  </p>
                ) : null}
              </article>
            ))}
          </div>
        </section>
      )}
    </div>
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
              {answerData.answer.split(/(\[\d+-c\d+\])/g).map((part, i) => {
                const m = part.match(/^\[(\d+)-c(\d+)\]$/);
                if (m) {
                  const idx = parseInt(m[2], 10) - 1;
                  const citation = answerData!.citations[idx];
                  return (
                    <sup
                      key={i}
                      className="cursor-help font-medium text-orange-600"
                      title={citation?.snippet}
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
                    key={c.chunk_id}
                    className="rounded-lg bg-slate-50/80 px-3 py-2 text-sm text-zenodrift-text"
                  >
                    <span className="font-mono text-xs text-orange-600">
                      [{c.chunk_id}]
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
