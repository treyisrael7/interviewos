"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { GradientShell } from "@/components/GradientShell";
import { LoadingCenter } from "@/components/ui/loading";
import { InterviewSessionView } from "@/components/interview/InterviewSessionView";
import { formatQueryError } from "@/lib/query-error";
import { useInterviewSession } from "@/hooks/use-interview-session";
import { useDocument } from "@/hooks/use-documents";

export default function InterviewSessionPage() {
  const params = useParams();
  const sessionId = params.sessionId as string;

  const {
    data: session,
    isPending: sessionLoading,
    isError: sessionIsError,
    error: sessionError,
  } = useInterviewSession(sessionId);

  const { data: docMeta } = useDocument(session?.document_id, {
    enabled: Boolean(session?.document_id),
  });

  const documentFilename = docMeta?.filename ?? "Job Description";

  const queryError =
    sessionIsError && sessionError
      ? formatQueryError(sessionError)
      : null;

  const loading = sessionLoading;

  return (
    <GradientShell fillViewport>
      {loading && (
        <div className="flex flex-1 items-center justify-center py-8">
          <LoadingCenter
            variant="light"
            message="Loading session…"
            label="Loading session"
          />
        </div>
      )}

      {queryError && !session && (
        <div
          className="dashboard-card w-full max-w-md border-red-200/50 bg-red-50/60 p-5 text-red-700"
          role="alert"
        >
          {queryError}
          <Link
            href="/dashboard"
            className="mt-4 block text-sm font-medium text-zenodrift-accent hover:text-zenodrift-accent-hover focus:outline-none focus-visible:underline"
          >
            ← Back to dashboard
          </Link>
        </div>
      )}

      {session && !loading && (
        <InterviewSessionView
          documentId={session.document_id}
          documentFilename={documentFilename}
          session={{
            sessionId: session.id,
            questions: session.questions,
            mode: session.mode.charAt(0).toUpperCase() + session.mode.slice(1),
            difficulty:
              session.difficulty.charAt(0).toUpperCase() +
              session.difficulty.slice(1),
            performanceProfile: session.performance_profile ?? null,
            adaptiveFocusLabel: session.adaptive_focus_label ?? null,
          }}
          onNewSessionHref={`/interview/setup/${session.document_id}`}
          backHref="/dashboard"
        />
      )}
    </GradientShell>
  );
}
