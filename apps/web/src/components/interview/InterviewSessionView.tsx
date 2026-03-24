"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ApiError, type InterviewQuestion, type InterviewEvaluateResponse } from "@/lib/api";
import { useEvaluateAnswerMutation } from "@/hooks/use-interview-evaluate";
import { EvaluationDrawer } from "./EvaluationDrawer";
import { ReferenceDrawer } from "./ReferenceDrawer";

function formatQuestionChip(
  type: string | undefined,
  focusArea: string | undefined,
  competencyLabel?: string | null
): string {
  // Prefer competency chip when available (replaces "TECHNICAL" / type+focus_area display)
  if (competencyLabel?.trim()) {
    return competencyLabel;
  }
  const typeLabel =
    type === "role_specific"
      ? "Role-specific"
      : type === "scenario"
        ? "Scenario"
        : type === "behavioral"
          ? "Behavioral"
          : type
            ? type.replace(/_/g, " ")
            : "";
  if (focusArea) {
    return typeLabel ? `${typeLabel} • ${focusArea}` : focusArea;
  }
  return typeLabel || "";
}

export interface SessionData {
  sessionId: string;
  questions: InterviewQuestion[];
  mode: string;
  difficulty: string;
}

interface InterviewSessionViewProps {
  documentId: string;
  documentFilename: string;
  session: SessionData;
  onNewSessionHref?: string;
  /** Optional back link (e.g. Dashboard) */
  backHref?: string;
}

export function InterviewSessionView({
  documentId,
  documentFilename,
  session,
  backHref,
}: InterviewSessionViewProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answerText, setAnswerText] = useState("");
  const [evalResult, setEvalResult] = useState<InterviewEvaluateResponse | null>(null);
  const [evalError, setEvalError] = useState<string | null>(null);
  const evaluateMutation = useEvaluateAnswerMutation(documentId);
  const [evalDrawerOpen, setEvalDrawerOpen] = useState(false);
  const [referenceDrawerOpen, setReferenceDrawerOpen] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const questionScrollRef = useRef<HTMLDivElement>(null);

  const currentQuestion = session.questions[currentIndex];
  const canNext = currentIndex < session.questions.length - 1;
  const docName = documentFilename.replace(/\.pdf$/i, "");

  const handleSubmit = () => {
    if (!currentQuestion || !answerText.trim() || evaluateMutation.isPending) return;
    setEvalError(null);
    setEvalResult(null);
    evaluateMutation.mutate(
      { questionId: currentQuestion.id, answerText: answerText.trim() },
      {
        onSuccess: (res) => {
          setEvalResult(res);
          setEvalDrawerOpen(true);
        },
        onError: (e) => {
          setEvalError(
            e instanceof ApiError
              ? String(e.detail || e.message)
              : "Evaluation failed"
          );
        },
      }
    );
  };

  const handleNext = () => {
    setEvalResult(null);
    setEvalDrawerOpen(false);
    setAnswerText("");
    setEvalError(null);
    setReferenceDrawerOpen(false);
    setCurrentIndex((i) => Math.min(i + 1, session.questions.length - 1));
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [currentIndex, answerText]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Focus Panel: frosted glass, viewport-constrained */}
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-3xl border border-white/25 bg-white/55 shadow-[0_8px_32px_rgba(0,0,0,0.06),0_1px_3px_rgba(0,0,0,0.03)] backdrop-blur-xl">
        {/* Top bar - minimal */}
        <div className="flex shrink-0 items-center justify-between gap-2 border-b border-white/25 px-3 py-2 sm:px-4">
          <span
            className="truncate text-sm font-medium text-zenodrift-text-strong"
            title={docName}
          >
            {docName}
          </span>
          {backHref && (
            <Link
              href={backHref}
              className="text-sm text-red-600 hover:text-red-700 hover:underline"
            >
              End Session
            </Link>
          )}
        </div>

        {/* Question area - scrollable */}
        <div
          ref={questionScrollRef}
          className="min-h-0 flex-1 overflow-y-auto px-3 py-4 sm:px-4"
        >
          <div className="flex flex-col gap-3">
            {currentQuestion && (
              <div className="flex justify-start">
                <div className="max-w-[85%] rounded-2xl rounded-tl-sm bg-white/70 px-3 py-2.5 shadow-sm backdrop-blur-sm sm:max-w-[75%]">
                  <p className="text-zenodrift-text leading-relaxed">
                    {(currentQuestion.type || currentQuestion.focus_area || currentQuestion.competency_label) && (
                      <span className="mr-2 inline-flex rounded-md bg-white/60 px-2 py-0.5 text-xs capitalize text-zenodrift-text-muted backdrop-blur-sm">
                        {formatQuestionChip(currentQuestion.type, currentQuestion.focus_area, currentQuestion.competency_label)}
                      </span>
                    )}
                    {currentQuestion.question}
                  </p>
                </div>
              </div>
            )}
            {answerText.trim() && (
              <div className="flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-zenodrift-accent/15 px-3 py-2.5 text-zenodrift-text">
                  <p className="whitespace-pre-wrap text-sm leading-relaxed">{answerText}</p>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Composer - inside scroll? No - user said "composer: shrink-0 (sticky within the panel, NOT causing page scroll)" */}
        </div>

        {/* Composer - shrink-0, sticky at bottom of panel */}
        {currentQuestion && (
          <div className="shrink-0 border-t border-white/25 px-3 py-3 sm:px-4">
            <div className="flex flex-col gap-2">
              <textarea
                value={answerText}
                onChange={(e) => setAnswerText(e.target.value)}
                placeholder="Type your answer here…"
                rows={3}
                disabled={evaluateMutation.isPending}
                className="w-full resize-none rounded-xl border-0 bg-white/60 px-3 py-2.5 text-sm text-zenodrift-text shadow-inner placeholder-neutral-400 backdrop-blur-sm focus:bg-white/80 focus:ring-2 focus:ring-zenodrift-accent/25 focus:outline-none disabled:opacity-70"
              />
              <div className="flex items-center justify-between gap-2">
                <button
                  onClick={() => setReferenceDrawerOpen(true)}
                  className="text-sm text-zenodrift-text-muted hover:text-zenodrift-text"
                >
                  Reference
                </button>
                <button
                  onClick={handleSubmit}
                  disabled={evaluateMutation.isPending || !answerText.trim()}
                  className="rounded-xl bg-zenodrift-accent px-5 py-2 text-sm font-semibold text-white shadow-sm hover:bg-zenodrift-accent-hover focus:outline-none focus:ring-2 focus:ring-zenodrift-accent disabled:opacity-50"
                >
                  {evaluateMutation.isPending ? "Evaluating…" : "Submit Answer"}
                </button>
              </div>
              {evalError && (
                <p className="text-xs text-red-600" role="alert">
                  {evalError}
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      <EvaluationDrawer
        isOpen={evalDrawerOpen}
        onClose={() => setEvalDrawerOpen(false)}
        result={evalResult}
        onRetry={() => setEvalResult(null)}
        onNextQuestion={handleNext}
        canNext={canNext}
      />

      <ReferenceDrawer
        isOpen={referenceDrawerOpen}
        onClose={() => setReferenceDrawerOpen(false)}
        question={currentQuestion ?? null}
        lastEval={evalResult}
        initialTab="evidence"
      />
    </div>
  );
}
