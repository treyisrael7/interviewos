"use client";

import { useCallback, useEffect, useRef, useState, useMemo } from "react";
import { useRouter, usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import type { DocumentSummary } from "@/lib/api";
import { useDocuments } from "@/hooks/use-documents";

const STATUS_LABELS: Record<string, string> = {
  pending: "Pending",
  uploaded: "Uploaded",
  processing: "Processing",
  ready: "Ready",
  failed: "Failed",
};

const STATUS_STYLES: Record<string, string> = {
  pending: "text-amber-700 bg-amber-100/80",
  uploaded: "text-blue-700 bg-blue-100/80",
  processing: "text-indigo-700 bg-indigo-100/80 animate-pulse",
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

interface LibraryModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const FOCUSABLE =
  'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

export function LibraryModal({ isOpen, onClose }: LibraryModalProps) {
  const router = useRouter();
  const pathname = usePathname();
  const overlayRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const [search, setSearch] = useState("");
  const [pendingSwitchDoc, setPendingSwitchDoc] = useState<DocumentSummary | null>(null);

  const { data: docs = [], isPending: loading } = useDocuments({
    enabled: isOpen,
    refetchOnMount: "always",
  });

  const isInterviewMode = pathname?.startsWith("/interview/") ?? false;

  const filteredDocs = useMemo(() => {
    if (!search.trim()) return docs;
    const q = search.trim().toLowerCase();
    return docs.filter((d) => d.filename.toLowerCase().includes(q));
  }, [docs, search]);

  useEffect(() => {
    if (isOpen) {
      setSearch("");
      setPendingSwitchDoc(null);
      document.body.style.overflow = "hidden";
      setTimeout(() => searchInputRef.current?.focus(), 0);
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [isOpen]);

  const handleEscape = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (pendingSwitchDoc) {
          setPendingSwitchDoc(null);
        } else {
          onClose();
        }
      }
    },
    [onClose, pendingSwitchDoc]
  );

  useEffect(() => {
    if (isOpen) {
      document.addEventListener("keydown", handleEscape);
    }
    return () => document.removeEventListener("keydown", handleEscape);
  }, [isOpen, handleEscape]);

  const handleOverlayClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === overlayRef.current) {
        setPendingSwitchDoc(null);
        onClose();
      }
    },
    [onClose]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key !== "Tab") return;
      const panel = panelRef.current;
      if (!panel) return;
      const focusable = panel.querySelectorAll<HTMLElement>(FOCUSABLE);
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey) {
        if (document.activeElement === first) {
          e.preventDefault();
          last.focus();
        }
      } else {
        if (document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    },
    []
  );

  const handleUseForInterview = useCallback(
    (doc: DocumentSummary) => {
      if (isInterviewMode) {
        setPendingSwitchDoc(doc);
      } else {
        onClose();
        router.push(`/interview/setup/${doc.id}`);
      }
    },
    [isInterviewMode, onClose, router]
  );

  const handleConfirmSwitch = useCallback(() => {
    if (!pendingSwitchDoc) return;
    const docId = pendingSwitchDoc.id;
    setPendingSwitchDoc(null);
    onClose();
    router.replace(`/interview/setup/${docId}`);
  }, [pendingSwitchDoc, onClose, router]);

  const handleCancelSwitch = useCallback(() => {
    setPendingSwitchDoc(null);
  }, []);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            ref={overlayRef}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-50 bg-slate-900/25 backdrop-blur-sm"
            onClick={handleOverlayClick}
            aria-hidden
          />
          <div
            className="fixed inset-0 z-50 flex items-center justify-center p-4 focus:outline-none"
            role="dialog"
            aria-modal="true"
            aria-labelledby="library-modal-title"
            onKeyDown={handleKeyDown}
          >
            <motion.div
              ref={panelRef}
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
              className="flex max-h-[85vh] w-full max-w-lg flex-col rounded-3xl bg-white shadow-[0_8px_30px_rgb(0,0,0,0.06)]"
            >
              <div className="flex shrink-0 items-center justify-between px-6 py-5">
                <h2 id="library-modal-title" className="text-xl font-semibold text-zenodrift-text-strong">
                  Your Documents
                </h2>
                <button
                  onClick={onClose}
                  className="rounded-full p-2 text-zenodrift-text-muted transition-colors hover:bg-slate-100 hover:text-zenodrift-text focus:outline-none focus:ring-2 focus:ring-orange-400/20 focus:ring-offset-2"
                  aria-label="Close"
                >
                  <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <div className="flex shrink-0 px-6 pb-4">
                <input
                  ref={searchInputRef}
                  type="search"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search documents…"
                  className="w-full rounded-xl bg-slate-50 px-4 py-3 text-zenodrift-text-strong placeholder-zenodrift-text-muted focus:bg-white focus:ring-2 focus:ring-orange-400/20 focus:outline-none"
                  aria-label="Filter documents by name"
                />
              </div>
              <div className="flex-1 overflow-y-auto px-6 pb-6">
                {loading ? (
                  <div className="flex items-center justify-center py-12">
                    <div
                      className="h-8 w-8 animate-spin rounded-full border-2 border-slate-200 border-t-orange-500"
                      aria-hidden
                    />
                  </div>
                ) : filteredDocs.length === 0 ? (
                  <p className="py-8 text-center text-sm text-zenodrift-text-muted">
                    {search.trim() ? "No documents match your search." : "No documents yet."}
                  </p>
                ) : (
                  <ul className="space-y-3">
                    {filteredDocs.map((doc) => {
                      const isProcessing =
                        doc.status === "processing" ||
                        doc.status === "pending" ||
                        doc.status === "uploaded";
                      const isReady = doc.status === "ready";
                      return (
                        <li
                          key={doc.id}
                          className="flex flex-col gap-3 rounded-2xl bg-slate-50/80 p-4 sm:flex-row sm:items-center sm:justify-between sm:gap-4"
                        >
                          <div
                            className="min-w-0 flex-1 cursor-pointer"
                            onClick={() => isReady && handleUseForInterview(doc)}
                            onKeyDown={(e) => {
                              if (isReady && (e.key === "Enter" || e.key === " ")) {
                                e.preventDefault();
                                handleUseForInterview(doc);
                              }
                            }}
                            role={isReady ? "button" : undefined}
                            tabIndex={isReady ? 0 : undefined}
                            aria-label={isReady ? `Open ${doc.filename} for interview setup` : undefined}
                          >
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="truncate font-medium text-zenodrift-text-strong">
                                {doc.filename}
                              </span>
                              <span
                                className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ${
                                  STATUS_STYLES[doc.status] ??
                                  "bg-slate-100 text-zenodrift-text"
                                }`}
                              >
                                {STATUS_LABELS[doc.status] ?? doc.status}
                              </span>
                            </div>
                            {doc.created_at && formatUploadedAt(doc.created_at) && (
                              <div className="mt-1.5 text-xs text-zenodrift-text-muted">
                                {formatUploadedAt(doc.created_at)}
                              </div>
                            )}
                          </div>
                          <div className="shrink-0">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                if (isReady) handleUseForInterview(doc);
                              }}
                              disabled={isProcessing}
                              className="rounded-xl bg-orange-500 px-4 py-2.5 text-sm font-medium text-white shadow-sm transition-colors hover:bg-orange-600 focus:outline-none focus:ring-2 focus:ring-orange-400 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60"
                            >
                              {isProcessing ? "Processing…" : "Use for Interview"}
                            </button>
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>
              {pendingSwitchDoc && (
                <div className="shrink-0 border-t border-slate-100 bg-amber-50/80 px-6 py-4">
                  <p className="mb-3 text-sm font-medium text-zenodrift-text-strong">
                    Switch document?
                  </p>
                  <p className="mb-4 text-sm text-zenodrift-text">
                    You&apos;ll go to Interview Setup for{" "}
                    <span className="font-medium text-zenodrift-text">
                      {pendingSwitchDoc.filename.replace(/\.pdf$/i, "")}
                    </span>
                    . Your current session will not be saved.
                  </p>
                  <div className="flex gap-3">
                    <button
                      onClick={handleCancelSwitch}
                      className="rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-zenodrift-text transition-colors hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-orange-400/20"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleConfirmSwitch}
                      className="rounded-xl bg-orange-500 px-4 py-2.5 text-sm font-medium text-white shadow-sm transition-colors hover:bg-orange-600 focus:outline-none focus:ring-2 focus:ring-orange-400 focus:ring-offset-2"
                    >
                      Switch document
                    </button>
                  </div>
                </div>
              )}
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  );
}
