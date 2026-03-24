/** Stable query keys for TanStack Query (see useDocuments, useDocument, useInterviewSession). */
export const queryKeys = {
  documents: () => ["documents"] as const,
  document: (id: string) => ["document", id] as const,
  interview: (sessionId: string) => ["interview", sessionId] as const,
  documentSources: (documentId: string) =>
    ["document", documentId, "sources"] as const,
  userResume: () => ["userResume"] as const,
};
