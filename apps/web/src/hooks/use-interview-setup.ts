"use client";

import {
  useMutation,
  useQuery,
  useQueryClient,
  type QueryClient,
} from "@tanstack/react-query";
import {
  listSources,
  addTextSource,
  addSourceFromUrl,
  generateInterview,
  type InterviewGenerateOverrides,
} from "@/lib/api";
import { queryKeys } from "@/lib/query-keys";

type Difficulty = "junior" | "mid" | "senior";

export type AddInterviewSourcePayload =
  | {
      kind: "text";
      sourceType: "resume" | "company" | "notes";
      content: string;
      title?: string;
    }
  | { kind: "url"; url: string; title?: string };

function invalidateDocumentTree(qc: QueryClient, documentId: string) {
  qc.invalidateQueries({ queryKey: queryKeys.documentSources(documentId) });
  qc.invalidateQueries({ queryKey: queryKeys.document(documentId) });
  qc.invalidateQueries({ queryKey: queryKeys.documents() });
}

export function useDocumentSources(documentId: string) {
  return useQuery({
    queryKey: queryKeys.documentSources(documentId),
    queryFn: () => listSources(documentId),
    enabled: Boolean(documentId),
  });
}

export function useAddInterviewSourceMutation(documentId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (payload: AddInterviewSourcePayload) => {
      if (payload.kind === "text") {
        return addTextSource(
          documentId,
          payload.sourceType,
          payload.content,
          payload.title
        );
      }
      return addSourceFromUrl(documentId, payload.url, payload.title);
    },
    onSuccess: () => invalidateDocumentTree(qc, documentId),
  });
}

export function useGenerateInterviewMutation(documentId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (args: {
      difficulty: Difficulty;
      numQuestions: number;
      overrides?: InterviewGenerateOverrides;
    }) =>
      generateInterview(
        documentId,
        args.difficulty,
        args.numQuestions,
        args.overrides
      ),
    onSuccess: () => invalidateDocumentTree(qc, documentId),
  });
}
