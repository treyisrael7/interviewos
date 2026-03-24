"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { evaluateAnswer } from "@/lib/api";
import { queryKeys } from "@/lib/query-keys";

export function useEvaluateAnswerMutation(documentId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (args: { questionId: string; answerText: string }) =>
      evaluateAnswer(documentId, args.questionId, args.answerText),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.document(documentId) });
      qc.invalidateQueries({ queryKey: queryKeys.documents() });
    },
  });
}
