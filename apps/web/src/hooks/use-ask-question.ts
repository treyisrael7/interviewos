"use client";

import { useMutation } from "@tanstack/react-query";
import { ask } from "@/lib/api";

export function useAskQuestionMutation(documentId: string) {
  return useMutation({
    mutationFn: (question: string) => ask(documentId, question),
  });
}
