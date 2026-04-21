"use client";

import { useMutation } from "@tanstack/react-query";
import { generateStudyPlan } from "@/lib/api";

export function useStudyPlanMutation(documentId: string) {
  return useMutation({
    mutationFn: async (opts: { days: number; focus?: string }) =>
      generateStudyPlan({
        documentId,
        days: opts.days,
        focus: opts.focus,
      }),
  });
}

