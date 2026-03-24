"use client";

import { useQuery } from "@tanstack/react-query";
import { getInterviewSession } from "@/lib/api";
import { queryKeys } from "@/lib/query-keys";

export function useInterviewSession(sessionId: string | undefined) {
  return useQuery({
    queryKey: queryKeys.interview(sessionId!),
    queryFn: () => getInterviewSession(sessionId!),
    enabled: Boolean(sessionId),
  });
}
