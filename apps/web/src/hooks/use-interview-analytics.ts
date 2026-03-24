"use client";

import { useQuery } from "@tanstack/react-query";
import { getInterviewAnalyticsOverview } from "@/lib/api";
import { queryKeys } from "@/lib/query-keys";

export function useInterviewAnalyticsOverview() {
  return useQuery({
    queryKey: queryKeys.interviewAnalyticsOverview(),
    queryFn: getInterviewAnalyticsOverview,
  });
}
