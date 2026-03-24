"use client";

import { useState } from "react";
import {
  QueryClient,
  QueryClientProvider,
  type QueryClientConfig,
} from "@tanstack/react-query";
import { ApiError, AuthRequiredError } from "@/lib/api";

function shouldRetryQuery(failureCount: number, error: unknown): boolean {
  if (error instanceof AuthRequiredError) return false;
  if (error instanceof ApiError) {
    if (error.status === 408 || error.status === 429) return failureCount < 2;
    if (error.status >= 400 && error.status < 500) return false;
  }
  return failureCount < 2;
}

function createQueryClient(): QueryClient {
  const config: QueryClientConfig = {
    defaultOptions: {
      queries: {
        staleTime: 60 * 1000,
        gcTime: 5 * 60 * 1000,
        retry: shouldRetryQuery,
        refetchOnWindowFocus: true,
        refetchOnReconnect: true,
      },
      mutations: {
        retry: 0,
      },
    },
  };
  return new QueryClient(config);
}

export function QueryProvider({ children }: { children: React.ReactNode }) {
  const [client] = useState(createQueryClient);
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
