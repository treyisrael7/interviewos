"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  listDocuments,
  getDocument,
  presign,
  uploadToPresignedUrl,
  confirmUpload,
  ingestDocument,
  deleteDocument,
  deleteAllDocuments,
} from "@/lib/api";
import { queryKeys } from "@/lib/query-keys";

export const DOCUMENT_POLL_INTERVAL_MS = 2000;

function documentsRefetchInterval(data: unknown): number | false {
  if (!Array.isArray(data)) return false;
  return data.some((d) => d.status === "processing")
    ? DOCUMENT_POLL_INTERVAL_MS
    : false;
}

function documentRefetchInterval(data: unknown): number | false {
  if (!data || typeof data !== "object") return false;
  return (data as { status: string }).status === "processing"
    ? DOCUMENT_POLL_INTERVAL_MS
    : false;
}

type UseDocumentsOptions = {
  enabled?: boolean;
  refetchOnMount?: boolean | "always";
};

export function useDocuments(options?: UseDocumentsOptions) {
  return useQuery({
    queryKey: queryKeys.documents(),
    queryFn: listDocuments,
    enabled: options?.enabled ?? true,
    refetchOnMount: options?.refetchOnMount,
    refetchInterval: (q) => documentsRefetchInterval(q.state.data),
  });
}

type UseDocumentOptions = {
  enabled?: boolean;
};

export function useDocument(id: string | undefined, options?: UseDocumentOptions) {
  return useQuery({
    queryKey: queryKeys.document(id!),
    queryFn: () => getDocument(id!),
    enabled: (options?.enabled ?? true) && Boolean(id),
    refetchInterval: (q) => documentRefetchInterval(q.state.data),
  });
}

export function useUploadJobDescriptionMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (file: File) => {
      const { document_id, s3_key, upload_url } = await presign(
        file.name,
        file.size
      );
      await uploadToPresignedUrl(upload_url, file);
      await confirmUpload(document_id, s3_key);
      await ingestDocument(document_id);
      return document_id;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.documents() });
    },
  });
}

export function useIngestDocumentMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (documentId: string) => ingestDocument(documentId),
    onSuccess: (_, documentId) => {
      qc.invalidateQueries({ queryKey: queryKeys.documents() });
      qc.invalidateQueries({ queryKey: queryKeys.document(documentId) });
    },
  });
}

export function useDeleteDocumentMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (documentId: string) => deleteDocument(documentId),
    onSuccess: (_, documentId) => {
      qc.invalidateQueries({ queryKey: queryKeys.documents() });
      qc.removeQueries({ queryKey: queryKeys.document(documentId) });
    },
  });
}

export function useDeleteAllDocumentsMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => deleteAllDocuments(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.documents() });
      qc.removeQueries({ queryKey: ["document"], exact: false });
    },
  });
}
