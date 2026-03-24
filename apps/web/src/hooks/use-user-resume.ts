"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  getUserResume,
  presignUserResume,
  uploadToPresignedUrl,
  confirmUserResume,
  deleteUserResume,
} from "@/lib/api";
import { queryKeys } from "@/lib/query-keys";

export function useUserResume() {
  return useQuery({
    queryKey: queryKeys.userResume(),
    queryFn: getUserResume,
  });
}

export function useUploadUserResumeMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (file: File) => {
      const { s3_key, upload_url } = await presignUserResume(file.name, file.size);
      await uploadToPresignedUrl(upload_url, file);
      return confirmUserResume(s3_key);
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.userResume() });
      qc.invalidateQueries({ queryKey: queryKeys.documents() });
    },
  });
}

export function useDeleteUserResumeMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => deleteUserResume(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.userResume() });
      qc.invalidateQueries({ queryKey: queryKeys.documents() });
    },
  });
}
