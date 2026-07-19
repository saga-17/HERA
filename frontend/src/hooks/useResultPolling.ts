import { useCallback, useEffect, useRef, useState } from "react";
import { getResult } from "../services/api";
import type { HeraResult } from "../services/types";

interface UsePollingResult {
  result: HeraResult | null;
  loading: boolean;
  error: string | null;
}

export function useResultPolling(resultId: string | null, intervalMs = 2000): UsePollingResult {
  const [result, setResult] = useState<HeraResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<number | null>(null);

  const poll = useCallback(async () => {
    if (!resultId) return;
    try {
      const data = await getResult(resultId);
      setResult(data);
      if (data.pipeline_status.stage === "complete" || data.pipeline_status.stage === "error") {
        setLoading(false);
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch result");
      setLoading(false);
    }
  }, [resultId]);

  useEffect(() => {
    if (!resultId) return;

    setLoading(true);
    setError(null);
    poll();

    intervalRef.current = window.setInterval(poll, intervalMs);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [resultId, poll, intervalMs]);

  return { result, loading, error };
}
