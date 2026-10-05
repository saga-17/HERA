import type {
  AskResponse,
  HealthResponse,
  HeraResult,
  UploadResponse,
} from "./types";

const API_BASE = "/api";

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || "Request failed");
  }
  return response.json();
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  return handleResponse<HealthResponse>(res);
}

export async function uploadImage(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/upload-image`, {
    method: "POST",
    body: formData,
  });
  return handleResponse<UploadResponse>(res);
}

export async function askQuestion(
  imageId: string,
  question: string
): Promise<AskResponse> {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_id: imageId, question }),
  });
  return handleResponse<AskResponse>(res);
}

export async function cancelInference(resultId: string): Promise<{ result_id: string; status: string }> {
  const res = await fetch(`${API_BASE}/cancel/${resultId}`, {
    method: "POST",
  });
  return handleResponse<{ result_id: string; status: string }>(res);
}

export async function getResult(resultId: string): Promise<HeraResult> {
  const res = await fetch(`${API_BASE}/result/${resultId}`);
  return handleResponse<HeraResult>(res);
}

export function getImageUrl(imageId: string): string {
  return `${API_BASE}/images/${imageId}`;
}
