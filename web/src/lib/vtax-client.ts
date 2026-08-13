import type { VtaxJob, VtaxProject } from "@/types/vtax";

const ROOT = "/api/vtax/api/v1";

export async function vtaxRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${ROOT}${path}`, {
    ...init,
    cache: "no-store",
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  const payload = (await response.json()) as T & { detail?: string };
  if (!response.ok) throw new Error(payload.detail ?? `VTAX request failed: ${response.status}`);
  return payload;
}

export function createProject(input: {
  name: string;
  description: string;
  labels: string[];
  task_type: "detection";
  start_stage: string;
}) {
  return vtaxRequest<VtaxProject>("/projects", { method: "POST", body: JSON.stringify(input) });
}

export function runStage(projectId: string, stage: string, config: Record<string, unknown>) {
  return vtaxRequest<VtaxJob>(`/projects/${projectId}/stages/${stage}/run`, {
    method: "POST",
    body: JSON.stringify({ config }),
  });
}

export function completeStage(
  projectId: string,
  stage: string,
  metadata: Record<string, unknown>,
  force = false,
) {
  return vtaxRequest<VtaxProject>(`/projects/${projectId}/stages/${stage}/complete`, {
    method: "POST",
    body: JSON.stringify({ metadata, force }),
  });
}
