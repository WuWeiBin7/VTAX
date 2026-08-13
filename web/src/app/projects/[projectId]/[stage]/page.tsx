import { ProjectWorkspace } from "@/components/project-workspace";
import { VTAX_STAGES, type VtaxStageName } from "@/types/vtax";
import { redirect } from "next/navigation";

export default async function ProjectStagePage({
  params,
}: {
  params: Promise<{ projectId: string; stage: string }>;
}) {
  const { projectId, stage } = await params;
  if (stage === "definition") redirect(`/projects/${projectId}/data`);
  if (stage === "review") redirect(`/projects/${projectId}/annotation`);
  if (stage === "configuration") redirect(`/projects/${projectId}/training`);
  const normalized = VTAX_STAGES.includes(stage as VtaxStageName) ? (stage as VtaxStageName) : "data";
  return <ProjectWorkspace projectId={projectId} stageName={normalized} />;
}
