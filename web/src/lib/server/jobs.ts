import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { randomUUID } from "node:crypto";

import type { CategoryGuide, JobEvent, JobState, ModelProvider } from "@/types/annotation";

export type CreateJobInput = {
  inputDir: string;
  outputDir?: string;
  labels: string[];
  overwrite: boolean;
  coordinateSystem: "normalized-1000" | "pixel";
  provider?: ModelProvider;
  guides?: CategoryGuide[];
};

type InternalJob = JobState & {
  process?: ChildProcessWithoutNullStreams;
  subscribers: Set<(event: JobEvent) => void>;
};

const jobs = new Map<string, InternalJob>();
const repoRoot = path.resolve(process.cwd(), "..");

export function createAnnotationJob(input: CreateJobInput) {
  const id = randomUUID();
  const job: InternalJob = {
    id,
    status: "queued",
    events: [],
    subscribers: new Set(),
  };
  jobs.set(id, job);

  const python = resolvePython();
  const args = [
    "-m",
    "data_annotation_agent.web_job",
    "--input-dir",
    input.inputDir,
    "--labels",
    ...input.labels,
    "--output-dir",
    input.outputDir || input.inputDir,
    "--coordinate-system",
    input.coordinateSystem,
    "--config",
    path.join(repoRoot, "config.json"),
  ];
  if (input.provider) args.push("--provider", input.provider);
  if (input.overwrite) args.push("--overwrite");
  const guidesPath = input.guides?.length ? writeGuidesFile(id, input.guides) : null;
  if (guidesPath) args.push("--guides-json", guidesPath);

  const child = spawn(python, args, { cwd: repoRoot, env: pythonEnv() });
  job.process = child;
  job.status = "running";
  pushEvent(job, { type: "job_spawned" });

  let stdoutBuffer = "";
  child.stdout.on("data", (chunk: Buffer) => {
    stdoutBuffer += chunk.toString("utf-8");
    const lines = stdoutBuffer.split(/\r?\n/);
    stdoutBuffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const event = JSON.parse(line) as JobEvent;
        if (event.type === "job_finished" || event.type === "job_summary") {
          job.status = "finished";
          job.summary = event.summary;
        }
        if (event.type === "job_error") {
          job.status = "failed";
          job.error = event.error ?? "Unknown job error";
        }
        pushEvent(job, event);
      } catch {
        pushEvent(job, { type: "log", error: line });
      }
    }
  });

  child.stderr.on("data", (chunk: Buffer) => {
    pushEvent(job, { type: "stderr", error: chunk.toString("utf-8").trim() });
  });

  child.on("exit", (code) => {
    if (guidesPath) safeUnlink(guidesPath);
    if (job.status === "running" || job.status === "queued") {
      job.status = code === 0 ? "finished" : "failed";
    }
    pushEvent(job, { type: "process_exit", status: job.status, error: code ? `exit ${code}` : null });
  });

  child.on("error", (error) => {
    if (guidesPath) safeUnlink(guidesPath);
    job.status = "failed";
    job.error = error.message;
    pushEvent(job, { type: "job_error", error: error.message });
  });

  return getJobSnapshot(id);
}

export async function generateCategoryGuides(input: Pick<CreateJobInput, "inputDir" | "outputDir" | "labels">) {
  const python = resolvePython();
  const args = [
    "-m",
    "data_annotation_agent.web_job",
    "--input-dir",
    input.inputDir,
    "--labels",
    ...input.labels,
    "--output-dir",
    input.outputDir || input.inputDir,
    "--config",
    path.join(repoRoot, "config.json"),
    "--research-only",
  ];

  return new Promise<CategoryGuide[]>((resolve, reject) => {
    const child = spawn(python, args, { cwd: repoRoot, env: pythonEnv() });
    let stdoutBuffer = "";
    let stderrBuffer = "";
    let guides: CategoryGuide[] | null = null;

    child.stdout.on("data", (chunk: Buffer) => {
      stdoutBuffer += chunk.toString("utf-8");
      const lines = stdoutBuffer.split(/\r?\n/);
      stdoutBuffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const event = JSON.parse(line) as JobEvent;
          if (event.type === "research_done" && event.guides?.length) {
            guides = event.guides;
          }
          if (event.type === "job_error") {
            reject(new Error(event.error ?? "类别指南生成失败"));
          }
        } catch {
          // Ignore plain log lines from Python dependencies.
        }
      }
    });

    child.stderr.on("data", (chunk: Buffer) => {
      stderrBuffer += chunk.toString("utf-8");
    });

    child.on("error", (error) => reject(error));
    child.on("exit", (code) => {
      if (code !== 0) {
        reject(new Error(stderrBuffer.trim() || `类别指南生成失败：exit ${code}`));
        return;
      }
      resolve(guides ?? []);
    });
  });
}

export function getJobSnapshot(id: string) {
  const job = jobs.get(id);
  if (!job) return null;
  return {
    id: job.id,
    status: job.status,
    events: job.events,
    summary: job.summary,
    error: job.error,
  } satisfies JobState;
}

export function subscribeToJob(id: string, subscriber: (event: JobEvent) => void) {
  const job = jobs.get(id);
  if (!job) return null;
  job.subscribers.add(subscriber);
  return () => {
    job.subscribers.delete(subscriber);
  };
}

export function cancelJob(id: string) {
  const job = jobs.get(id);
  if (!job) return null;
  job.status = "cancelled";
  job.process?.kill();
  pushEvent(job, { type: "job_cancelled" });
  return getJobSnapshot(id);
}

function pushEvent(job: InternalJob, event: JobEvent) {
  job.events.push(event);
  if (job.events.length > 500) job.events = job.events.slice(-500);
  for (const subscriber of job.subscribers) {
    subscriber(event);
  }
}

function resolvePython() {
  const venvPython = path.join(repoRoot, ".venv", "Scripts", "python.exe");
  return fs.existsSync(venvPython) ? venvPython : "python";
}

function writeGuidesFile(jobId: string, guides: CategoryGuide[]) {
  const filePath = path.join(os.tmpdir(), `data-annotation-guides-${jobId}.json`);
  fs.writeFileSync(filePath, JSON.stringify({ guides }, null, 2), "utf-8");
  return filePath;
}

function safeUnlink(filePath: string) {
  try {
    fs.unlinkSync(filePath);
  } catch {
    // Temporary guide files are best-effort cleanup.
  }
}

function pythonEnv() {
  return {
    ...process.env,
    PYTHONIOENCODING: "utf-8",
    PYTHONUTF8: "1",
  };
}
