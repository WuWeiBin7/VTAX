import fs from "node:fs";
import path from "node:path";

import type { ModelProvider, ModelProviderSummary } from "@/types/annotation";

const repoRoot = path.resolve(process.cwd(), "..");
const defaultOllamaBaseUrl = "http://127.0.0.1:11434/v1";

export function readModelProviderSummary(): ModelProviderSummary {
  const config = readConfigFile();
  const provider = normalizeProvider(readString(config.provider) ?? "dashscope");
  const ollamaBaseUrl = readString(config.ollama_base_url) ?? process.env.OLLAMA_BASE_URL ?? defaultOllamaBaseUrl;
  const ollamaVisionModel = readString(config.ollama_vision_model) ?? process.env.OLLAMA_VISION_MODEL ?? null;

  return {
    provider,
    dashscopeConfigured: Boolean(readString(config.api_key) ?? process.env.DASHSCOPE_API_KEY),
    ollamaConfigured: Boolean(ollamaBaseUrl && ollamaVisionModel),
    ollamaBaseUrl,
    ollamaVisionModel,
  };
}

function readConfigFile(): Record<string, unknown> {
  const configPath = path.join(repoRoot, "config.json");
  if (!fs.existsSync(configPath)) return {};
  const payload = JSON.parse(fs.readFileSync(configPath, "utf-8")) as unknown;
  return payload && typeof payload === "object" && !Array.isArray(payload)
    ? (payload as Record<string, unknown>)
    : {};
}

function normalizeProvider(value: string): ModelProvider {
  return value.trim().toLowerCase() === "ollama" ? "ollama" : "dashscope";
}

function readString(value: unknown) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}
