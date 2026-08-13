import { NextResponse } from "next/server";

import { createAnnotationJob } from "@/lib/server/jobs";
import type { CategoryGuide, ModelProvider } from "@/types/annotation";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as {
      inputDir?: string;
      outputDir?: string;
      labels?: string[];
      boxType?: string;
      overwrite?: boolean;
      coordinateSystem?: "normalized-1000" | "pixel";
      provider?: ModelProvider;
      guides?: CategoryGuide[];
    };
    if (!body.inputDir) {
      return NextResponse.json({ error: "inputDir is required" }, { status: 400 });
    }
    if (!body.labels?.length) {
      return NextResponse.json({ error: "At least one label is required" }, { status: 400 });
    }
    if (body.boxType && body.boxType !== "rectangle") {
      return NextResponse.json({ error: "Only rectangle boxes are supported" }, { status: 400 });
    }

    const job = createAnnotationJob({
      inputDir: body.inputDir,
      outputDir: body.outputDir || body.inputDir,
      labels: body.labels,
      overwrite: Boolean(body.overwrite),
      coordinateSystem: body.coordinateSystem ?? "normalized-1000",
      provider: body.provider,
      guides: body.guides,
    });
    return NextResponse.json(job);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to create job" },
      { status: 500 },
    );
  }
}
