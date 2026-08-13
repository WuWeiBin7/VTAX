import { NextResponse } from "next/server";

import { generateCategoryGuides } from "@/lib/server/jobs";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as {
      inputDir?: string;
      outputDir?: string;
      labels?: string[];
    };
    if (!body.inputDir) {
      return NextResponse.json({ error: "inputDir is required" }, { status: 400 });
    }
    if (!body.labels?.length) {
      return NextResponse.json({ error: "At least one label is required" }, { status: 400 });
    }

    const guides = await generateCategoryGuides({
      inputDir: body.inputDir,
      outputDir: body.outputDir || body.inputDir,
      labels: body.labels,
    });
    return NextResponse.json({ guides });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to generate category guides" },
      { status: 500 },
    );
  }
}
