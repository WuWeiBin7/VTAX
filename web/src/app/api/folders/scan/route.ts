import { NextResponse } from "next/server";

import {
  collectLabelCounts,
  readAnnotationForImage,
  scanImageFolder,
} from "@/lib/server/local-files";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as { inputDir?: string };
    if (!body.inputDir) {
      return NextResponse.json({ error: "inputDir is required" }, { status: 400 });
    }

    const images = await scanImageFolder(body.inputDir);
    const annotations = await Promise.all(images.map((image) => readAnnotationForImage(image.path)));
    return NextResponse.json({
      inputDir: body.inputDir,
      images,
      labelCounts: collectLabelCounts(annotations),
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to scan folder" },
      { status: 400 },
    );
  }
}
