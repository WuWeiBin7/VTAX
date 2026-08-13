import { NextResponse } from "next/server";

import { readAnnotationForImage, writeAnnotationForImage } from "@/lib/server/local-files";
import type { LabelMeAnnotation } from "@/types/annotation";

export const runtime = "nodejs";

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const filePath = url.searchParams.get("path");
    if (!filePath) {
      return NextResponse.json({ error: "path is required" }, { status: 400 });
    }
    const annotation = await readAnnotationForImage(filePath);
    return NextResponse.json({ annotation });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to read annotation" },
      { status: 400 },
    );
  }
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as {
      imagePath?: string;
      annotation?: LabelMeAnnotation;
    };
    if (!body.imagePath || !body.annotation) {
      return NextResponse.json({ error: "imagePath and annotation are required" }, { status: 400 });
    }
    const annotation = await writeAnnotationForImage(body.imagePath, body.annotation);
    return NextResponse.json({ annotation });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to save annotation" },
      { status: 400 },
    );
  }
}
