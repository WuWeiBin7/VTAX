import { NextResponse } from "next/server";

import { readLocalImage } from "@/lib/server/local-files";

export const runtime = "nodejs";

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const filePath = url.searchParams.get("path");
    if (!filePath) {
      return NextResponse.json({ error: "path is required" }, { status: 400 });
    }
    const { bytes, mimeType } = await readLocalImage(filePath);
    return new Response(new Uint8Array(bytes), {
      headers: {
        "content-type": mimeType,
        "cache-control": "no-store",
      },
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to read image" },
      { status: 400 },
    );
  }
}
