import { NextResponse } from "next/server";

import { browseServerFolders, validateServerFolder } from "@/lib/server/server-folders";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const targetPath = searchParams.get("path") ?? undefined;
    return NextResponse.json(await browseServerFolders(targetPath));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to browse server folders" },
      { status: 400 },
    );
  }
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as {
      path?: string;
      purpose?: "annotation" | "video";
    };
    if (!body.path) {
      return NextResponse.json({ error: "path is required" }, { status: 400 });
    }
    return NextResponse.json(await validateServerFolder(body.path, body.purpose));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to validate server folder" },
      { status: 400 },
    );
  }
}
