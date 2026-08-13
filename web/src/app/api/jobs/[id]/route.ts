import { NextResponse } from "next/server";

import { getJobSnapshot } from "@/lib/server/jobs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(_: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  const snapshot = getJobSnapshot(id);
  if (!snapshot) {
    return NextResponse.json({ error: "job not found" }, { status: 404 });
  }
  return NextResponse.json(snapshot, {
    headers: { "Cache-Control": "no-store, max-age=0" },
  });
}
