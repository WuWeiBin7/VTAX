import { NextResponse } from "next/server";

import { readModelProviderSummary } from "@/lib/server/model-config";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return NextResponse.json(readModelProviderSummary());
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to read model provider config" },
      { status: 500 },
    );
  }
}
