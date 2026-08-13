import path from "node:path";

import dotenv from "dotenv";

let loaded = false;

export function ensureServerEnvLoaded() {
  if (loaded) return;
  loaded = true;

  dotenv.config({ path: path.resolve(process.cwd(), ".env.local"), override: false });
  dotenv.config({ path: path.resolve(process.cwd(), "..", ".env"), override: false });
}
