import fs from "node:fs/promises";
import path from "node:path";

import { ensureServerEnvLoaded } from "./load-env";

const IMAGE_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]);

export type ServerFolderEntry = {
  name: string;
  path: string;
};

export type ServerFolderBrowseResult = {
  roots: ServerFolderEntry[];
  currentPath: string | null;
  parentPath: string | null;
  directories: ServerFolderEntry[];
};

export type ServerFolderValidationResult = {
  resolvedPath: string;
  exists: boolean;
  isDirectory: boolean;
  readable: boolean;
  writable: boolean;
  imageCount?: number;
};

export async function browseServerFolders(targetPath?: string): Promise<ServerFolderBrowseResult> {
  const roots = await listAvailableRoots();
  if (!roots.length) {
    throw new Error("No accessible server folders were found.");
  }

  if (!targetPath) {
    if (process.platform !== "win32" && roots.length === 1) {
      return browseResolvedDirectory(roots[0].path, roots);
    }
    return {
      roots,
      currentPath: null,
      parentPath: null,
      directories: [],
    };
  }

  const resolvedPath = await resolveBrowsableDirectory(targetPath);
  return browseResolvedDirectory(resolvedPath, roots);
}

export async function validateServerFolder(
  targetPath: string,
  purpose: "annotation" | "video" = "annotation",
): Promise<ServerFolderValidationResult> {
  const resolvedPath = await resolveBrowsableDirectory(targetPath);
  const stat = await fs.stat(resolvedPath);
  if (!stat.isDirectory()) {
    throw new Error(`Not a directory: ${resolvedPath}`);
  }

  let writable = false;
  try {
    const probeDir = path.join(resolvedPath, ".daa-write-probe");
    await fs.mkdir(probeDir, { recursive: true });
    await fs.rmdir(probeDir);
    writable = true;
  } catch {
    writable = false;
  }

  const result: ServerFolderValidationResult = {
    resolvedPath,
    exists: true,
    isDirectory: true,
    readable: true,
    writable,
  };

  if (purpose === "annotation") result.imageCount = await countImages(resolvedPath);

  return result;
}

async function browseResolvedDirectory(
  resolvedPath: string,
  roots: ServerFolderEntry[],
): Promise<ServerFolderBrowseResult> {
  const stat = await fs.stat(resolvedPath);
  if (!stat.isDirectory()) {
    throw new Error(`Not a directory: ${resolvedPath}`);
  }

  const entries = await fs.readdir(resolvedPath, { withFileTypes: true });
  const directories = entries
    .filter((entry) => entry.isDirectory())
    .map((entry) => ({
      name: entry.name,
      path: path.join(resolvedPath, entry.name),
    }))
    .sort((left, right) => left.name.localeCompare(right.name, "zh-CN"));

  return {
    roots,
    currentPath: resolvedPath,
    parentPath: parentDirectory(resolvedPath, roots),
    directories,
  };
}

async function resolveBrowsableDirectory(targetPath: string) {
  const resolved = path.resolve(targetPath);
  const stat = await fs.stat(resolved);
  if (!stat.isDirectory()) {
    throw new Error(`Not a directory: ${resolved}`);
  }

  const allowedRoots = await getAllowedRoots();
  if (!allowedRoots.length) return resolved;

  const isAllowed = allowedRoots.some((root) => resolved === root || resolved.startsWith(root + path.sep));
  if (!isAllowed) {
    throw new Error(`Folder is outside allowed roots: ${resolved}`);
  }
  return resolved;
}

async function listAvailableRoots(): Promise<ServerFolderEntry[]> {
  const configuredRoots = await getAllowedRoots();
  if (configuredRoots.length) {
    return configuredRoots.map((root) => ({
      name: root,
      path: root,
    }));
  }

  if (process.platform === "win32") {
    const roots: ServerFolderEntry[] = [];
    for (let code = 67; code <= 90; code += 1) {
      const drive = `${String.fromCharCode(code)}:\\`;
      try {
        const stat = await fs.stat(drive);
        if (stat.isDirectory()) {
          roots.push({ name: drive, path: drive });
        }
      } catch {
        // Ignore missing drive letters.
      }
    }
    return roots;
  }

  return [{ name: "/", path: "/" }];
}

async function getAllowedRoots() {
  ensureServerEnvLoaded();
  const configured = parseAllowedRoots(process.env.SERVER_FOLDER_ROOTS);
  if (!configured.length) return [];

  const unique = new Set<string>();
  for (const root of configured) {
    try {
      const resolved = path.resolve(root);
      const stat = await fs.stat(resolved);
      if (stat.isDirectory()) unique.add(resolved);
    } catch {
      // Skip invalid configured roots.
    }
  }
  return [...unique].sort((left, right) => left.localeCompare(right, "zh-CN"));
}

function parseAllowedRoots(value?: string) {
  if (!value?.trim()) return [];
  return value
    .split(/[\r\n;,]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function parentDirectory(resolvedPath: string, roots: ServerFolderEntry[]) {
  if (roots.some((root) => root.path === resolvedPath)) return null;
  const parsed = path.parse(resolvedPath);
  if (resolvedPath === parsed.root) return null;
  const parent = path.dirname(resolvedPath);
  const allowed = roots.some((root) => parent === root.path || parent.startsWith(root.path + path.sep));
  return parent === resolvedPath || !allowed ? null : parent;
}

async function countImages(root: string): Promise<number> {
  const entries = await fs.readdir(root, { withFileTypes: true });
  let total = 0;
  for (const entry of entries) {
    const entryPath = path.join(root, entry.name);
    if (entry.isDirectory()) {
      total += await countImages(entryPath);
      continue;
    }
    if (entry.isFile() && IMAGE_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) {
      total += 1;
    }
  }
  return total;
}
