import fs from "node:fs/promises";
import path from "node:path";

import type { FolderImage, LabelMeAnnotation } from "@/types/annotation";

const IMAGE_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]);
const allowedRoots = new Set<string>();

export async function scanImageFolder(inputDir: string): Promise<FolderImage[]> {
  const root = path.resolve(inputDir);
  const stat = await fs.stat(root);
  if (!stat.isDirectory()) {
    throw new Error(`Not a directory: ${root}`);
  }
  allowRoot(root);

  const files = await collectImages(root);
  return Promise.all(
    files.map(async (filePath) => {
      const annotationPath = replaceExtension(filePath, ".json");
      const [fileStat, dimensions, hasAnnotation] = await Promise.all([
        fs.stat(filePath),
        readImageDimensions(filePath),
        exists(annotationPath),
      ]);
      return {
        path: filePath,
        name: path.basename(filePath),
        annotationPath,
        hasAnnotation,
        width: dimensions?.width ?? null,
        height: dimensions?.height ?? null,
        size: fileStat.size,
        status: hasAnnotation ? "done" : "idle",
      };
    }),
  );
}

export function allowRoot(root: string) {
  allowedRoots.add(path.resolve(root));
}

export function assertAllowedFile(filePath: string) {
  const resolved = path.resolve(filePath);
  const isAllowed = [...allowedRoots].some(
    (root) => resolved === root || resolved.startsWith(root + path.sep),
  );
  if (!isAllowed) {
    throw new Error("Scan the containing folder before reading this file.");
  }
  return resolved;
}

export async function readLocalImage(filePath: string) {
  const resolved = assertAllowedFile(filePath);
  if (!IMAGE_EXTENSIONS.has(path.extname(resolved).toLowerCase())) {
    throw new Error("Unsupported image file.");
  }
  return {
    bytes: await fs.readFile(resolved),
    mimeType: mimeFromExtension(resolved),
  };
}

export async function readAnnotationForImage(imagePath: string): Promise<LabelMeAnnotation | null> {
  const resolved = assertAllowedFile(imagePath);
  const annotationPath = replaceExtension(resolved, ".json");
  if (!(await exists(annotationPath))) {
    return null;
  }
  const payload = JSON.parse(await fs.readFile(annotationPath, "utf-8")) as LabelMeAnnotation;
  return payload;
}

export async function writeAnnotationForImage(
  imagePath: string,
  annotation: LabelMeAnnotation,
): Promise<LabelMeAnnotation> {
  const resolved = assertAllowedFile(imagePath);
  const annotationPath = replaceExtension(resolved, ".json");
  const payload: LabelMeAnnotation = {
    version: annotation.version ?? "4.0.0-beta.5",
    flags: annotation.flags ?? {},
    checked: annotation.checked ?? false,
    shapes: annotation.shapes ?? [],
    imagePath: path.basename(resolved),
    imageData: null,
    imageHeight: annotation.imageHeight,
    imageWidth: annotation.imageWidth,
    description: annotation.description ?? "",
  };
  await fs.writeFile(annotationPath, JSON.stringify(payload, null, 2), "utf-8");
  return payload;
}

export function collectLabelCounts(annotations: Array<LabelMeAnnotation | null>) {
  const counts: Record<string, number> = {};
  for (const annotation of annotations) {
    for (const shape of annotation?.shapes ?? []) {
      counts[shape.label] = (counts[shape.label] ?? 0) + 1;
    }
  }
  return counts;
}

function replaceExtension(filePath: string, extension: string) {
  return path.join(path.dirname(filePath), path.basename(filePath, path.extname(filePath)) + extension);
}

async function collectImages(root: string): Promise<string[]> {
  const entries = await fs.readdir(root, { withFileTypes: true });
  const results: string[] = [];
  for (const entry of entries) {
    const entryPath = path.join(root, entry.name);
    if (entry.isDirectory()) {
      results.push(...(await collectImages(entryPath)));
    } else if (entry.isFile() && IMAGE_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) {
      results.push(entryPath);
    }
  }
  return results.sort((a, b) => a.localeCompare(b));
}

async function exists(filePath: string) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function mimeFromExtension(filePath: string) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === ".png") return "image/png";
  if (ext === ".webp") return "image/webp";
  if (ext === ".bmp") return "image/bmp";
  if (ext === ".tif" || ext === ".tiff") return "image/tiff";
  return "image/jpeg";
}

async function readImageDimensions(filePath: string): Promise<{ width: number; height: number } | null> {
  const handle = await fs.open(filePath, "r");
  try {
    const header = Buffer.alloc(64 * 1024);
    const { bytesRead } = await handle.read(header, 0, header.length, 0);
    const buffer = header.subarray(0, bytesRead);
    const ext = path.extname(filePath).toLowerCase();
    if (ext === ".png") return readPngDimensions(buffer);
    if (ext === ".jpg" || ext === ".jpeg") return readJpegDimensions(buffer);
    if (ext === ".bmp") return readBmpDimensions(buffer);
    return null;
  } finally {
    await handle.close();
  }
}

function readPngDimensions(buffer: Buffer) {
  if (buffer.length < 24 || buffer.toString("ascii", 1, 4) !== "PNG") return null;
  return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20) };
}

function readBmpDimensions(buffer: Buffer) {
  if (buffer.length < 26 || buffer.toString("ascii", 0, 2) !== "BM") return null;
  return { width: Math.abs(buffer.readInt32LE(18)), height: Math.abs(buffer.readInt32LE(22)) };
}

function readJpegDimensions(buffer: Buffer) {
  if (buffer.length < 4 || buffer[0] !== 0xff || buffer[1] !== 0xd8) return null;
  let offset = 2;
  while (offset < buffer.length) {
    if (buffer[offset] !== 0xff) return null;
    const marker = buffer[offset + 1];
    const size = buffer.readUInt16BE(offset + 2);
    if (marker >= 0xc0 && marker <= 0xc3) {
      return { height: buffer.readUInt16BE(offset + 5), width: buffer.readUInt16BE(offset + 7) };
    }
    offset += 2 + size;
  }
  return null;
}
