# Data Annotation Agent Web

This directory contains the local Next.js + shadcn/ui frontend for Data Annotation Agent.

It can:

- scan a local image folder by absolute path;
- preview images through local API routes;
- read existing LabelMe JSON files and draw rectangle overlays;
- start the Python annotation backend;
- stream per-image progress, time, shape counts, token usage, and final summary through SSE.

See [LOCAL_RUN.md](./LOCAL_RUN.md) for setup and launch steps.
