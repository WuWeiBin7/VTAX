#!/usr/bin/env bash
set -euo pipefail

vtax_root=/root/WWB/VTAX
repo_root="$vtax_root/Data_Annotation_Agent"
conda_bin=/root/miniconda3/bin/conda

if [[ "$(id -u)" != "0" ]]; then
  echo "Run this installer as root on Ubuntu 22.04." >&2
  exit 1
fi
if ! command -v node >/dev/null 2>&1 || [[ "$(node -p 'process.versions.node.split(`.`)[0]')" != "22" ]]; then
  echo "Node.js 22 LTS must be installed before running this installer." >&2
  exit 1
fi
if [[ ! -x "$conda_bin" ]]; then
  echo "Miniconda was not found at $conda_bin" >&2
  exit 1
fi

mkdir -p \
  "$vtax_root/workspaces" \
  "$vtax_root/LabeledDataset" \
  "$vtax_root/Weight" \
  "$vtax_root/ONNX" \
  "$vtax_root/var/torch-kernel-cache" \
  "$vtax_root/models/dinov3" \
  "$vtax_root/models/yolo26"

if ! "$conda_bin" env list | awk '{print $1}' | grep -qx vtax; then
  "$conda_bin" create -y -n vtax python=3.10
fi
"$conda_bin" run -n vtax python -m pip install -e "$repo_root[test]"

cd "$repo_root"
npm ci
npm install --no-save -w web @tailwindcss/oxide-linux-x64-gnu@4.3.0 lightningcss-linux-x64-gnu@1.32.0
npm run lint:web
npm run build:web

if [[ ! -f "$repo_root/config.json" ]]; then
  cp "$repo_root/config_example.json" "$repo_root/config.json"
  echo "Created $repo_root/config.json; configure DashScope or Ollama before pre-annotation." >&2
fi
if [[ ! -f "$vtax_root/.env.vtax" ]]; then
  cp "$repo_root/deploy/vtax.env.example" "$vtax_root/.env.vtax"
fi
chmod 600 "$vtax_root/.env.vtax" "$repo_root/config.json"

install -m 0644 "$repo_root/deploy/vtax-worker.service" /etc/systemd/system/vtax-worker.service
install -m 0644 "$repo_root/deploy/vtax-web.service" /etc/systemd/system/vtax-web.service
systemctl daemon-reload
systemctl enable --now vtax-worker.service vtax-web.service
