#!/usr/bin/env bash
# Launcher for single-process stages:  scripts/ocrfix.sh <command> [args...]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
exec ocrfix "$@"
