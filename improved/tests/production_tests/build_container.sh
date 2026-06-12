#!/bin/bash
# Executable script to build and run the circuit knitting Docker container
# Usage: ./build_container.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Building Docker container ==="
echo "Script directory: $SCRIPT_DIR"
echo "Repo root: $REPO_ROOT"

# Build the Docker image
docker build -t circuit-knitting -f "$SCRIPT_DIR/Dockerfile" "$REPO_ROOT"

echo ""
echo "=== Container built successfully ==="
echo ""
echo "To run the container:"
echo "  docker run -it --rm -v \"$REPO_ROOT:/app\" circuit-knitting"
echo ""
echo "To run with PYTHONHASHSEED=0:"
echo "  docker run -it --rm -v \"$REPO_ROOT:/app\" -e PYTHONHASHSEED=0 circuit-knitting"
