#!/bin/sh
set -e

echo "Running black..."
black ./

echo "Running isort..."
isort ./

echo "Running ruff..."
ruff check ./

echo "All checks passed."
