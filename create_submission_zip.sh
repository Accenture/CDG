#!/bin/bash
# Create anonymous zip for paper supplementary materials
#
# Excludes (aligned with .gitignore):
#   - .git/ : contains author info and commit history
#   - deepconf/*, gpt-oss/* : git submodule contents
#   - results/ : very large directory with cached outputs
#   - __pycache__/, *.pyc, *.pyo : compiled Python files
#   - .ipynb_checkpoints/ : Jupyter checkpoints
#   - .env, .venv/, venv/ : environment files
#   - *_api_key.txt : API key files
#   - .idea/, .vscode/ : IDE configs
#   - .DS_Store, Thumbs.db : OS metadata files

# Clean up existing zip
rm -f supplymentary_12618.zip

zip -r supplymentary_12618.zip . \
    -x "*.git*" \
    -x "deepconf/*" \
    -x "gpt-oss/*" \
    -x "results/*" \
    -x "*__pycache__*" \
    -x "*.pyc" \
    -x "*.pyo" \
    -x "*.py[cod]" \
    -x "*\$py.class" \
    -x ".ipynb_checkpoints/*" \
    -x ".env" \
    -x ".venv/*" \
    -x "venv/*" \
    -x "*_api_key.txt" \
    -x ".idea/*" \
    -x ".vscode/*" \
    -x "*.swp" \
    -x "*.swo" \
    -x ".DS_Store" \
    -x "Thumbs.db" \
    -x "supplymentary_12618.zip"

echo "Created supplymentary_12618.zip"
